'''
Finetuning pipeline for BERT model on Classification task.
'''

import contextlib
import json
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup as lin_sched

import bert.parameters as params
from bert.data_set import ds_path, split_to_loader
from bert.datatypes import BertClsModel
from bert.evl import clean_eval, evaluate_model

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
TRIM_PADDING = True
MEM_PROBE = os.environ.get('MEM_PROBE') == '1'

torch.manual_seed(params.seed)
torch.cuda.manual_seed_all(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(params.seed)
random.seed(params.seed)
os.environ['PYTHONHASHSEED'] = str(params.seed)


# Usage for distributed training
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

ddp = world_size > 1
device = torch.device('cuda', local_rank) if ddp else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
if ddp:
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=20))
rank = dist.get_rank() if ddp else 0
is_main = rank == 0
# End usage for distributed training


# train, val, test = do_all(ds_path('protocols_speeches_clean.json'), create_y=True, shuffle=(True, False, False))
# NOTE: uses implicit extraction as dict is ordered the same
train, val, test = split_to_loader(ds_path('datasplits.json'), shuffle=(True, False, False)).values()  # type: ignore

# Usage for distributed training
train_sampler = DistributedSampler(
    train.dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=params.seed
) if ddp else None

if ddp:
    train = DataLoader(
        train.dataset,
        batch_size=params.batch_size,
        sampler=train_sampler,
        collate_fn=train.collate_fn,
        num_workers=train.num_workers,
        pin_memory=train.pin_memory,
    )
# End usage for distributed training


#####################################
# Instantiate the model             #
#####################################

# model = BertClsModel() # COMMENTED OUT FOR DDP USAGE
# model.half()
# model = model.to(device) # COMMENTED OUT FOR DDP USAGE

# Usage for distributed training
raw_model = BertClsModel().to(device)
if ddp:
    raw_model.bert.activation_checkpointing_enable(activation_checkpointing_kwargs={'use_reentrant': False})
model = DDP(raw_model, device_ids=[local_rank], gradient_as_bucket_view=True) if ddp else raw_model

torch.manual_seed(params.seed + rank)
# End usage for distributed training

#####################################
# Training / Fine-tuning the model  #
#####################################

files = [int(i[6:].split('.', 1)[0].split('_', 1)[0]) for i in os.listdir(os.path.join(params.repo_base_path, 'finetuned_model_stats/'))] + [0]
new_name = max(files) + 1

# Usage for distributed training
if ddp:
    t = torch.tensor([new_name], device=device)
    dist.broadcast(t, src=0)
    new_name = int(t.item())
# End usage for distributed training

freeze_layers = 0
'''freeze_layers = 3

for name, p in model.named_parameters():
    if 'encoder.layer' in name:
        layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
        if layer_num < freeze_layers:
            p.requires_grad = False
    elif 'embeddings' in name:
        p.requires_grad = False'''

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in raw_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in raw_model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=params.learning_rate, betas=params.betas, eps=params.epsilon)
loss_fn = torch.nn.CrossEntropyLoss() # ignore_index=-100)

steps_per_epoch = np.ceil(len(train) / params.accum_steps)
total_steps = steps_per_epoch * params.num_epochs
scheduler = lin_sched(
    optimizer=optimizer,
    num_warmup_steps=int(0.06 * total_steps),
    num_training_steps=total_steps
)

if MEM_PROBE:
    model.train()
    dummy = {
        'input_ids': torch.randint(0, 1000, (params.batch_size, params.cutoff), device=device),
        'attention_mask': torch.ones(params.batch_size, params.cutoff, dtype=torch.long, device=device),
    }
    torch.cuda.reset_peak_memory_stats()
    with (model.no_sync() if ddp else contextlib.nullcontext()):  # type: ignore
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            probe_loss = model(dummy).float().mean()
        probe_loss.backward()
    peak = torch.cuda.max_memory_allocated() / 2**30
    # AdamW keeps two fp32 states per parameter, allocated on the first optimizer.step()
    adam_state = sum(p.numel() for p in raw_model.parameters()) * 8 / 2**30
    print(f'[rank {rank}] peak fwd+bwd: {peak:.1f} GiB, + AdamW state ~{adam_state:.1f} GiB, '
          f'GPU total {torch.cuda.get_device_properties(device).total_memory / 2**30:.1f} GiB')
    if ddp:
        dist.destroy_process_group()
    raise SystemExit(0)

best_f1 = -1
best_state = None
losses = []
val_f1_scores = []

# compute the overhang after each epoch that has not been accumulated
overhang = len(train) % params.accum_steps
first_overhang = (len(train) // params.accum_steps) * params.accum_steps

for n in range(params.num_epochs):
    model.train()
    if ddp:
        # Shuffle the data differently at each epoch
        train_sampler.set_epoch(n) # type: ignore
    it = iter(train) # Create the iterator from the training dataset
    epoch_loss = 0 # To keep track of the current epoch's loss

    for index, (X, y) in tqdm(enumerate(it), total=len(train), desc=f'Epoch {n + 1}', disable=not is_main):
        X = {k: v.to(device, non_blocking=True) for k, v in X.items()}
        y = y.to(device, non_blocking=True)
        if ddp and TRIM_PADDING and 'attention_mask' in X:
            longest = int(X['attention_mask'].sum(dim=1).max().item())
            X = {k: (v[:, :longest] if v.dim() == 2 else v) for k, v in X.items()}
        is_step = (index < first_overhang and (index + 1) % params.accum_steps == 0) \
            or (index >= first_overhang and (index + 1 - first_overhang) % overhang == 0)
        ctx = model.no_sync() if ddp and not is_step else contextlib.nullcontext() # type: ignore
        with ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_pred = model(X)
                loss = loss_fn(y_pred, y) / (params.accum_steps if index < first_overhang else overhang)
            loss.backward()
        if is_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad() # NOTE: use at least torch>=2.0.0, otherwise for ddp use zero_grad(set_to_none=True)

        epoch_loss += loss.item() * (params.accum_steps if index < first_overhang else overhang)

    if ddp:
        t = torch.tensor([epoch_loss], device=device)
        dist.all_reduce(t)
        epoch_loss = t.item()

    losses.append(epoch_loss)

    stop = torch.tensor([0], device=device)
    if is_main:
        if ddp:
            torch.cuda.empty_cache()
        raw_model.eval()
        res, _ = evaluate_model(raw_model, val)
        val_f1 = res['overall_f1']
        val_f1_scores.append(val_f1)
        if params.early_stopping_patience is not None and np.argmax(val_f1_scores) < len(val_f1_scores) - params.early_stopping_patience:
            print(f'Early stopping at epoch {n + 1} due to no improvement in validation F1 score for {params.early_stopping_patience} epochs.')
            stop[0] = 1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
            print(f'New best model found at epoch {n + 1} with F1 score on val: {best_f1}')
            torch.save(raw_model.state_dict(), os.path.join(params.repo_base_path, f'finetuned_models/model_{new_name}_best_val_f1_state.pth'))
        print(f'Epoch {n + 1}:  train loss: {epoch_loss},    val F1: {val_f1}')

    if ddp:
        dist.broadcast(stop, src=0)
    if stop.item():
        break

if not is_main:
    dist.destroy_process_group()
    raise SystemExit(0)

# Save model states to .pth (not JSON - tensors are not JSON serializable)
torch.save(raw_model.state_dict(), os.path.join(params.repo_base_path, f'finetuned_models/model_{new_name}_latest.pth'))
# torch.save(best_state, os.path.join(params.repo_base_path, f'finetuned_models/model_{new_name}_best_val_f1_state.pth'))

# Save metadata separately as JSON (optional)
with open(os.path.join(params.repo_base_path, f'finetuned_models/model_{new_name}_metadata.json'), 'w') as f:
    json.dump({'note': f'Model weights saved in model_{new_name}_latest.pth and model_{new_name}_best_val_f1_state.pth', 'best_f1_val': best_f1}, f, indent=4)


# evaluate on test set
res_test_latest, label_data = evaluate_model(raw_model, test)
# test_result = clean_eval(res_test_latest)

raw_model.load_state_dict(best_state) # type: ignore

res, label_data = evaluate_model(raw_model, val)
result = clean_eval(res)

# evaluate on test set
res_test, label_data = evaluate_model(raw_model, test)
# test_result = clean_eval(res_test)

# Save thethe stats of the training
result['model'] = params.mdl
result['input_length'] = params.input_length
result['num_epochs'] = params.num_epochs
result['batch_size'] = params.batch_size
result['accum_steps'] = params.accum_steps
result['effective_batch_size'] = params.batch_size * params.accum_steps * world_size
result['world_size'] = world_size
result['seed'] = params.seed
result['freeze_layers'] = freeze_layers
result['early_stopping_patience'] = params.early_stopping_patience
result['learning_rate'] = params.learning_rate
result['betas'] = params.betas
result['epsilon'] = params.epsilon
result['subtokens'] = -100
result['loss'] = loss_fn.__class__.__name__
result['optimizer'] = optimizer.__class__.__name__
result['scheduler'] = scheduler.__class__.__name__
result['model_layers'] = str(raw_model)
result['train_losses'] = losses
result['overall_test_f1'] = res_test['overall_f1']
result['overall_test_precision'] = res_test['overall_precision']
result['overall_test_recall'] = res_test['overall_recall']
result['overall_test_accuracy'] = res_test['overall_accuracy']
result['overall_latest_test_precision'] = res_test_latest['overall_precision']
result['overall_latest_test_recall'] = res_test_latest['overall_recall']
result['overall_latest_test_accuracy'] = res_test_latest['overall_accuracy']
result['overall_latest_test_f1'] = res_test_latest['overall_f1']

# TODO: remove commented code
# save the model and the stats of the training
# torch.save(model.state_dict(), os.path.join(params.repo_base_path, f'finetuned_models/model_{new_name}_final'))
with open(os.path.join(params.repo_base_path, f'finetuned_model_stats/model_{new_name}_final.json'), 'w') as f:
    json.dump(result, f, indent=4)
print(json.dumps(result, indent=4))

if ddp:
    dist.destroy_process_group()
