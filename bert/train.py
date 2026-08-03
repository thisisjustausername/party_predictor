'''
Finetuning pipeline for BERT model on Classification task.
'''

import json
import os
import random

import numpy as np
import torch
from torch import optim
from transformers import get_linear_schedule_with_warmup as lin_sched

import bert.parameters as params
from bert.data_set import do_all, ds_path
from bert.datatypes import BertClsModel
from bert.evl import evaluate_model

random_state = np.random.RandomState(params.seed)

predictions = []

def clean_eval(eval_res, add_key_name: str | None = None) -> dict:
    '''
    Clean the evaluation results by converting all values to float.

    Args:
        eval_res (dict): The evaluation results to clean.

    Returns:
        dict: The cleaned evaluation results.
    '''
    if add_key_name is None:
        add_key_name = ''
    return {add_key_name + k: {vk: float(vv) for vk, vv in v.items()} if isinstance(v, dict) else float(v) for k, v in eval_res.items()}


train, val, test = do_all(ds_path('protocols_speeches_clean.json'), create_y=True, shuffle=False)


# Always fun with the random seeds ...
# We need to set them such that our results will be replicable.
# (Hint: for an experiment later, you can change the random seed here and check what happens.
# But for now, let's keep the answer to all questions of the universe, 42.)
torch.manual_seed(params.seed)
torch.cuda.manual_seed_all(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(params.seed)
random.seed(params.seed)
os.environ['PYTHONHASHSEED'] = str(params.seed)

#####################################
# Instantiate the model             #
#####################################

model = BertClsModel()
model = model.to(params.device)

#####################################
# Training / Fine-tuning the model  #
#####################################

files = [int(i[6:].split('.')[0]) for i in os.listdir(os.path.join(params.repo_base_path, 'finetuned_model_stats/'))] + [0]
new_name = max(files) + 1

freeze_layers = 6

for name, p in model.named_parameters():
    if 'encoder.layer' in name:
        layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
        if layer_num < freeze_layers:
            p.requires_grad = False
    elif 'embeddings' in name:
        p.requires_grad = False

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=params.learning_rate, betas=params.betas, eps=params.epsilon)
loss_fn = torch.nn.CrossEntropyLoss() # ignore_index=-100)
scheduler = lin_sched(
    optimizer=optimizer,
    num_warmup_steps=int(0.06*len(train)*params.num_epochs),
    num_training_steps=int(len(train)*params.num_epochs)
)

best_f1 = -1
best_state = None

accum_steps = 4
if params.batch_size % accum_steps != 0:
    raise ValueError(f'Batch size {params.batch_size} is not divisible by accumulation steps {accum_steps}.')


for n in range(params.num_epochs):
    model.train()
    it = iter(train)  # Create the iterator from the training dataset
    epoch_loss, steps = 0, 0      # To keep track of the current epoch's loss

    for index, (X, y) in enumerate(it):              # Obtain a tensor X = batch of X-values, y accordingly
        X = {k: v.to(params.device) for k, v in X.items()}
        y = y.to(params.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            y_pred = model(X)
            loss = loss_fn(y_pred, y) / accum_steps
        loss.backward()
        if (index + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accum_steps

    model.eval()
    res, _ = evaluate_model(model, val)
    val_f1 = res['overall_f1']
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(model.state_dict(), os.path.join(params.repo_base_path, f'finetuned_models/model_{new_name}'))
    print(f'Epoch {n + 1}: {epoch_loss}')

model.load_state_dict(best_state) # type: ignore

res, label_data = evaluate_model(model, val)
result = clean_eval(res)

result['model'] = params.mdl
result['input_length'] = params.input_length
result['num_epochs'] = params.num_epochs
result['batch_size'] = params.batch_size
result['learning_rate'] = params.learning_rate
result['betas'] = params.betas
result['epsilon'] = params.epsilon
result['subtokens'] = -100


result['loss'] = loss_fn.__class__.__name__
result['optimizer'] = optimizer.__class__.__name__
result['scheduler'] = scheduler.__class__.__name__
result['model_layers'] = str(model)


torch.save(model.state_dict(), os.path.join(params.repo_base_path, f'finetuned_models/model_{new_name}'))
with open(os.path.join(params.repo_base_path, f'finetuned_model_stats/model_{new_name}.json'), 'w') as f:
    json.dump(result, f, indent=4)
print(json.dumps(result, indent=4))
