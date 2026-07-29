'''
Finetuning pipeline for BERT model on NER task.
'''

import json
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim
from transformers import get_linear_schedule_with_warmup as lin_sched

import bert.parameters as params
from bert.data_set import do_all, ds_path
from bert.datatypes import BertNerModel
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


train = do_all([ds_path('train/de.json'), ds_path('train/fr.json')], split=params.input_length, create_y=True, shuffle=True)
val = do_all([ds_path('dev/de.json'), ds_path('dev/fr.json'), ds_path('dev/en.json')], split=params.input_length, create_y=True, shuffle=False)
german = do_all([ds_path('dev/de.json')], split=params.input_length, create_y=True, shuffle=False)
french = do_all([ds_path('dev/fr.json')], split=params.input_length, create_y=True, shuffle=False)
english = do_all([ds_path('dev/en.json')], split=params.input_length, create_y=True, shuffle=False)


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

model = BertNerModel()
model = model.to(params.device)

#####################################
# Training / Fine-tuning the model  #
#####################################

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=params.learning_rate, betas=params.betas, eps=params.epsilon)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
scheduler = lin_sched(
    optimizer=optimizer,
    num_warmup_steps=int(0.06*len(train)*params.num_epochs),
    num_training_steps=int(len(train)*params.num_epochs)
)

epos = []
plt.plot()

best_f1 = -1
best_state = None

for n in range(params.num_epochs):
    model.train()
    it = iter(train)  # Create the iterator from the training dataset
    epoch_loss, steps = 0, 0      # To keep track of the current epoch's loss

    for X, y in it:              # Obtain a tensor X = batch of X-values, y accordingly
        X = {k: v.to(params.device) for k, v in X.items()}
        y = y.to(params.device)
        y_pred = model(X)  # Have our model with current weights make a prediction
        # outputs should be of shape [batch, sequence, logits] (where sequence values indicate the token indices within one sequence)
        y_pred = torch.permute(y_pred, (0, 2, 1))  # swap the sequence and the logit dimensions
        loss = loss_fn(y_pred, y) # ... sucht that the loss function can take care of the rest for us!
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    model.eval()
    res, _ = evaluate_model(model, val)
    val_f1 = res['overall_f1']
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f'Epoch {n + 1}: {epoch_loss}')

model.load_state_dict(best_state) # type: ignore

res, label_data = evaluate_model(model, val)
result = clean_eval(res)

res_german, _ = evaluate_model(model, german)
result_german = clean_eval(res_german, add_key_name='German_')

res_french, _ = evaluate_model(model, french)
result_french = clean_eval(res_french, add_key_name='French_')

res_english, _ = evaluate_model(model, english)
result_english = clean_eval(res_english, add_key_name='English_')

print(json.dumps(result_german, indent=4))
print(json.dumps(result_french, indent=4))
print(json.dumps(result_english, indent=4))


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


files = [int(i[6:].split('.')[0]) for i in os.listdir(os.path.join(params.repo_base_path, 'finetuning/stats/'))] + [0]
new_name = max(files) + 1
torch.save(model.state_dict(), os.path.join(params.repo_base_path, f'finetuning/ft/model_{new_name}'))
with open(os.path.join(params.repo_base_path, f'finetuning/stats/model_{new_name}.json'), 'w') as f:
    json.dump(result, f, indent=4)
print(json.dumps(result, indent=4))
