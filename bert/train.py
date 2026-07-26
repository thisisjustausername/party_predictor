'''
Finetuning pipeline for BERT model on NER task.
'''

import numpy as np
import random
import os
import torch
from torch import optim
import json
from transformers import get_linear_schedule_with_warmup

from bert.datatypes import BertModel
from bert import parameters as params
from bert.data_set import do_all
from bert.evl import evaluate


random_state = np.random.RandomState(params.seed)
# weights = params.weights.to(params.device)
# TODO: instead of weights use continuous weight function


predictions = []

train, val, test = do_all(path=os.path.join(params.repo_base_path, 'data', 'politifact.json'), train_val_test_split=(0.7, 0.1, 0.2))
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

model = BertModel()
model = model.to(params.device)

#####################################
# Training / Fine-tuning the model  #
#####################################

total_steps = len(train) * params.num_epochs
warmup_steps = int(round(total_steps * 0.1, 0))

optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate, betas=params.betas, eps=params.epsilon)
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.HuberLoss(delta=0.1, reduction='mean')
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

losses = []

for n in range(params.num_epochs):
    model.train()
    it = iter(train)  # Create the iterator from the training dataset
    epoch_loss, steps = 0, 0      # To keep track of the current epoch's loss

    for X, y in it:              # Obtain a tensor X = batch of X-values, y accordingly
        X = {k: v.to(params.device) for k, v in X.items()}
        y = y.to(params.device)
        y_pred = model(X)
        mask = (y != -100)
        filtered_pred = y_pred[mask]
        filtered_y = y[mask].float()
        loss = loss_fn(filtered_pred, filtered_y)
        if len(filtered_y) == 0:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
    losses.append(epoch_loss)
    print(f'Epoch {n + 1}: {epoch_loss}')

result = dict()
mse, information = evaluate(model, val)
result['mse'] = mse
result['model'] = params.mdl
result['input_length'] = params.input_length
result['num_epochs']= params.num_epochs
result['batch_size'] = params.batch_size
result['learning_rate'] = params.learning_rate
result['betas']= params.betas
result['epsilon']= params.epsilon
# result['weights'] = params.weights.tolist()
result['subtokens'] = -100
result['loss'] = loss_fn.__class__.__name__
result['optimizer'] = optimizer.__class__.__name__
result['model_layers'] = str(model)


files = [int(i[6:].split('.')[0]) for i in os.listdir(os.path.join(params.repo_base_path, 'stats/'))] + [0]
new_name = max(files) + 1
torch.save(model.state_dict(), os.path.join(params.repo_base_path, f'ft_models/model_{new_name}'))
with open(os.path.join(params.repo_base_path, f'stats/model_{new_name}.json'), 'w') as f:
    json.dump(result, f, indent=4)
print(json.dumps(result, indent=4))
