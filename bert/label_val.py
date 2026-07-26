'''
Get insights on model performance
'''
import json
import os

import torch

from bert import parameters as params
from bert.data_set import do_all
from bert.datatypes import BertModel
from bert.evl import evaluate

model_name = 'model_1'

with open(os.path.join(params.repo_base_path, f'stats/{model_name}.json'), 'r') as f:
    stats = json.load(f)
batch_size = stats['batch_size']

model = BertModel()
model.load_state_dict(torch.load(os.path.join(params.repo_base_path, f'ft_models/{model_name}')))
model = model.to(params.device)

train, val, test = do_all(path=os.path.join(params.repo_base_path, 'data', 'politifact.json'), train_val_test_split=(0.7, 0.1, 0.2))

mse, information = evaluate(model, test)
print(mse)
print(f"Total number of test samples: {len(information)}")
corr = sum([1 for i in information if i['text_label'] == i['text_prediction']])
print(f"Number of correct predictions: {corr}")
print(f"Percentage of correct predictions: {corr / len(information) * 100:.2f}%")

for i in information[:50]:
    print(f"Label: {i['text_label']}, Predicted: {i['text_prediction']}")
