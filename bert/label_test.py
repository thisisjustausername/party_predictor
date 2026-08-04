
import json
import os

import numpy as np
import torch

import bert.parameters as params
from bert.data_set import do_all, ds_path
from bert.datatypes import BertClsModel
from bert.evl import clean_eval, evaluate_model

model_name = 'model_1'

random_state = np.random.RandomState(params.seed)

predictions = []

_, _, test = do_all(ds_path('protocols_speeches_clean.json'), create_y=True, shuffle=False)

with open(os.path.join(params.repo_base_path, f'finetuned_model_stats/{model_name}.json'), 'r') as f:
    stats = json.load(f)
batch_size = stats['batch_size']

model = BertClsModel()
model.load_state_dict(torch.load(os.path.join(params.repo_base_path, f'finetuned_models/{model_name}')))
model = model.to(params.device)

res, label_data = evaluate_model(model, test)
result = clean_eval(res)
print(json.dumps(result, indent=4))
