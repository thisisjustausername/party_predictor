
import json
import os

import numpy as np
import torch

import bert.parameters as params
from bert.data_set import ds_path, split_to_loader
from bert.datatypes import BertClsModel
from bert.evl import clean_eval, evaluate_model

model_name = 'model_3_latest.pth'

random_state = np.random.RandomState(params.seed)

predictions = []

model = BertClsModel()
model.load_state_dict(torch.load(os.path.join(params.repo_base_path, f'finetuned_models/{model_name}')))
model = model.to(params.device)

test = split_to_loader(ds_path('datasplits.json'), shuffle=(True, False, False))['test'] # type: ignore

res, label_data = evaluate_model(model, test)
result = clean_eval(res)
print(json.dumps(result, indent=4))
