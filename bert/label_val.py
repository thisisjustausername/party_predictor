'''
Label the val set using the finetuned model in order to find error sources.
'''

import json
import os

import numpy as np
import pandas as pd

from bert.data_set import do_all, ds_path
from bert.datatypes import BertClsModel
from bert.evl import evaluate_model
from bert.parameters import *

model_name = 'model_99'

val = do_all([ds_path('dev/de.json'), ds_path('dev/fr.json'), ds_path('dev/en.json')], split=input_length, create_y=True, shuffle=False, add_tokens=False)
german = do_all([ds_path('dev/de.json')], split=input_length, create_y=True, shuffle=False, add_tokens=False)
french = do_all([ds_path('dev/fr.json')], split=input_length, create_y=True, shuffle=False, add_tokens=False)
english = do_all([ds_path('dev/en.json')], split=input_length, create_y=True, shuffle=False, add_tokens=False)

validation_sets = {
    'all': val,
    'french': french,
    'german': german,
    'english': english
}

with open(os.path.join(repo_base_path, f'finetuning/stats/{model_name}.json'), 'r') as f:
    stats = json.load(f)
batch_size = stats['batch_size']

model = BertClsModel()
model.load_state_dict(torch.load(os.path.join(repo_base_path, f'finetuning/ft/{model_name}')))
model = model.to(device)

res = {}

def np_to_item(input: dict) -> dict:
    return {k: v.item() if isinstance(v, np.generic) else np_to_item(v) if isinstance(v, dict) else v for k, v in input.items()} if isinstance(input, dict) else input

for key, value in validation_sets.items():
    output = evaluate_model(model, value)[0]
    output = {k: np_to_item(v) for k, v in output.items()}
    res[key] = output

# print(json.dumps(res, indent=4))

df = pd.DataFrame()
lang = []
precision = []
recall = []
accuracy = []
f1 = []
for k, v in res.items():
    lang.append(k)
    precision.append(v['overall_precision'])
    recall.append(v['overall_recall'])
    accuracy.append(v['overall_accuracy'])
    f1.append(v['overall_f1'])
df['language'] = lang
df['precision'] = precision
df['recall'] = recall
df['accuracy'] = accuracy
df['f1'] = f1
# df.sort_values(by='f1', ascending=False)
print(df.head(4))
print()

langs = []
for lang, content in res.items():
    lang_df = pd.DataFrame()
    items = list(content.items())[:5]
    tag = []
    precision = []
    recall = []
    f1 = []
    amount = []
    for k, v in items:
        tag.append(k)
        precision.append(v['precision'])
        recall.append(v['recall'])
        f1.append(v['f1'])
        amount.append(v['number'])
    lang_df['tag'] = tag
    lang_df['precision'] = precision
    lang_df['recall'] = recall
    lang_df['f1'] = f1
    lang_df['amount'] = amount
    print(f'LANGUAGE: {lang}')
    print(lang_df)
    print()


with open(os.path.join(repo_base_path, f'finetuning/validation_stats/{model_name}.json'), 'w') as f:
    json.dump(res, f, indent=4)
