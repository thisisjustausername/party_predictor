import json
import os

import pandas as pd
import torch

import bert.parameters as params
from bert.data_set import do_all, ds_path
from bert.datatypes import BertNerModel

model_name = 'model_99'

paths = [ds_path('test/de.json'), ds_path('test/fr.json'), ds_path('test/en.json')]

raw_data = []
for path in paths:
    with open(path, 'r') as f:
        raw_data.append(json.load(f))
articles = [len(i) for i in raw_data]
article_mapping = []
num = 0
for index, i in enumerate(raw_data):
    for j in range(len(i)):
        article_mapping.append((index, j, num))
        num += 1

test = do_all(paths, split=params.input_length, create_y=False, shuffle=False)


predictions = []

with open(os.path.join(params.repo_base_path, f'finetuning/stats/{model_name}.json'), 'r') as f:
    stats = json.load(f)
batch_size = stats['batch_size']

model = BertNerModel()
model.load_state_dict(torch.load(os.path.join(params.repo_base_path, f'finetuning/ft/{model_name}')))
model = model.to(params.device)
labels = []
idx = 0
seen_by_doc = {}
with torch.no_grad():
    model.eval()
    for X in iter(test):
        X = {k: v.to(params.device) for k, v in X.items()}
        y_probs = model(X)
        y_preds = torch.argmax(y_probs, dim=-1).clone().detach()

        for row in range(y_preds.shape[0]):
            word_ids = test.dataset.encodings[idx]['word_ids']
            doc_idx = test.dataset.encodings[idx]['sample_map']
            seen = seen_by_doc.setdefault(doc_idx, set())
            for pos, wid in enumerate(word_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                labels.append(params.ner_labels[y_preds[row, pos].item()])
            idx += 1

data = []
for path in paths:
    with open(path, 'r') as f:
        d = json.load(f)
    data.extend(d)
clean_data = [params.clean_tokens(i['tokens']) for i in data]
X = [[e['TOKEN'] for e in i] for i in clean_data]
x_flat = [e for i in X for e in i]

labeled = [{'TOKEN': token, 'PRED': pred} for token, pred in zip(x_flat, labels)]

df = pd.DataFrame()
df['TOKEN'] = x_flat
df['PRED'] = labels
pd.options.display.max_rows = None
# print(df)


doc_lengths = [len(i) for i in X]
labels_per_doc = []
offset = 0
for article in doc_lengths:
    labels_per_doc.append(labels[offset:offset + article])
    offset += article

merged = []

for doc_idx, (doc_clean, doc_labels) in enumerate(zip(clean_data, labels_per_doc)):
    pairs = list(zip(doc_clean, doc_labels))
    m = []
    for cleaned_tok, lab in pairs:
        m.append({
            'TOKEN': cleaned_tok['TOKEN'],
            'PRED': lab,
            'ids': cleaned_tok['ids']
        })
    merged.append(m)

for index, labels in enumerate(labels_per_doc):
    cd = clean_data[index]
    language_idx, article_idx = article_mapping[index][:2]
    for cl, lab in zip(cd, labels):
        indices: list = cl['ids']
        lbs = [lab for _ in indices] if len(indices) < 2 or not lab.startswith('B-') else [lab] + ['I-' + lab[2:] for _ in range(len(indices) - 1)]
        for i, l in zip(indices, lbs):
            raw_data[language_idx][article_idx]['tokens'][i]['TAG'] = l

for path, data in zip(paths, raw_data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
print(raw_data)
