
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import bert.parameters as params
from bert.data_set import tokenize
from bert.datatypes import BertClsModel

with open(os.path.join(params.repo_base_path, 'datasets/additional_speeches.json'), 'r') as f:
    speeches = json.load(f)

model_name = 'model_6_latest.pth'
dims = 2

random_state = np.random.RandomState(params.seed)

predictions = []

model = BertClsModel()
model.load_state_dict(torch.load(os.path.join(params.repo_base_path, f'finetuned_models/{model_name}')))
model = model.to(params.device)

tokens = tokenize([i['speech'] for i in speeches], cutoff=params.cutoff)
tokens = {k: v.to(params.device) for k, v in tokens.items()}
output = model(tokens)
for speech, author in zip(output, speeches):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(params.ner_labels, speech.cpu().detach().numpy(), color=params.party_colors)
    ax.set_title(f'Predictions for speech from {author["author"]}')
    fig.savefig(os.path.join(params.repo_base_path, f'finetuned_model_visualized/other_speeches/speech_{author["author"]}_{model_name.split(".", 1)[0]}.png'))
    plt.close(fig)
