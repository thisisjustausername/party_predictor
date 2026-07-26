'''
Datatypes for the NER task.
'''

import json
import os

import torch
import transformers
from torch.nn import Linear, Module
from torch.utils.data import Dataset

from bert import parameters as params

with open(os.path.join(params.model_base_path, params.mdl, 'config.json'), 'r') as f:
    config = json.load(f)
hidden_size = config['hidden_size']


class DataSet(Dataset):

    def __init__(self, encodings, labels: list | None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index], dtype=torch.float32, device=params.device) if self.labels is not None else None
        item = {'attention_mask': self.encodings[index]['attention_mask'].clone().detach(),
                'input_ids': self.encodings[index]['input_ids'].clone().detach()}
        if self.labels is None:
            return item
        return item, label

    def __len__(self) -> int:
        return len(self.encodings)


class BertModel(Module):

    def __init__(self, use_activation_function: bool = False):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(os.path.join(params.model_base_path, params.mdl))
        self.linear = Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.use_activation_function = use_activation_function

    def forward(self, inputs):
        outputs = self.bert(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        cls_representation = last_hidden_state[:, 0, :]
        logits = self.linear(cls_representation).squeeze(-1)
        if self.use_activation_function:
            logits = self.sigmoid(logits)
        return logits
