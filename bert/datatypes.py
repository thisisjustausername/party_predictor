'''
Datatypes for the NER task.
'''

import json
import os

import torch
import transformers
from torch.nn import Linear, Module
from torch.utils.data import Dataset
from torchcrf import CRF

from bert import parameters as params

with open(os.path.join(params.model_base_path, params.mdl, 'config.json'), 'r') as f:
    config = json.load(f)
hidden_size = config['hidden_size']
num_tags = len(params.ner_labels)

class NERDataset(Dataset):

    def __init__(self, encodings, labels: list | None, tokens: list | None=None) -> None:
        self.encodings = encodings
        self.labels = labels
        self.tokens = tokens

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index], dtype=torch.long, device=params.device) if self.labels is not None else None
        item = {'attention_mask': self.encodings[index]['attention_mask'].clone().detach(),
                'input_ids': self.encodings[index]['input_ids'].clone().detach()}
        if self.labels is None and self.tokens is not None:
            return item, self.tokens[index]
        if self.labels is None:
            return item
        if self.tokens is not None:
            return item, label, self.tokens[index]
        return item, label

    def __len__(self) -> int:
        return len(self.encodings)


class BertNerModel(Module):

    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(os.path.join(params.model_base_path, params.mdl))
        # self.dropout = torch.nn.Dropout(0.1)
        self.linear = Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)

    def forward(self, inputs):
        outputs = self.bert(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        logits = self.linear(last_hidden_state)
        return logits
