'''
Datatypes for the Classification task.
'''

import json
import os

import torch
import transformers
from torch.nn import Module
from torch.utils.data import Dataset

from bert import parameters as params

with open(os.path.join(params.model_base_path, params.mdl, 'config.json'), 'r') as f:
    config = json.load(f)
hidden_size = config['hidden_size']
num_tags = len(params.ner_labels)

class ClsDataset(Dataset):
    '''
    Dataset for the Classification task.

    Parameters:
        encodings: The encodings of the input data.
        labels (list | None): The labels of the input data.
    '''
    def __init__(self, encodings, labels: list | None) -> None:
        '''
        Initializes the dataset.

        Args:
            encodings: The encodings of the input data.
            labels (list | None): The labels of the input data.
        '''
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index) -> dict  | tuple[dict, torch.Tensor]:
        label = torch.tensor(self.labels[index], dtype=torch.long, device=params.device) if self.labels is not None else None
        item = {'attention_mask': self.encodings[index]['attention_mask'].clone().detach(),
                'input_ids': self.encodings[index]['input_ids'].clone().detach()}
        if self.labels is None:
            return item
        return item, label # type: ignore

    def __len__(self) -> int:
        return len(self.encodings)


class BertClsModel(Module):
    '''
    BERT model for the Classification task.

    Parameters:
        bert: The BERT model for sequence classification.
        '''
    def __init__(self):
        super().__init__()
        self.bert = transformers.AutoModelForSequenceClassification.from_pretrained(os.path.join(params.model_base_path, params.mdl), num_labels=num_tags)
        # self.dropout = torch.nn.Dropout(0.1)

    def forward(self, inputs):
        outputs = self.bert(**inputs) # , output_hidden_states=True)
        return outputs.logits
