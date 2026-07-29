'''
Create datasets for training, validation and testing.
* datasets are created from the json files in the dataset folder
* datasets are tokenized using the BERT tokenizer and the labels are created using the ner_labels_reverse dictionary
* datasets are then returned as DataLoader objects
'''

# TODO: for training no overlap but still render edges unusable because of missing context (avoid óverlaps as they will be duplicates if trained on, maybe it is excluded)

import json
import os
from typing import Annotated

import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import bert.parameters as params
from bert.datatypes import NERDataset

#########################
# INITIALIZE LOCAL PARAMETERS
#########################

train_langs = ['de', 'fr']
val_langs = ['de', 'fr', 'en']
test_langs = ['de', 'fr', 'en']
lang: int = 0

# set up tokenizer
tokenizer = BertTokenizer.from_pretrained(os.path.join(params.model_base_path, params.mdl), do_lower_case=True)
print('Is fast encoder (should be True):', tokenizer.is_fast)


#########################
# CREATE FUNCTIONS
#########################

def ds_path(subpath: str) -> str:
    '''
    create the full path to the dataset file given a subpath
    Args:
        subpath (str): the subpath to the dataset file
    Returns:
        str: the full path to the dataset file
    '''
    return os.path.join(params.dataset_base_path, subpath)


def load_data(paths: list[str], create_y: bool = True) -> tuple[list[list[str]], list[list[str]]] | list[list[str]]:
    '''
    Load train / val / test data from json files and clean the tokens using the preprocessing module.
    For test set create_y to False

    Args:
        paths (list[str]): list of paths to json files
        create_y (bool): whether to create y labels or not

    Returns:
        tuple[list[list[str]], list[list[str]]] | list[list[str]]: X and y if create_y is True, else only X
    '''
    data = []
    for path in paths:
        with open(path, 'r') as f:
            d = json.load(f)
        data.extend(d)
    X = [[e['TOKEN'] for e in i] for i in data]
    y = None
    if create_y:
        y = [[e['TAG'] for e in i] for i in data]
        return X, y
    return X


def tokenize(tokens, max_length: int = 512):
    '''
    Tokenize a list of tokens using the BERT tokenizer.

    Args:
        tokens (list[list[str]] | list[str]): A list of tokens or a list of lists of tokens.
        max_length (int): The maximum length of the tokenized sequence.

    Returns:
        dict: A dictionary containing the tokenized input IDs, attention mask, and alignment data.
    '''
    # if not list of lists (dataset, so set of instances), make it a list of lists as each instance is a list of tokens again
    if not isinstance(tokens[0], list):
        tokens = [tokens]

    # tokenize the tokens using the BERT tokenizer
    result = tokenizer(tokens, return_tensors='pt', add_special_tokens=True,
            return_attention_mask=True, is_split_into_words=True,
            truncation=True, padding=True, max_length=max_length,
            return_overflowing_tokens=True, stride=2*params.stride) # NOTE: don't label the last stride tokens in the first window and not the first stride tokens in the next window (except for the last window)

    num_rows = result['input_ids'].shape[0]          # actual output rows, >= len(tokens) if overflow happened
    sample_map = result['overflow_to_sample_mapping']  # row -> original chunk index

    # create alignment data for each instance in the batch
    alignment_data = [[(t, i) for t, i in zip(result.tokens(i), result.word_ids(i))] for i in range(num_rows)]


    # return the tokenized input IDs, attention mask, and alignment data
    return {
      'alignment_data': alignment_data,
      'attention_mask': result['attention_mask'],
      'input_ids': result['input_ids'],
      'sample_map': sample_map
    }




def label(tokens: list[list[str]] | list[str],
    tags: list[list[str]] | list[str] | None,
    create_y: Annotated[bool, 'Explicit with tags = None'] = True,
    split: int = 512) -> list[dict]:
    '''
    Label the tokens using the provided tags and the BERT tokenizer.

    Args:
        tokens (list[list[str]] | list[str]): a train / val / test set of tokens
        tags (list[list[str]] | list[str] | None): a train / val / test set of tags
        create_y (bool): whether to create y labels or not
        split (int): the maximum length of the tokenized sequence
    Returns:
        list[dict]: a list of dictionaries containing the processed tokens, labels, attention mask, and input IDs.
    '''
    if tags is not None and create_y is False:
        raise ValueError('create_y must be True if tags are provided')
    if tags is None and create_y is True:
        raise ValueError('create_y must be False if tags are not provided')

    # tokenize tokens using BERT tokenizer
    output = tokenize(tokens, max_length=split)
    # fetch the important outputs
    result = output['alignment_data']
    attention_mask = output['attention_mask']
    input_ids = output['input_ids']
    sample_map = output['sample_map']
    row_tags = [tags[int(sample_map[row_idx])] for row_idx in range(len(result))] if tags is not None else None
    # initialize new data list to store the processed tokens and labels
    new_data = []

    # TODO: optimize
    # for each instance (sublist) in the batch, process the tokens and labels
    max_chunk_idx = len(result) - 1
    iterator = iter(zip(row_tags, result, attention_mask, input_ids, sample_map)) if create_y else iter(zip(result, attention_mask, input_ids, sample_map)) # type: ignore
    for idx, i in enumerate(iterator):
        if tags is not None:
            ner_tags, new_tokens, mask, ids, sample_map = i # type: ignore
        else:
            new_tokens, mask, ids, sample_map = i # type: ignore
            ner_tags = None

        # convert to numpy array for easier processing
        data = np.array(new_tokens)
        ntokens = data[:, 0].tolist()
        word_ids = data[:, 1].tolist()

        # differentiate between None and subtokens
        if create_y is True:
            nids = []
            for index, i in enumerate(word_ids):
                if (index < params.stride and idx != 0) or (index >= len(word_ids) - params.stride and idx != max_chunk_idx):
                    nids.append(-100)
                    continue
                if (index == 0 or i is None or word_ids[index-1] != i):
                    nids.append(i)
                    continue
                nids.append(-100)
            nlabels = [ner_labels_reverse[ner_tags[i]] if (i is not None and i != -100) else -100 for i in nids] # type: ignore
        else:
            raise NotImplementedError('overlap not added to avoided create_y')
            nlabels = [ner_labels_reverse[ner_tags[i]] if i is not None else -100 for i in word_ids]
            group_list = [i if i is not None else -100 for i in word_ids]


        # create a dictionary for the processed tokens and labels and append it to the new_data list
        res = {
                'new_tokens': ntokens,
                'attention_mask': mask,
                'input_ids': ids,
                'word_ids': word_ids,
                'sample_map': int(sample_map)
            }
        if create_y is True:
            res['new_labels'] = nlabels

        new_data.append(res)

    # return the list of processed tokens and labels
    return new_data


def do_all_labeling(tokens, tags: list | None, create_y: Annotated[bool, 'Explicit with tags = None'] = True, split: int = 512) -> list[dict]:
    '''
    Label the tokens using the provided tags and the BERT tokenizer.

    Args:
        tokens (list[list[str]] | list[str]): a train / val / test set of tokens
        tags (list[list[str]] | list[str]): a train / val / test set of tags
        create_y (bool): whether to create y labels or not
        split (int): the maximum length of the tokenized sequence
    Returns:
        list[dict]: a list of dictionaries containing the processed tokens, labels, attention mask, and input IDs.
    '''
    l = label(tokens, tags, create_y=create_y, split=split)
    res =  [{'tokens': i['new_tokens'], 'attention_mask': i['attention_mask'], 'input_ids': i['input_ids'], 'word_ids': i['word_ids'], 'sample_map': i['sample_map']} | ({'labels': i['new_labels']} if create_y else {}) for i in l]
    return res


def do_all(paths: list[str], split: int = 512, create_y: bool = True, shuffle: bool = False, add_tokens: bool = False):
    res = load_data(paths, create_y=create_y)
    if create_y:
        X, y = res
    else:
        X = res
        y = None
    res = do_all_labeling(X, y, create_y=create_y, split=split)
    if add_tokens:
        tokens = [i['tokens'] for i in res]

    items_list = ['attention_mask', 'input_ids', 'word_ids', 'sample_map']
    res = NERDataset([{k: v for k, v in i.items() if k in items_list} for i in res], [i['labels'] for i in res] if create_y else None, tokens=tokens if add_tokens else None)
    dl = DataLoader(res, batch_size=params.batch_size, shuffle=shuffle)
    return dl


if __name__ == '__main__':
    val_paths = [ds_path(f'dev/{lang}.json') for lang in val_langs]

    val_dl = do_all(val_paths, split=512, create_y=True, shuffle=False)

    print('Validation DataLoader:', val_dl)
