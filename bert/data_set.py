'''
Create datasets for training, validation and testing.
* datasets are created from the json files in the dataset folder
* datasets are tokenized using the BERT tokenizer and the labels are created using the ner_labels_reverse dictionary
* datasets are then returned as DataLoader objects
'''

# TODO: for training no overlap but still render edges unusable because of missing context (avoid óverlaps as they will be duplicates if trained on, maybe it is excluded)

import json
import os

import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split as tts

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
    return os.path.join(params.repo_base_path, params.dataset_base_path, subpath)


def train_test_split(
        X: list[str],
        y: list[np.ndarray | float],
        test_size: float = 0.2,
        val_size: float | None = 0.1,
        random_seed: int = 42
    ) -> tuple:
    '''
    Split the data into training and testing sets, normalize the X-data and fix the y-data

    Args:
        X (list[str]): The X data
        y (list[np.ndarray | float]): classes
        test_size (float): The proportion of the data to use for testing
        val_size (float): The proportion of the data to use for validation
        random_seed (int): The random seed to use for reproducibility
    Returns:
        tuple: A tuple containing the training and testing sets for X and y
    '''
    X_mid, X_test, y_mid, y_test = tts(X, y, test_size=test_size, random_state=random_seed)
    X_train, X_val, y_train, y_val = tts(X_mid, y_mid, test_size=val_size, random_state=random_seed)
    del X_mid, y_mid

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_data(
        path: str,
        create_y: bool = True,
        test_size: float = 0.2,
        val_size: float | None = 0.1,
    ) -> tuple[list[list[str]], list[list[str]]] | list[list[str]]:
    '''
    Load data from json file
    For avoiding test and val set create_y to False

    Args:
        path (str): path to json file
        create_y (bool): whether to create y labels or not

    Returns:
        tuple[list[list[str]], list[list[str]]] | list[list[str]]: X and y if create_y is True, else only X
    '''
    with open(path, 'r') as f:
        data = json.load(f)
    X = [i['speech'] for i in data]
    y = None
    if create_y:
        y = [i['class'] for i in data]
        return X, y
    return X


def tokenize(tokens: list[str] | str):
    '''
    Tokenize a list of tokens using the BERT tokenizer.

    Args:
        tokens (list[str] | str): A list of speeches or a speech.

    Returns:
        dict: A dictionary containing the tokenized input IDs, attention mask, and alignment data.
    '''
    # if not list of lists (dataset, so set of instances), make it a list of lists as each instance is a list of tokens again
    if not isinstance(tokens, list):
        tokens = [tokens]

    # tokenize the tokens using the BERT tokenizer
    result = tokenizer(tokens, return_tensors='pt', add_special_tokens=True,
            return_attention_mask=True, is_split_into_words=False, padding=True)

    num_rows = result['input_ids'].shape[0]

    # create alignment data for each instance in the batch
    alignment_data = [[(t, i) for t, i in zip(result.tokens(i), result.word_ids(i))] for i in range(num_rows)]


    # return the tokenized input IDs, attention mask, and alignment data
    return {
      'alignment_data': alignment_data,
      'attention_mask': result['attention_mask'],
      'input_ids': result['input_ids'],
    }




def label(tokens: list[str] | str,
    tags: list[str] | None) -> list[dict]:
    '''
    Label the tokens using the provided tags and the BERT tokenizer.

    Args:
        tokens (list[list[str]] | list[str]): a train / val / test set of tokens
        tags (list[str] | None): a train / val / test set of class tags
    Returns:
        list[dict]: a list of dictionaries containing the processed tokens, labels, attention mask, and input IDs.
    '''

    # tokenize tokens using BERT tokenizer
    output = tokenize(tokens)
    # fetch the important outputs
    result = output['alignment_data']
    attention_mask = output['attention_mask']
    input_ids = output['input_ids']
    # initialize new data list to store the processed tokens and labels
    new_data = []

    for index, i in enumerate(zip(result, attention_mask, input_ids)):
        new_tokens, mask, ids = i

        # convert to numpy array for easier processing
        data = np.array(new_tokens)
        ntokens = data[:, 0].tolist()
        word_ids = data[:, 1].tolist()

        # create a dictionary for the processed tokens and labels and append it to the new_data list
        res = {
                'new_tokens': ntokens,
                'attention_mask': mask,
                'input_ids': ids,
                'word_ids': word_ids,
            }
        if tags is not None:
            res['class'] = tags[index]

        new_data.append(res)

    # return the list of processed tokens and labels
    return new_data


def do_all_labeling(tokens, tags: list | None) -> list[dict]:
    '''
    Label the tokens using the provided tags and the BERT tokenizer.

    Args:
        tokens (list[list[str]] | list[str]): a train / val / test set of tokens
        tags (list[str] | None): a train / val / test set of class tags
    Returns:
        list[dict]: a list of dictionaries containing the processed tokens, labels, attention mask, and input IDs.
    '''
    l = label(tokens, tags)
    res =  [{'tokens': i['new_tokens'], 'attention_mask': i['attention_mask'], 'input_ids': i['input_ids'], 'word_ids': i['word_ids']} | ({'class': i['class']} if tags is not None else {}) for i in l]
    return res


def do_all(path: str, create_y: bool = True, shuffle: bool = False):
    res = load_data(path, create_y=create_y)
    if create_y:
        X_all, y_all = res
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X=X_all, y=y_all)
        data = ({'input': (X_train, y_train)},
                {'input': (X_val, y_val)},
                {'input': (X_test, y_test)})

    else:
        X_all = res
        y_all = None
        data = ({'input': (X_all, y_all)})
    for index, i in enumerate(data):
        res = do_all_labeling(*i['input'])

        items_list = ['attention_mask', 'input_ids', 'word_ids']
        res = NERDataset([{k: v for k, v in i.items() if k in items_list} for i in res], [i['class'] for i in res] if create_y else None)
        dl = DataLoader(res, batch_size=params.batch_size, shuffle=shuffle)

        data[index]['dataloader'] = dl

    if len(data) == 1:
        return data['dataloader']
    return tuple([i['dataloader'] for i in data])


if __name__ == '__main__':
    path = ds_path('protocols_speeches_clean.json')
    data = do_all(path, create_y=True, shuffle=False)

    print('Validation DataLoader:', data)
