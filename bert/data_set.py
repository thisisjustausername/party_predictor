'''
Create datasets for training, validation and testing.
* datasets are created from the json files in the dataset folder
* datasets are tokenized using the BERT tokenizer and the labels are created using the ner_labels_reverse dictionary
* datasets are then returned as DataLoader objects
'''
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from bert import parameters as params
from bert.datatypes import DataSet

#########################
# INITIALIZE LOCAL PARAMETERS
#########################

ner_labels_reverse = {i: j for j, i in enumerate(params.ner_labels)}

# set up tokenizer
tokenizer = BertTokenizer.from_pretrained(os.path.join(params.model_base_path, params.mdl), do_lower_case=True)
print('Is fast encoder (should be True):', tokenizer.is_fast)


#########################
# CREATE FUNCTIONS
#########################

def load_data(path: str) -> tuple[list[str], list[float]]:
    '''
    Load data from a json file and return a list of tuples containing the statement and verdict.

    Args:
        path (str): path to json file

    Returns:
        list[tuple[str, str]]: a list of tuples containing the statement and verdict
    '''
    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame.from_records(data)

    labels = df['verdict'].unique()

    # print(f"Labels match parameters ner_label list: {Counter(labels) == Counter(params.ner_labels)}")

    df_clean = pd.DataFrame()
    df_clean["PHRASE"] = df.statement
    df_clean["LABEL"] = df.verdict.apply(lambda x: params.ner_labels.index(x) if x in params.ner_labels else -100)

    return df_clean.PHRASE.tolist(), df_clean.LABEL.tolist()
    # return list(zip(df_clean.PHRASE, df_clean.LABEL))


def tokenize(tokens: list[str] | str):
    '''
    Tokenize a list of tokens using the BERT tokenizer.

    Args:
        tokens (list[str] | str): A list of tokens or a list of lists of tokens.
        max_length (int): The maximum length of the tokenized sequence.

    Returns:
        dict: A dictionary containing the tokenized input IDs, attention mask, and alignment data.
    '''
    # if not list of lists (dataset, so set of instances), make it a list of lists as each instance is a list of tokens again
    if isinstance(tokens, str):
        tokens = [tokens]
    # tokenize the tokens using the BERT tokenizer
    result = tokenizer(tokens, return_tensors='pt', add_special_tokens=True,
            return_attention_mask=True, truncation=True, padding=True, max_length=params.input_length)

    num_rows = result['input_ids'].shape[0]          # actual output rows, >= len(tokens) if overflow happened

    # create alignment data for each instance in the batch
    alignment_data = [[(t, i) for t, i in zip(result.tokens(i), result.word_ids(i))] for i in range(num_rows)]

    # return the tokenized input IDs, attention mask, and alignment data
    return {
      'alignment_data': alignment_data,
      'attention_mask': result['attention_mask'],
      'input_ids': result['input_ids'],
    }


def label(tokens: list[str] | str, tags: list[str] | str) -> list[dict]:
    '''
    Args:
    tokens (list[list[str]] | list[str]): a train / val / test set of tokens
    tags (list[list[str]] | list[str] | None): a train / val / test set of tags
    '''

    # tokenize tokens using BERT tokenizer
    output = tokenize(tokens)
    # fetch the important outputs
    result = output['alignment_data']
    attention_mask = output['attention_mask']
    input_ids = output['input_ids']
    # initialize new data list to store the processed tokens and labels
    new_data = []

    # TODO: optimize
    # for each instance (sublist) in the batch, process the tokens and labels
    for ner_tag, new_tokens, mask, ids in zip(tags, result, attention_mask, input_ids):

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
        res['label'] = params.ner_labels[ner_tag]

        new_data.append(res)

    # return the list of processed tokens and labels
    return new_data


def do_all_labeling(tokens, tags: list) -> list[dict]:
    '''
    Args:
        tokens (list[list[str]] | list[str]): a train / val / test set of tokens
        tags (list[list[str]] | list[str]): a train / val / test set of tags
    Returns:
        list[dict]: a list of dictionaries containing the processed tokens, labels, attention mask, and input IDs. If all_tokens is True, also includes group_list.
    '''
    lbl = label(tokens, tags)
    res =  [
        {
            'tokens': i['new_tokens'],
            'attention_mask': i['attention_mask'],
            'input_ids': i['input_ids'],
            'word_ids': i['word_ids'],
            'label': i['label']}
       for i in lbl
    ]
    return res


def do_all(
    path: str,
    train_val_test_split: tuple[float, float, float] = (0.7, 0.1, 0.2),
    shuffle: tuple[bool, bool, bool] = (True, False, False)
) -> tuple[DataLoader, DataLoader, DataLoader]:
    '''
    '''
    res = load_data(path)
    X, y = res
    res = do_all_labeling(X, y)

    items_list = ['attention_mask', 'input_ids', 'word_ids']
    X = [{k: v for k, v in i.items() if k in items_list} for i in res]
    y = [i['label'] for i in res]

    X_train, X_after, y_train, y_after = train_test_split(X, y, test_size=train_val_test_split[1] + train_val_test_split[2], random_state=params.seed, stratify=y, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_after, y_after, test_size=train_val_test_split[2], random_state=params.seed, stratify=y_after, shuffle=True)
    del X_after, y_after

    X_all = [X_train, X_val, X_test]
    y_all = [y_train, y_val, y_test]

    dls = []
    for index, (X_curr, y_curr) in enumerate(zip(X_all, y_all)):
        dls.append(DataLoader(DataSet(X_curr, y_curr), batch_size=params.batch_size, shuffle=shuffle[index]))

    return tuple(dls) # type: ignore
