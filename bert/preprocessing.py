'''
preprocess the raw historical data for further computation
'''

import os
import json
import pandas as pd
from IPython.display import display

from bert import parameters as params

name = 'politifact'

# read data document
with open(os.path.join(params.dataset_base_path, f'{name}.json'), 'r') as f:
    data = json.load(f)


def clean_tokens(data: pd.DataFrame, replace_tokens: list[tuple[str, str]] = [(r' "([^"]*)" ', ' [CITESTART] \1 [CITEEND] ')]) -> list[tuple[str, str]]:
    '''
    replace bad tokens with more meaningful ones

    Args:
        data (pd.DataFrame): original input data
        replace_tokens (dict[str, str]): tokens to replace with their corresponding values
    Returns:
        list[tuple[str, str]]: cleaned data in format: [(phrase, label), ...]
    '''
    for k, v in replace_tokens:
        data['PHRASE'] = data['PHRASE'].str.replace(k, v, regex=True)

    data: list[tuple[str, str]] = list(zip(data['PHRASE'], data['LABEL']))
    return data

if __name__ == '__main__':

    # print(data[0]['tokens'][0].keys())
    text = clean_tokens(data[0]['tokens'])
    t = ''
    for i in text:
        t += (i['TOKEN'] + i['SEP']) # type: ignore
    print(t)

    pd.options.display.max_rows = None
    df = pd.DataFrame(text)
    display(df) # .head(5))
