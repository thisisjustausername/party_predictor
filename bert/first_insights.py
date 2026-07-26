'''
Load and show raw data
'''
import os
import pandas as pd
import json
from IPython.display import display

from bert import parameters as params
from bert import data_set as lbl

path = os.path.join(params.repo_base_path, 'data', 'politifact.json')

with open(path, "r") as f:
    data = json.load(f)

df = pd.DataFrame.from_records(data)

display(df.head(10))

labels = df['verdict'].unique()

df_clean = pd.DataFrame()
df_clean["PHRASE"] = df.statement
df_clean["LABEL"] = df.verdict

# print(list(zip(df_clean.PHRASE, df_clean.LABEL)))

lbl.do_all(path)
