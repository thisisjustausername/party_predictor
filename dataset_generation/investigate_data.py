import json

import pandas as pd
from IPython.display import display

with open('datasets/protocols_speeches_clean.json', 'r') as f:
    data = json.load(f)

lens = [i['speech_length'] for i in data]
print(f'Number of speeches: {len(lens)}')
print(f'min: {min(lens)}\nmax: {max(lens)}\nmean: {sum(lens)/len(lens)}\nmedian: {sorted(lens)[len(lens)//2]}')

df_data = [dict(**i['talker'] | dict( e for e in i.items() if e [0] != 'talker')) for i in data]
df = pd.DataFrame(df_data)
del df_data

display(df.head(15))

display(df.groupby('party').agg({'speech_length': ['mean', 'median', 'min', 'max']}).sort_values(('speech_length', 'mean'), ascending=False))
