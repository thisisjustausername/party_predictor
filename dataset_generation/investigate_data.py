'''
Investigate the data to find distributions and more.
Can be used as ZED-Notebook.
'''
# %% Imports
import json

import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt

from bert import parameters as params

# %% Import data
with open('datasets/protocols_speeches_clean.json', 'r') as f:
    data = json.load(f)

# %% Data overview
lens = [i['speech_length'] for i in data]
print(f'Number of speeches: {len(lens)}')
print(f'min: {min(lens)}\nmax: {max(lens)}\nmean: {sum(lens)/len(lens)}\nmedian: {sorted(lens)[len(lens)//2]}')


df_data = [dict(**i['talker'] | dict( e for e in i.items() if e [0] != 'talker')) for i in data]
df = pd.DataFrame(df_data)
del df_data

display(df.head(15))


# %% Data analysis
print()
print('Parties:')
print(df['party'].unique().tolist())
print()


display(df.groupby('party').agg({'speech_length': ['mean', 'median', 'min', 'max']}).sort_values(('speech_length', 'mean'), ascending=False))
display(df.groupby('party').sum())
print()


print('Number of speeches per party:')
display(df.groupby('party').count().speech_length.sort_values(ascending=False))
print()

print('Number of speeches per party staying vs cut off:')
display(f'{df[df['speech_length'] <= params.cutoff].speech_length.count()} vs {df[df['speech_length'] > params.cutoff].speech_length.count()}')

# %% plots
sns.histplot(data=df, x='speech_length', hue='party', multiple='stack', bins=50, palette=dict(zip(params.ner_labels, params.party_colors)))
plt.show()
