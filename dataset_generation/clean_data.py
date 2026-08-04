'''
Clean the dataset of speeches and filter out speeches that are too short or are Nachfragen.
'''
import json
import os

from transformers import BertTokenizer

from bert import parameters as params

### INITIALIZE PARAMETERS
write: bool = False
min_speech_length: int = 300 # avoid 'Nachfragen'
### END INITIALIZE PARAMETERS


with open('datasets/protocols_speeches.json', 'r') as f:
    protocols_speeches = json.load(f)

with open('datasets/party_scores.json', 'r') as f:
    party_scores = json.load(f)

for i in protocols_speeches:
    if i['talker']['party'] == 'SPDSPD':
        i['talker'] = {k: v[:len(v)//2] for k, v in i['talker'].items()}

if write:
    with open('datasets/protocols_speeches.json', 'w') as f:
        json.dump(protocols_speeches, f, indent=4)

clean_data = [i for i in protocols_speeches if i['talker']['party'] in party_scores]
clean_data = [i for i in clean_data if i['speech'] != '']
# for i in clean_data:
#     i['speech'] = i['speech'].split('\nVielen Dank.\n')[0]

print(f' Number of speeches: {len(clean_data)}')

tokenizer = BertTokenizer.from_pretrained(os.path.join(params.model_base_path, params.mdl), do_lower_case=False)

results = []

result = tokenizer(
    [i['speech'] for i in clean_data],
    add_special_tokens=True,
    return_attention_mask=True,
    truncation=False,
    padding=False,
)
for index, i in enumerate(clean_data):
    i['speech_length'] = len(result['input_ids'][index])

clean_data = [i for i in clean_data if i['speech_length'] >= min_speech_length]
for i in clean_data:
    i['class'] = params.label_to_index(i['talker']['party'])

with open('datasets/protocols_speeches_clean.json', 'w') as f:
    json.dump(clean_data, f, indent=4)
