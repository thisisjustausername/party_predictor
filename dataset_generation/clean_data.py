'''
Clean the dataset of speeches and filter out speeches that are too short or are Nachfragen.
'''
import json
import os
from collections import defaultdict
from itertools import groupby

from transformers import BertTokenizer

from bert import parameters as params
from bert.data_set import train_test_split

######### INITIALIZE PARAMETERS #########
write: bool = False
min_speech_length: int = 300 # avoid 'Nachfragen'
min_chunk_length: int = 200
######### END INITIALIZE PARAMETERS #########


######### LOAD DATA #########
# Load the protocols and party scores from the JSON files.
with open('datasets/protocols_speeches.json', 'r') as f:
    protocols_speeches = json.load(f)

# Load the party scores from the JSON file.
with open('datasets/party_scores.json', 'r') as f:
    party_scores = json.load(f)
######### END LOAD DATA #########

######### CLEAN DATA #########
# Clean the talker's party names.
for i in protocols_speeches:
    if i['talker']['party'] == 'SPDSPD':
        i['talker'] = {k: v[:len(v)//2] for k, v in i['talker'].items()}

# If write is True, save the cleaned data to a JSON file.
if write:
    with open('datasets/protocols_speeches.json', 'w') as f:
        json.dump(protocols_speeches, f, indent=4)

# only keep speeches from parties, that are represented in the party scores
clean_data = [i for i in protocols_speeches if i['talker']['party'] in party_scores]
# remove empty speeches
clean_data = [i for i in clean_data if i['speech'] != '']
# for i in clean_data:
#     i['speech'] = i['speech'].split('\nVielen Dank.\n')[0]

# debug: print number of speeches
print(f'Number of speeches: {len(clean_data)}')


# TODO: optimize so that tokenization is only necessary once
# tokenize speeches in order to remove too short ones
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

# drop speeches that are too short and therefore not important or even Nachfragen instead of speeches
clean_data = [i for i in clean_data if i['speech_length'] >= min_speech_length]
for i in clean_data:
    i['class'] = params.label_to_index(i['talker']['party'])

# save data
with open('datasets/protocols_speeches_clean.json', 'w') as f:
    json.dump(clean_data, f, indent=4)
######### END CLEAN DATA #########


######### CREATE TOKENIZED DATASET #########
# removed Nachfragen from dataset above, now actually create the tokenized dataset for training
# max len sets the input for BERT, since standard BERT is used, 512 is the limit
max_len = 512
# stride is 20% of max len for overlaps
stride = int(max_len * 0.2)

# tokenize speeches using BERT tokenizer
result = tokenizer(
    [i['speech'] for i in clean_data],
    add_special_tokens=True,
    return_attention_mask=True,
    truncation=True,
    padding=True,
    max_length=max_len,
    return_overflowing_tokens=True,
    stride=stride,
)
# extract important data
sample_map = result['overflow_to_sample_mapping']
input_ids = result["input_ids"]
attention_mask = result["attention_mask"]

# counts chunks for each speech
counts = {key: len(list(group)) for key, group in groupby(sample_map)}

# chunk_counter for index of each chunk within a speech
chunk_counters = defaultdict(int)
# stores chunked data
chunked_data = []
# iterates over sample_map to create information about each chunk
for index, (chunk, sm_id) in enumerate(zip(input_ids, sample_map)):
    speech_idx = sm_id
    dta = clean_data[speech_idx]
    # NOTE: removing too short chunks
    if len(chunk) < min_chunk_length:
        continue
    entry = {
        'id': index,
        'orig_speech': dta['speech'],
        'orig_speech_length': dta['speech_length'],
        'party': dta['talker']['party'],
        'class': dta['class'],
        'talker': dta['talker'].copy(),
        'input_ids': chunk,
        'attention_mask': attention_mask[index],
        'chunk_length': len(chunk),
        'speech_id': speech_idx,
        'chunk_index': chunk_counters[speech_idx],
        'chunks_count': counts[speech_idx],
    }
    chunk_counters[speech_idx] += 1
    chunked_data.append(entry)

# write chunked and tokenized data to file
with open('datasets/protocols_speeches_chunked.json', 'w') as f:
    json.dump(chunked_data, f, indent=4)
######### END CREATE TOKENIZED DATASET #########

######### CREATE TRAIN/VAL/TEST SPLITS #########
# X = chunked_data
X = [{key: value for key, value in entry.items() if key in ['id', 'input_ids', 'attention_mask']} for entry in chunked_data]
y = [i['class'] for i in chunked_data]
# for entry in X:
#     y.append(entry.pop('class'))

X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y, test_size=0.2, val_size=0.1, random_seed=42)

datasplits = {
    'train': {
        'X': X_train,
        'y': y_train
    },
    'val': {
        'X': X_val,
        'y': y_val
    },
    'test': {
        'X': X_test,
        'y': y_test
    }
}

with open('datasets/datasplits.json', 'w') as f:
    json.dump(datasplits, f) # don't use indents, as data isn't meant to be human-readable
