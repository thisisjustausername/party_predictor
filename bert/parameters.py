'''
Set up parameters for finetuning the BERT model on the Classification task and evaluating it.
'''

import json

import numpy as np
import torch

#########################
# INITIALIZE PARAMETERS
#########################

# after what length of tokens (subword tokens) to cut off the input speech
cutoff = 2500

# model to use
mdl = 'modern-german-bert'

# random seed for training
seed = 42

# training hyperparameters
num_epochs = 64

# early stopping
early_stopping_patience: int | None = 5

# NOTE: The effective batch size is the product of the batch size and the number of gradient accumulation steps.
batch_size = 2
accum_steps = 8

learning_rate = 3e-5
betas=(0.9,0.999)
epsilon=1e-08

print(f'Effective batch size: {batch_size * accum_steps}')


#########################
# OTHER PARAMETERS
#########################

# paths to dataset, model and repo
dataset_base_path = 'datasets/'
model_base_path = 'models/'
repo_base_path = '/home/lpwgf/programming/party_predictor/'

# parties
ner_labels = ['CDU/CSU', 'SPD', 'BÜNDNIS 90/DIE GRÜNEN', 'Die Linke', 'AfD']

# TODO: maybe remove this
one_hot_encoded_labels = np.identity(len(ner_labels))
def label_to_one_hot(label):
    return one_hot_encoded_labels[ner_labels.index(label)]

# match label to index in ner_labels list
def label_to_index(label):
    return ner_labels.index(label)

# inialize party colors for plotting
party_colors =  ['black', 'red', 'green', 'purple', 'blue']

# set device for fast computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(repo_base_path + 'datasets/party_scores.json', 'r') as f:
    party_scores = json.load(f)
scores = np.array(list(party_scores.values()))
party_mean = np.mean(scores)
party_std = np.std(scores)
party_scores = dict(zip(list(party_scores.keys()), (scores - party_mean) / party_std))
del scores, party_mean, party_std

# input length for tokenized articles
with open(repo_base_path + f'models/{mdl}/config.json', 'r') as f:
    config = json.load(f)
input_length = config['max_position_embeddings']
