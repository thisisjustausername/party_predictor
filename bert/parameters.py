'''
Set up parameters for finetuning the BERT model on the NER task and evaluating it.
'''

import json

import numpy as np
import torch

#########################
# INITIALIZE PARAMETERS
#########################

# input length for tokenized articles
input_length = 300
stride = 50

# model to use
# mdl = 'bert-tiny'
mdl = 'bert-base-german-cased'

# random seed for training
seed = 42

# training hyperparameters
num_epochs = 30
batch_size = 8
learning_rate = 3e-5
betas=(0.9,0.999)
epsilon=1e-08

# weights = torch.tensor([0.01] + [5.0, 10.0, 5.0, 3.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=torch.float)


#########################
# OTHER PARAMETERS
#########################
dataset_base_path = 'data/'
model_base_path = 'models/'
repo_base_path = '/home/lpwgf/programming/party_predictor/'

ner_labels = ['CDU', 'SPD', 'FDP', 'GRÜNE', 'LINKE', 'AFD']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(repo_base_path + 'datasets/party_scores.json', 'r') as f:
    party_scores = json.load(f)
scores = np.array(list(party_scores.values()))
party_mean = np.mean(scores)
party_std = np.std(scores)
party_scores = dict(zip(list(party_scores.keys()), (scores - party_mean) / party_std))
del scores, party_mean, party_std
