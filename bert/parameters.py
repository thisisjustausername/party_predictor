'''
Set up parameters for finetuning the BERT model on the NER task and evaluating it.
'''

import numpy as np
import torch

#########################
# INITIALIZE PARAMETERS
#########################

# input length for tokenized articles
input_length = 300

# model to use
# mdl = 'bert-tiny'
mdl = 'bert-base-uncased'

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
repo_base_path = '/home/lpwgf/programming/lie_ability/'

ner_labels = ['CDU', 'SPD', 'FDP', 'GRÜNE', 'LINKE', 'AFD']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
