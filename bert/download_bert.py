'''
Downloads a model from the Hugging Face Hub and saves it to a local directory.
'''
from huggingface_hub import snapshot_download

repo = 'google-bert/bert-base-german-cased'
repo = 'LSX-UniWue/ModernGBERT_1B'
snapshot_download(repo, local_dir='models/modern-german-bert')
