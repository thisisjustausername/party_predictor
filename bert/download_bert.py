from huggingface_hub import snapshot_download

repo = 'google-bert/bert-base-german-cased'
snapshot_download(repo, local_dir='models/bert-base-german-cased')
