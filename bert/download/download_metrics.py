'''
Downloads the metrics used for evaluation from the Hugging Face Hub and saves them to a local directory.
'''
from huggingface_hub import snapshot_download

metrics = ["f1", "precision", "recall", "accuracy"]

for name in metrics:
    snapshot_download(
        repo_id=f"evaluate-metric/{name}",
        repo_type="space",
        local_dir=f"models/metrics/{name}",
    )
