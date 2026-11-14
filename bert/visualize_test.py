
import os

import matplotlib
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import bert.parameters as params
from bert.data_set import ds_path, split_to_loader
from bert.datatypes import BertClsModel

model_name = 'model_6.pth'
dims = 3

matplotlib.use("TkAgg")

random_state = np.random.RandomState(params.seed)

predictions = []

model = BertClsModel()
model.load_state_dict(torch.load(os.path.join(params.repo_base_path, f'finetuned_models/{model_name}')))
model = model.to(params.device)

test = split_to_loader(ds_path('datasplits.json'), shuffle=(True, False, False))['test'] # type: ignore

labels = []
outputs = []
with torch.no_grad():
    model.eval()

    for i in iter(test):
        X, y = i
        X = {k: v.to(params.device) for k, v in X.items()}
        y_probs = model(X)

        outputs.extend(y_probs.cpu().numpy().tolist())
        labels.extend(y.cpu().numpy().tolist())

labels = [params.ner_labels[i] for i in labels]

np.random.seed(params.seed)
pca = PCA(n_components=dims)
reduced = pca.fit_transform(np.array(outputs))
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
print(f'Total explained variance: {np.sum(pca.explained_variance_ratio_)}')

plt.figure(figsize=(10, 10))
if dims == 2:
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, hue_order=params.ner_labels, palette=params.party_colors, legend='full')
else:
    labels = np.asarray(labels)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection='3d')

    for index, label in enumerate(params.ner_labels):                      # same order as hue_order
        mask = labels == label
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
            color=params.party_colors[index],            # see note below
            label=label,
            s=20, alpha=0.8,
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend(title="Label")
    plt.show()

plt.title(f'PCA of BERT Predictions (dims={dims})')
if dims == 2:
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
plt.legend(title='Labels', loc='best')
plt.tight_layout()
plt.savefig(os.path.join(params.repo_base_path, f'finetuned_model_visualized/pca_{model_name.split(".", 1)[0]}_{dims}d.png'))
plt.show()
