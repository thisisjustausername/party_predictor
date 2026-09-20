'''
DEPRECATED
This script did not gain any improvements.
'''

import json
import os

import joblib
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import parameters as params
from bert.data_set import ds_path, split_to_loader
from bert.datatypes import BertClsModel

model_name = 'model_6_latest.pth'
recompute_outputs = False


if recompute_outputs:
    model = BertClsModel()
    model.load_state_dict(torch.load(os.path.join(params.repo_base_path, f'finetuned_models/{model_name}')))
    model = model.to(params.device)


    def get_data(data: DataLoader) -> tuple[list, list]:
        """
        Get data from DataLoader and return model output with ground truth labels.

        Args:
            data (DataLoader): DataLoader object containing the data.

        Returns:
            tuple[list, list]: Tuple containing two lists: First is a list of 5-dim tuples with the classes, second are the ground labels.
        """
        outputs = []
        labels = []
        with torch.no_grad():
            model.eval()

            for i in tqdm(iter(data), desc='Computing outputs'):
                X, y = i
                X = {k: v.to(params.device) for k, v in X.items()}
                y_probs = model(X)

                outputs.extend(y_probs.cpu().numpy().tolist())
                labels.extend(y.cpu().numpy().tolist())

        labels = [params.ner_labels[i] for i in labels]
        return outputs, labels


    dataloaders = split_to_loader(ds_path('datasplits.json'), shuffle=(True, False, False))
    dataloaders = {key: get_data(value) for key, value in dataloaders.items()} # type: ignore
    train, val, test = dataloaders.values()

    with open(os.path.join(params.repo_base_path, 'datasets', f'{model_name}_outputs.json'), 'w') as f:
        json.dump({'train': train, 'val': val, 'test': test}, f)
else:
    with open(os.path.join(params.repo_base_path, 'datasets', f'{model_name}_outputs.json'), 'r') as f:
        data = json.load(f)
    train = data['train']
    val = data['val']
    test = data['test']

X_tv = np.vstack([train[0], val[0]])
y_tv = np.hstack([train[1], val[1]])
val_mask = np.concatenate([np.full(len(train[0]), -1), np.zeros(len(val[0]))])
ps = PredefinedSplit(val_mask)

pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight='balanced'))

grid = GridSearchCV(
    pipe,
    {
        "svc__C": np.logspace(-2, 3, 6),
        "svc__gamma": np.logspace(-3, 1, 5),
    },
    scoring="f1_macro",
    cv=ps,
    refit=True,
    n_jobs=-1,
)
grid.fit(X_tv, y_tv)
print(grid.best_params_)
joblib.dump(grid, f'svm_{model_name}.joblib')
print("test f1:", grid.score(*test))
