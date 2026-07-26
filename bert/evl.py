'''
Evaluate the model by computing metrics.
'''

import evaluate as evl
import numpy as np
import torch

from bert import parameters as params

metric = evl.load("seqeval")

def evaluate(model, data_loader):
    # Compute accuracy of model on data provided by data_loader
    # num_instances = len(data_loader.dataset)
    labels = []
    predictions = []
    with torch.no_grad():
        model.eval()
        for i in iter(data_loader):
            X, y = i
            X = {k: v.to(params.device) for k, v in X.items()}
            y_preds = model(X)
            mask = (y != -100)
            filtered_pred = y_preds[mask]
            filtered_y = y[mask].float()
            labels.extend(filtered_y.cpu().numpy())
            predictions.extend(filtered_pred.cpu().numpy())
    res = metric.compute(predictions=predictions, references=labels)
    text_labels = [params.ner_labels[i] for i in  np.argmin(np.abs(labels - np.array(params.ner_labels)), axis=1).tolist()]
    text_preds = [params.ner_labels[i] for i in np.argmin(np.abs(predictions - np.array(params.ner_labels)), axis=1).tolist()]
    return res, [{"label": a, "prediction": b, "text_label": c, "text_prediction": d} for a, b, c, d in zip(labels, predictions, text_preds, text_labels)]
