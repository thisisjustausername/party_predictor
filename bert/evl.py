'''
Evaluate the model by computing metrics.
'''

import evaluate

from bert.parameters import *

metric = evaluate.load('seqeval')

def evaluate_model(model, data_loader, tokens_included: bool = False):
    # Compute accuracy of model on data provided by data_loader
    num_instances = len(data_loader.dataset)
    labels = []
    predictions = []
    with torch.no_grad(): # This tells the model that we're not training
                        # Will not remember gradients for this block
        model.eval()

        for i in iter(data_loader):
            if tokens_included:
                X, y, tokens = i
            else:
                X, y = i
            X = {k: v.to(device) for k, v in X.items()}
            y_probs = model(X)
            y_preds = torch.argmax(y_probs, dim=-1)
            for row_y, row_pred in zip(y, y_preds):
                label_row = []
                preds_row = []
                for yo, yp in zip(row_y, row_pred):
                    if yo != -100:
                        # if yo.item() == yp.item() and yo.item() != 0:
                            # print('Correct prediction:', yo.item(), yp.item())
                            # print('Predicted label:', ner_labels[yp.item()])
                        label_row.append(ner_labels[yo])
                        preds_row.append(ner_labels[yp])

                predictions.append(preds_row)
                labels.append(label_row)

    return metric.compute(references=labels, predictions=predictions), {'labels': labels, 'predictions': predictions}
