'''
Evaluate the model by computing metrics.
'''

import evaluate

from bert.parameters import *

f1_metric = evaluate.load('f1')
precision_metric = evaluate.load('precision')
recall_metric = evaluate.load('recall')
accuracy_metric = evaluate.load('accuracy')

def evaluate_model(model, data_loader):
    labels = []
    predictions = []
    with torch.no_grad():
        model.eval()

        for i in iter(data_loader):
            X, y = i
            X = {k: v.to(device) for k, v in X.items()}
            y_probs = model(X)
            y_preds = torch.argmax(y_probs, dim=-1)

            predictions.extend(y_preds.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    results = {
            'overall_f1': f1_metric.compute(references=labels, predictions=predictions, average='macro')['f1'], # type: ignore
            'overall_precision': precision_metric.compute(references=labels, predictions=predictions, average='macro', zero_division=0)['precision'], # type: ignore
            'overall_recall': recall_metric.compute(references=labels, predictions=predictions, average='macro', zero_division=0)['recall'], # type: ignore
            'overall_accuracy': accuracy_metric.compute(references=labels, predictions=predictions)['accuracy'], # type: ignore
        }

    return results, {'labels': labels, 'predictions': predictions}


def clean_eval(eval_res, add_key_name: str | None = None) -> dict:
    '''
    Clean the evaluation results by converting all values to float.

    Args:
        eval_res (dict): The evaluation results to clean.

    Returns:
        dict: The cleaned evaluation results.
    '''
    if add_key_name is None:
        add_key_name = ''
    return {add_key_name + k: {vk: float(vv) for vk, vv in v.items()} if isinstance(v, dict) else float(v) for k, v in eval_res.items()}
