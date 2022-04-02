import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    f1_scores = f1_score(y_true=labels, y_pred=predictions, average="macro", zero_division=1)
    precision_scores = precision_score(y_true=labels, y_pred=predictions, average="macro", zero_division=1)
    recall_scores = recall_score(y_true=labels, y_pred=predictions, average="macro", zero_division=1)
    accuracy_scores = accuracy_score(y_true=labels, y_pred=predictions)

    confustion_metrix = torch.zeros((26, 26), dtype=torch.long)
    for i in range(len(predictions)):
        confustion_metrix[labels[i]][predictions[i]] += 1
    metrics = {
        "accuracy": accuracy_scores,
        "precision": precision_scores,
        "recall": recall_scores,
        "f1": f1_scores,
    }
    for i in range(26):
        metrics["label_" + str(i) + "_success"] = confustion_metrix[i][i]
        metrics["label_" + str(i) + "_fails"] = confustion_metrix[i].sum(dim=-1) - confustion_metrix[i][i]
    return metrics