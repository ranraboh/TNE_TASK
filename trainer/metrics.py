import torch
import numpy as np
from datasets import load_metric

# Metrics
acc_metric = load_metric("accuracy")
f1_metric = load_metric("f1")
prec_metric = load_metric("precision")
rec_metric = load_metric("recall")

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    metrics = acc_metric.compute(predictions=predictions, references=labels)
    metrics.update(f1_metric.compute(predictions=predictions, references=labels, average="micro")) # macro (?)
    metrics.update(prec_metric.compute(predictions=predictions, references=labels, average="micro"))
    metrics.update(rec_metric.compute(predictions=predictions, references=labels, average="micro"))
    return metrics