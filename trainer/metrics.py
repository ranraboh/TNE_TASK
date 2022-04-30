from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from trainer.accuracy_metric import accuracy_score
import torch

from trainer.tne_config import TNE_Config


class MetricsEvaluation:
    def __init__(self, config : TNE_Config):
        """ Init metric evaluation object, the class is used to compute the metrics of the model """
        # Best results for each metric
        self.best_f1 = 0
        self.best_accuracy = 0
        self.best_recall = 0
        self.best_precision = 0

        # Init helper fields
        self.eval_samples = 500
        self.nof_classes =  config.num_labels
        self.labels = torch.arange(self.nof_classes, dtype=torch.long)
        self.reset_values()

    def reset_values(self):
        self.samples_counter = 0
        self.predictions= torch.zeros(0, dtype=torch.long)
        self.true_labels = torch.zeros(0, dtype=torch.long)

    def evaluate_metrics(self, labels, logits):
        # Concat the predictions and true labels of evaluation samples
        self.samples_counter += 1
        predictions = torch.argmax(logits, axis=-1).to("cpu")
        labels = labels.to("cpu")
        self.predictions = torch.concat((self.predictions, predictions), dim=0)
        self.true_labels = torch.concat((self.true_labels, labels), dim=0)
        if self.samples_counter < self.eval_samples:
            return

        # Compute the metrics over the evaluation set
        confustion_matrix = confusion_matrix(y_true=self.true_labels, y_pred=self.predictions, labels=self.labels)
        f1_scores = f1_score(y_true=self.true_labels, y_pred=self.predictions, average="macro", zero_division=1, labels=self.labels)
        precision_scores = precision_score(y_true=self.true_labels, y_pred=self.predictions, average="macro", zero_division=1, labels=self.labels)
        recall_scores = recall_score(y_true=self.true_labels, y_pred=self.predictions, average="macro", zero_division=1, labels=self.labels)
        accuracy_scores = accuracy_score(y_true=self.true_labels, y_pred=self.predictions, exclude_labels=[25])
        metrics = {
            "eval_precision": precision_scores,
            "eval_accuracy": accuracy_scores,
            "eval_recall": recall_scores,
            "eval_f1": f1_scores
        }

        # Print out the metrics results
        self.display_metrics(metrics, confustion_matrix)
        self.reset_values()

    def display_metrics(self, metrics, confusion_matrix):
        # Compute the best results for each metric so far
        self.best_f1 = max(self.best_f1, metrics['eval_f1'])
        self.best_accuracy = max(self.best_accuracy, metrics['eval_accuracy'])
        self.best_recall = max(self.best_recall, metrics['eval_recall'])
        self.best_precision = max(self.best_precision, metrics['eval_precision'])
        best_results = {
            "best_accuracy": self.best_accuracy,
            "best_recall": self.best_recall,
            "best_precision": self.best_precision,
            "best_f1": self.best_f1
        }

        # Print out metrics evaluation
        print ("-- metrics evaluation --")
        print (metrics)
        print("-- confusion matrix --")
        print(confusion_matrix)
        print("-- best metric results --")
        print (best_results)