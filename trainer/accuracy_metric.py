def accuracy_score(y_true, y_pred, exclude_labels):
    nof_predictions = len(y_pred)
    positive_predictions = 0
    negative_predictions = 0
    for i in range(nof_predictions):
        if y_true[i] in exclude_labels or y_pred[i] in exclude_labels:
            continue
        if y_true[i] == y_pred[i]:
            positive_predictions = positive_predictions + 1
        else:
            negative_predictions = negative_predictions + 1
    return positive_predictions / (negative_predictions + positive_predictions)