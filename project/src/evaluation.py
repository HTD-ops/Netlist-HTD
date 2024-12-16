# src/evaluation.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using various metrics.
    
    Args:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.
    
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return metrics

def print_evaluation(metrics):
    """
    Print evaluation metrics.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    print("Confusion Matrix:\n", metrics['confusion_matrix'])
    print("Accuracy: {:.2f}%".format(metrics['accuracy'] * 100))
    print("Precision: {:.2f}%".format(metrics['precision'] * 100))
    print("Recall (TPR): {:.2f}%".format(metrics['recall'] * 100))
    print("F1 Score: {:.2f}%".format(metrics['f1'] * 100))
