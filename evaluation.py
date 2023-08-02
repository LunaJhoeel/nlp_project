import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import label_binarize


def get_performance(predictions, y_test, labels=[1, 0]):
    """
    Calculates performance metrics of a given prediction model.

    Parameters
    ----------
    predictions : numpy.ndarray or list
        Predicted classes from the model.
    y_test : numpy.ndarray or list
        Ground truth classes.
    labels : list, optional
        Labels for the classification classes, by default [1, 0]

    Returns
    -------
    accuracy : float
        Accuracy of the model's predictions.
    precision : float
        Precision of the model's predictions.
    recall : float
        Recall of the model's predictions.
    f1_score : float
        F1 score of the model's predictions.

    """
    # Calculate accuracy score
    accuracy = metrics.accuracy_score(y_test, predictions)
    # Calculate precision score
    precision = metrics.precision_score(y_test, predictions, labels=labels)
    # Calculate recall score
    recall = metrics.recall_score(y_test, predictions, labels=labels)
    # Calculate f1 score
    f1_score = metrics.f1_score(y_test, predictions, labels=labels)
    # Get classification report
    report = metrics.classification_report(y_test, predictions, labels=labels)
    # Get confusion matrix
    cm = metrics.confusion_matrix(y_test, predictions, labels=labels)
    cm_as_dataframe = pd.DataFrame(data=cm)
    # Print performance metrics
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)
    
    return accuracy, precision, recall, f1_score


def plot_roc(model, y_test, features):
    """
    Plot the ROC curve for a binary classification model.

    Parameters
    ----------
    model: Model
        The binary classification model to evaluate.
    y_test: array-like
        Ground truth (correct) target values.
    features: array-like
        The input features to make predictions on.
        
    Returns
    -------
    roc_auc: float
        The area under the ROC curve.

    """
    # Calculate the ROC curve score
    y_score = model.predict_proba(features)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc