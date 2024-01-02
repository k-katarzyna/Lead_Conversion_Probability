import os
from joblib import load

import pandas as pd
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             balanced_accuracy_score)


def load_city_list(file_path):
    """
    Load a list of cities from a given txt file. Cities are expected to be 
    separated by commas.

    Args:
        file_path (str): The path to the file containing city names.

    Returns:
        list: A list containing all the city names from the file.
    """
    with open(file_path, "r") as file:
        line = file.readline()
        cities = [city.strip() for city in line.split(",")]
        
    return cities


def to_labels(y_proba, threshold):
    """
    Convert predicted probabilities to binary labels based on the given
    threshold.

    Args:
        y_proba (np.array): Predicted probabilities.
        threshold (float): Threshold value for converting probabilities to
            binary labels.

    Returns:
        np.array: Binary labels based on the threshold.
    """
    return (y_proba >= threshold).astype("int")


def load_estimators(folder_path):
    """
    Load previously optimized estimators from saved artifacts.

    Args:
        folder_path (str): Path to the folder containing saved PKL files.

    Returns:
        list of tuples: List of tuples containing names and corresponding
            estimators.
    """  
    estimators = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(folder_path, file_name)
            estimator = load(file_path)
            name = estimator.steps[2][1].__class__.__name__.split("Classifier")[0]
            estimators.append((name, estimator))

    return estimators


def calculate_classification_metrics(estimators, thresholds, 
                                     X_train, X_test, y_train, y_test): 
    """
    Calculate precision, recall, F1 score, balanced accuracy and the percentage
    of positive class predictions for a list of estimators, using specified
    classification thresholds.

    Args:
        estimators (list of tuples): A list where each tuple consists of an
            estimator's name (str) and its corresponding estimator object. Each
            tuple is structured as (estimator_name, estimator_object).
        thresholds (list of floats): A list of threshold values for classification,
            where each threshold value corresponds to the respective estimator in
            the 'estimators' list. The order of threshold values should match
            the order of estimators.
        X_train (pd.DataFrame): Training data.
        X_test (pd.DataFrame): Testing data.
        y_train (pd.Series): Target values for training data.
        y_test (pd.Series): Target values for testing data.
    
    Returns:
        tuple of (metrics, results):
            - metrics (list): A list of names (str) of the calculated metrics.
            - results (list): A list of lists, where each sublist contains 
              the scores (floats) of a calculated metric for each estimator.
    """
    metrics = ["Precision", "Recall", "F1 Score", "Balanced Accuracy",
               "% of positive class predictions"]
    results = []

    for (_, estimator), threshold in zip(estimators, thresholds):
        estimator.fit(X_train, y_train)
        y_proba = estimator.predict_proba(X_test)[:, 1]
        y_pred = to_labels(y_proba, threshold)

        scores = [precision_score(y_test, y_pred),
                  recall_score(y_test, y_pred),
                  f1_score(y_test, y_pred),
                  balanced_accuracy_score(y_test, y_pred),
                  np.mean(y_pred)]

        results.append(scores)
        
    return metrics, results


def display_final_results(estimators, auc_scores): 
    """
    Create a DataFrame displaying the final ROC AUC scores for a list of estimators.

    Args:
        estimators (list of tuples): List of (estimator_name, estimator) pairs.
        auc_scores (list): List of ROC AUC scores corresponding to the estimators.

    Returns:
        pd.DataFrame: A DataFrame containing the ROC AUC scores for each estimator,
            sorted in descending order.
    """
    results = {name: score 
               for (name, _), score in zip(estimators, auc_scores)}
        
    df = pd.DataFrame.from_dict(results, 
                                orient="index", 
                                columns=["ROC_AUC"])
    
    return df.sort_values(by="ROC_AUC", ascending=False)
    
    