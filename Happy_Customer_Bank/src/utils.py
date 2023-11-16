from joblib import load
import os
import pandas as pd


def to_labels(y_proba, threshold):
    
    """
    Converts predicted probabilities to binary labels based on the given threshold.

    Args:
    -----------
    y_proba (np.array): Predicted probabilities.
    threshold (float): Threshold value for converting probabilities to binary labels.

    Returns:
    -----------
    np.array: Binary labels based on the threshold.
    """
    
    return (y_proba >= threshold).astype("int")


def load_estimators(folder_path):
    
    """
    Loads previously optimized estimators from saved pickle files.

    Args:
    -----------
    folder_path (str): Path to the folder containing saved pickle files.

    Returns:
    -----------
    list of tuples: List of tuples containing names and corresponding estimators.
    """
    
    estimators = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(folder_path, file_name)
            estimator = load(file_path)
            name = estimator.steps[2][1].__class__.__name__.split("Classifier")[0]
            estimators.append((name, estimator))

    return estimators


def display_final_results(estimators, auc_scores):
    
    """
    Creates a DataFrame displaying the final ROC AUC scores for a list of estimators.

    Args:
    -----------
    estimators (list of tuples): List of (estimator_name, estimator) pairs.
    auc_scores (list): List of ROC AUC scores corresponding to the estimators.

    Returns:
    -----------
    pd.DataFrame: A DataFrame containing the ROC AUC scores for each estimator, sorted in descending order.
    """
    
    results = {name: score 
               for (name, _), score in zip(estimators, auc_scores)}
        
    df = pd.DataFrame.from_dict(results, 
                                orient = "index", 
                                columns = ["ROC_AUC"])
    
    return df.sort_values(by = "ROC_AUC", ascending = False)
    
    