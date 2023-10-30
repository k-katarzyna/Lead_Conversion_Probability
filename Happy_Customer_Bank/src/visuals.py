import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn import set_config
set_config(transform_output="pandas")

from src.utils import to_labels

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams.update({"axes.grid": True})
plt.rcParams["grid.linewidth"] = 0.2
plt.rcParams["grid.alpha"] = 0.5


def missings_plot(data):
    
    """
    Creates horizontal bar plots showing the number of missing values and zero values for selected features.

    Parameters:
    -----------
    data : pandas DataFrame
        The input DataFrame containing the data.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    features_with_nan = [feature for feature in data.columns if data[feature].isnull().sum() > 0]
    number_of_nan_values = [data[feature].isnull().sum() for feature in features_with_nan]

    ax1.barh(features_with_nan, number_of_nan_values, color="steelblue")
    ax1.set_title("Missing values")
    ax1.set_xscale("log")

    for number, feature in zip(number_of_nan_values, features_with_nan):
        ax1.annotate(number,(number, feature),
                     fontsize = 10,
                     va = "center", 
                     ha = "center",
                     bbox = dict(boxstyle="round",
                                 fc = "aliceblue"))

    features_with_zeros = ["Monthly_Income", "Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI", "Employer_Name"]
    number_of_zero_values = ((data[features_with_zeros] == 0) | (data[features_with_zeros] == "0")).sum()

    ax2.barh(features_with_zeros, number_of_zero_values, color = "lightseagreen")
    ax2.set_title("Zero values")
    ax2.set_xscale("log")

    for number, feature in zip(number_of_zero_values, features_with_zeros):
        ax2.annotate(number, (number, feature),
                     fontsize = 10,
                     va = "center",
                     ha = "center",
                     bbox = dict(boxstyle = "round",
                                 fc = "azure", 
                                 alpha = 0.5))

    plt.tight_layout()
    plt.show()


def histplots_grid(n_rows, n_cols, data, features = None):
    
    """
    Creates a grid of histograms.

    Args:
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        data (pd.DataFrame): The dataframe containing the data for plotting.
        features (list, optional): List of feature names to plot. 
            If not provided, it selects numeric features with more than 2 unique values.

    Number of rows and columns must correspond with the number of features.
    """

    if features is None:
        features = [feature for feature in data.select_dtypes([int, float]).columns 
                    if data[feature].nunique() > 2]
    
    width = n_cols * 4
    height = n_rows * 3
    
    plt.figure(figsize=(width, height))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(data[feature], color="steelblue")
        plt.title(feature)
        plt.locator_params(axis = 'x', nbins = 4)
        plt.locator_params(axis = 'y', nbins = 4)
        
        if feature in ["Monthly_Income", "Existing_EMI"]:
            plt.yscale("log")
        
    plt.tight_layout()    
    plt.show()


def countplots(*args, data):
    
    """
    Creates countplots for one or more categorical features.

    Args:
        *args (str): One or more feature names to create countplots for.
        data (pd.DataFrame): The dataframe containing the data to visualize.
    """

    for feature in args:
        plt.figure(figsize=(15, 3))
        order = data[feature].value_counts().index
        ax = sns.countplot(x = feature,
                           data = data,
                           order = order,
                           palette = "viridis")
        
        ax.set_yscale("log")
        plt.title(f'Distribution of {feature}')

        if len(str(data[feature].unique()[0])) >= 4:
            plt.xticks(rotation = 45)

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height),
                        ha = 'center', va = 'bottom', fontsize = 10)

        plt.tight_layout()
        plt.show()


def feature_importance_plot(importances, feature_names, title="Feature Importances"):
    
    """
    Creates horizontal barplot for feature importances.

    Args:
        importances (np.array): Feature importances values.
        feature_names (list): Feature names for plotting.
    """
    
    sorted_indices = importances.argsort()
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    plt.figure(figsize=(7, 6))
    plt.barh(range(len(sorted_names)), 
             sorted_importances, 
             align="center", color = "steelblue")
    
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def thresholds_results_plot(results, thresholds, optimal_thresholds):
    
    """
    Plots precision, recall, and F1 Score for different discrimination thresholds.

    Args:
    results (dict): A dictionary containing results for different estimators.
        Each key-value pair represents the name of an estimator and its corresponding scores.
        The scores includes precision, recall, and F1 Score.
    thresholds (np.array): An array of threshold values.
    optimal_thresholds (list): An array of optimal threshold values corresponding to each estimator.
    """
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()

    for i, (estimator_name, scores) in enumerate(results.items()): 

        f1_scores, precision_scores, recall_scores = scores
        max_f1_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_idx]

        ax[i].plot(thresholds, precision_scores, color="orange", label="Precision")
        ax[i].plot(thresholds, recall_scores, color="blue", label="Recall")
        ax[i].plot(thresholds, f1_scores, color="green", label="F1 Score")
        ax[i].scatter(thresholds[max_f1_idx], max_f1, c = "darkgreen", label = f"Max F1 = {max_f1:.2f}")

        ax[i].axvline(x=optimal_thresholds[i], color="black", linestyle="--", linewidth=0.8, label=f"Optimal threshold = {optimal_thresholds[i]}")

        ax[i].set_title(estimator_name)
        ax[i].set_xlabel("Threshold")
        ax[i].set_ylabel("Score")
        ax[i].set_xticks(np.arange(0, 1.1, 0.1))
        ax[i].set_yticks(np.arange(0, 1.1, 0.1))
        ax[i].legend()

    plt.tight_layout()
    plt.show()
    
    
def roc_curves(estimators, optimal_thresholds, X_train, X_test, y_train, y_test):
    
    """
    Plots ROC curves for each estimator, including the AUC (Area Under the Curve) score in the legend,
    and marks the optimal classification threshold for each estimator on the zoomed-in plot.

    The zoomed-in plot focuses on the top-left area of the ROC curves where the thresholds are marked.

    Args:
    estimators (list of tuples): List of tuples where each tuple contains: (estimator_name, estimator object).
    optimal_thresholds (list): List of optimal classification thresholds for each estimator.
    X_train (pd.DataFrame): Training data.
    X_test (pd.DataFrame): Testing data.
    y_train (pd.Series): Target values for training data.
    y_test (pd.Series): Target values for testing data.
    """

    fig, ax = plt.subplots(1, 2, figsize = (12,5))

    for (name, estimator), opt_threshold in zip(estimators, optimal_thresholds):

        estimator.fit(X_train, y_train)
        y_proba = estimator.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        idx = np.argmin(np.abs(thresholds - opt_threshold))
        fpr_value, tpr_value, threshold_value = fpr[idx], tpr[idx], thresholds[idx]

        ax[0].plot(fpr, tpr, label = f"{name} (AUC = {np.round(roc_auc_score(y_test, y_proba), 3)})")
        ax[1].plot(fpr, tpr)
        ax[1].scatter(fpr_value, tpr_value, s=150, label = f"{name} threshold = {np.round(threshold_value, 2)}")

    ax[0].plot([0, 1], [0, 1], linestyle = '--', color = 'grey', linewidth = 0.6, label = "No skill")
    ax[0].set_title("ROC curves")
    ax[1].plot([0, 1], [0, 1], linestyle = '--', color = 'grey', linewidth = 0.6)
    ax[1].set_xlim(0.0, 0.6)
    ax[1].set_ylim(0.4, 1.0)
    ax[1].set_title("ROC curves zoomed in at top left, optimal classification thresholds")

    for ax in [ax[0], ax[1]]:
        ax.legend()
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")

    plt.tight_layout()
    plt.show()
    

def classification_metrics(estimators, optimal_thresholds, X_train, X_test, y_train, y_test):
    
    """
    Calculates and compares several classification metrics, including Precision, Recall, F1 Score, Balanced Accuracy,
    and the percentage of positive class predictions for each estimator using their optimal discrimination thresholds.

    Args:
    estimators (list of tuples): List of tuples where each tuple contains: (estimator_name, estimator object).
    optimal_thresholds (list): List of optimal classification thresholds for each estimator.
    X_train (pd.DataFrame): Training data.
    X_test (pd.DataFrame): Testing data.
    y_train (pd.Series): Target values for training data.
    y_test (pd.Series): Target values for testing data.
    """
    
    metrics = ["Precision", "Recall", "F1 Score", "Balanced Accuracy", "% of positive class predictions"]

    results = []

    for (_, estimator), opt_threshold in zip(estimators, optimal_thresholds):
        estimator.fit(X_train, y_train)
        y_proba = estimator.predict_proba(X_test)[:, 1]
        y_pred = to_labels(y_proba, opt_threshold)

        scores = [precision_score(y_test, y_pred),
                  recall_score(y_test, y_pred),
                  f1_score(y_test, y_pred),
                  balanced_accuracy_score(y_test, y_pred),
                  np.mean(y_pred)]

        results.append(scores)

    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(12, 4))

    for i, (name, _) in enumerate(estimators):
        x = np.arange(len(metrics)) + i * bar_width
        bars = ax.bar(x, results[i], bar_width, label=name)

        for bar, value in zip(bars, results[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                    f"{value:.2f}", 
                    ha='center', va='bottom', 
                    bbox=dict(boxstyle = "round",
                              facecolor='white',
                              alpha=0.5))

    ax.set_xticks(np.arange(len(metrics)) + bar_width * (len(estimators) / 2))
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylabel("Score")
    ax.set_title("Comparison of model classification metrics using optimal discrimination thresholds")

    plt.tight_layout()
    plt.show()