import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

def missings_plot(data, return_features=False):
    """
    Creates horizontal bar plots showing the number of missing values and zero values for selected features.

    Parameters:
    -----------
    data : pandas DataFrame
        The input DataFrame containing the data.
    return_features : bool, optional (default=False)
        If True, returns a list of features with missing values.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    features_with_nan = [feature for feature in data.columns if data[feature].isnull().sum() > 0]
    number_of_nan_values = [data[feature].isnull().sum() for feature in features_with_nan]

    ax1.barh(features_with_nan, number_of_nan_values, color="steelblue")
    ax1.set_title("Missing values")
    ax1.set_xscale("log")

    for number, feature in zip(number_of_nan_values, features_with_nan):
        ax1.annotate(number, (number, feature),
                     fontsize=10,
                     va="center", ha="center",
                     bbox=dict(boxstyle="round", fc="aliceblue"))

    features_with_zeros = ["Monthly_Income", "Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI", "Employer_Name"]
    number_of_zero_values = ((data[features_with_zeros] == 0) | (data[features_with_zeros] == "0")).sum()

    ax2.barh(features_with_zeros, number_of_zero_values, color="lightseagreen")
    ax2.set_title("Zero values")
    ax2.set_xscale("log")

    for number, feature in zip(number_of_zero_values, features_with_zeros):
        ax2.annotate(number, (number, feature),
                     fontsize=10,
                     va="center", ha="center",
                     bbox=dict(boxstyle="round", fc="azure"))

    plt.tight_layout()
    plt.show()

    if return_features:
        return features_with_nan


def grid_of_hists(n_rows, n_cols, features, data):
	
    """ 
    Creates a grid of histograms. Inputs are: number of rows, number of columns, list of feature names, and dataframe. 
    Number of rows and columns must correspond with the number of features.
    """
	
    width = n_cols * 4
    height = n_rows * 3
    
    plt.figure(figsize=(width, height))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(data[feature], color="steelblue")
        plt.title(feature)
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        
        if feature in ["Monthly_Income", "Existing_EMI"]:
            plt.yscale("log")
        
    plt.tight_layout()    
    plt.show()


def countplot(*args, data):

    for feature in args:
        plt.figure(figsize=(15, 3))
        order = data[feature].value_counts().index
        ax = sns.countplot(x = feature, 
                           data = data, 
                           order = order,
						   palette = "viridis")
        ax.set_yscale("log")
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45)
    
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize = 10)
    
        plt.tight_layout()
        plt.show()