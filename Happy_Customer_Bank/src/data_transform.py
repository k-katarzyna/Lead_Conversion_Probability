import os
from warnings import filterwarnings

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils import load_city_list

filterwarnings("ignore")


PATH_TIER_1 = os.path.join("data", "cities", "tier_1.txt")
PATH_TIER_2 = os.path.join("data", "cities", "tier_2.txt")
TIER_1 = load_city_list(PATH_TIER_1)
TIER_2 = load_city_list(PATH_TIER_2)


def age_calculator(dates_of_birth, reference_dates):
    """
    Calculate age from date of birth using given reference dates.

    Args:
        dates_of_birth (pd.Series): A pandas Series containing dates of birth.
        reference_dates (pd.Series): A pandas Series containing reference dates 
            (such as lead creation dates).

    Returns:
        pd.Series: A pandas Series containing the calculated age. The age is 
            calculated as the difference in years between the reference date
            and the date of birth.
    """
    dates_of_birth = pd.to_datetime(dates_of_birth)
    reference_dates = pd.to_datetime(reference_dates)
        
    dates_of_birth = dates_of_birth.apply(lambda x: x - pd.DateOffset(years=100) 
                                          if x.year > 2014 
                                          else x)

    return (reference_dates - dates_of_birth).dt.days // 365


def data_preparing(data, basic_preparing=False):
    """
    Preprocess and transform the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing raw data.
        basic_preparing (bool, optional): If True, performs only the transformations 
            that are considered basic. The default value is False.

    Returns:
        pd.DataFrame: A preprocessed DataFrame with modified and transformed columns.
        pd.Series: The target variable extracted from the input DataFrame.
    """
    data = data.copy()
    
    data["DOB"] = age_calculator(data["DOB"], data["Lead_Creation_Date"])
    
    data["Employer_Name"] = data["Employer_Name"].apply(lambda x: 
                                                        0 if pd.isna(x) or (str(x).isdigit()) 
                                                        else 1)
    
    data.rename(columns = {"DOB": "Age",
                           "Employer_Name": "Employer_Provided"},
                inplace = True)
    
    if not basic_preparing:
        
        threshold = 1000
        for feature in ["Var1", "Var2", "Source"]:
            category_counts = data[feature].value_counts()
            rare_values = category_counts[category_counts < threshold].index.tolist()
            data[feature].replace(rare_values, "Others", 
                                  inplace=True)
                
        data[["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]].fillna(0, inplace=True)

        data["City"] = data["City"].apply(lambda x: 
                                          "Tier_1" if x in TIER_1 
                                          else "Tier_2" if x in TIER_2 
                                          else "Tier_3" if pd.notna(x) 
                                          else x)
        data.rename(columns={"City": "City_Size"}, inplace=True)
        
    X = data.drop(["ID", "Lead_Creation_Date", "Salary_Account",
                   "Device_Type", "LoggedIn", "Disbursed"], 
                  axis=1)
    y = data.Disbursed

    return X, y


class RareAggregator(BaseEstimator, TransformerMixin):
    """
    Transformer for grouping rare categories within categorical features. 
    
    Categories with a frequency less than the specified threshold will be grouped
    into a single category. The transformer can be applied to all categorical
    features as it automatically selects features with more than two categories. 
    The primary grouping is based on frequency, but different grouping options are
    available for cities.
    
    Parameters:
    -----------
    threshold: int, default=1000
        The frequency threshold for grouping. Categories with a frequency below this 
        are considered rare and grouped into a single 'Others' category.
    group_by: str, default='frequency'
        The method for grouping rare categories. Options include:
        - 'frequency': Groups rare categories based on their frequency across all 
           categorical features.
        - 'tiers/freq': In the 'City' column, groups cities based on the official 
           Indian city classification, while other features are grouped by frequency.
        - 'big_cities/tiers/freq': In the 'City' column, keeps the largest cities and 
           missing values unchanged, groups smaller cities into Tier_2 and Tier_3. 
           Other features are grouped by frequency.
    rare_category_name: str, default='Others'
        The name assigned to the grouped category of rare values.

    Attributes:
    -----------
    features_to_group_by_frequency_: list
        A list of column names containing multilevel categorical features to group
        by frequency.
    frequent_categories_: dict
        A dictionary mapping column names to lists of categories that were grouped into 
        the 'Others' category. Each key represents a column, and the corresponding value 
        is a list of the rare categories grouped in that column.
    """

    def __init__(self,
                 threshold=1000,
                 group_by="frequency",
                 rare_category_name="Others"):
        """Initialize the transformer with given parameters."""
        
        self.threshold = threshold
        self.group_by = group_by
        self.rare_category_name = rare_category_name
        
    def frequency_fitting(self, X):
        """Determine frequent categories in each feature based on the threshold."""
        
        self.frequent_categories_ = {}
        
        for col in self.features_to_group_by_frequency_:
            category_counts = X[col].value_counts()
            self.frequent_categories_[col] = category_counts.index[category_counts > self.threshold].to_list()

    def fit(self, X, y=None):
        """Fit the transformer to the data by identifying features to group by frequency."""
        
        self.features_to_group_by_frequency_ = [col for col in X.columns if X[col].nunique() > 2]
            
        if self.group_by == "tiers/freq" or self.group_by == "big_cities/tiers/freq":
            self.features_to_group_by_frequency_.remove("City")
            
        self.frequency_fitting(X)
        
        return self
    
    def frequency_encoding(self, X_transformed):
        """Apply frequency-based encoding to the given DataFrame."""
        
        for col in self.features_to_group_by_frequency_:
            X_transformed[col] = X_transformed[col].apply(lambda x: 
                                                          self.rare_category_name if x not in self.frequent_categories_[col] 
                                                          else x)
        return X_transformed

    def transform(self, X):
        """Transform the data by grouping rare categories and applying special rules for cities."""
        
        X_transformed = X.copy()
            
        if self.group_by == "tiers/freq":
            X_transformed["City"] = X_transformed["City"].apply(lambda x:
                                                                "Tier_1" if x in TIER_1 
                                                                else "Tier_2" if x in TIER_2 
                                                                else "Tier_3" if pd.notna(x) 
                                                                else x)
        if self.group_by == "big_cities/tiers/freq":
            X_transformed["City"] = X_transformed["City"].apply(lambda x: 
                                                                x if x in TIER_1 or pd.isna(x)
                                                                else "Tier_2" if x in TIER_2 
                                                                else "Tier_3")
            
        self.frequency_encoding(X_transformed)
         
        return X_transformed
    

class MixedImputer(BaseEstimator, TransformerMixin):
    """
    Transformer for imputing missing values.

    Parameters:
    -----------
    how: str or NoneType, default='simple'
        The strategy used for imputation. Supported strategies:
        - 'simple': Imputes missing values in 'Loan_Amount_Applied', 
          'Loan_Tenure_Applied', and 'Existing_EMI' with 0. All other missing
          values are filled with -1.
        - 'applied_submitted_compression': Imputes missing values in 
          'Loan_Amount_Submitted' and 'Loan_Tenure_Submitted' using corresponding
          values from 'Loan_Amount_Applied' and 'Loan_Tenure_Applied', respectively.
          After imputation, 'Loan_Amount_Applied' and 'Loan_Tenure_Applied' are
          removed from the dataset. 'Existing_EMI' missing values are filled with 0,
          and all other missing values with -1.
        - None: If set to None, imputation is not performed.
    """
    
    def __init__(self, how="simple"):
        """Initialize the imputer with the specified imputation strategy."""
        
        self.how = how
        self.fillna_with_0 = ["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]

    def fit(self, X, y=None):
        """Fit the imputer to the data (no action needed)."""
        
        return self

    def transform(self, X):
        """Apply the imputation strategy to the provided data."""
        
        X_transformed = X.copy()
        
        if self.how is None:
            return X_transformed
        
        X_transformed[self.fillna_with_0].fillna(0, inplace=True)

        if self.how == "simple":
            X_transformed.fillna(-1, inplace=True)

        if self.how == "applied_submitted_compression":
            
            X_transformed["Loan_Amount_Submitted"].fillna(X_transformed["Loan_Amount_Applied"],
                                                          inplace=True)
            X_transformed["Loan_Tenure_Submitted"].fillna(X_transformed["Loan_Tenure_Applied"],
                                                          inplace=True)
            X_transformed.drop(["Loan_Amount_Applied", "Loan_Tenure_Applied"], 
                               axis=1, inplace=True)
            X_transformed.fillna(-1, inplace=True)
            
        return X_transformed


class ColumnRemover(BaseEstimator, TransformerMixin):
    """
    Transformer for removing specified features based on their importance ranking.
    
    The features are removed from a predefined list, which should be ordered by increasing 
    importance.

    Parameters:
    -----------
    to_drop: int, default=0 
        The number of least important features to remove. The range is 0-8 or more, 
        depending on the length of the 'least_important_features' list. If to_drop=2, 
        the two least important features from the list are removed. A value of 0 means 
        no features will be dropped.
    least_important_features: list, default=["num_pipe__Loan_Tenure_Applied",
                                             "num_pipe__EMI_Loan_Submitted",
                                             "cat_pipe__multi__Var1",
                                             "cat_pipe__multi__Var2",
                                             "cat_pipe__binary__Mobile_Verified_Y",
                                             "num_pipe__Loan_Amount_Submitted",
                                             "num_pipe__Interest_Rate",
                                             "num_pipe__Loan_Tenure_Submitted"]
        A list of features to be considered for removal, sorted in ascending order
        of importance. The 'to_drop' parameter will remove features from the beginning
        of this list.
    """
    
    def __init__(self, 
                 to_drop=0,
                 least_important_features=["num_pipe__Loan_Tenure_Applied",
                                           "num_pipe__EMI_Loan_Submitted",
                                           "cat_pipe__multi__Var1",
                                           "cat_pipe__multi__Var2",
                                           "cat_pipe__binary__Mobile_Verified_Y",
                                           "num_pipe__Loan_Amount_Submitted",
                                           "num_pipe__Interest_Rate",
                                           "num_pipe__Loan_Tenure_Submitted"]
                ):
        """
        Initialize the transformer with the number of features to drop and a list
        of least important features.
        """
        self.to_drop = to_drop
        self.least_important_features = least_important_features

    def fit(self, X, y=None):
        """Fit the transformer to the data (no action needed)."""
        
        return self

    def transform(self, X):
        """Transform the DataFrame by dropping the specified least important features."""
        
        features_to_drop = [col for col in self.least_important_features[:self.to_drop] 
                            if col in X.columns]

        return X.drop(features_to_drop, axis=1)