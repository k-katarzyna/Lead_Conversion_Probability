import os
from warnings import filterwarnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

filterwarnings("ignore")

from src.utils import load_city_list


PATH_TIER_1 = os.path.join("data", "cities", "tier_1.txt")
PATH_TIER_2 = os.path.join("data", "cities", "tier_2.txt")
TIER_1 = load_city_list(PATH_TIER_1)
TIER_2 = load_city_list(PATH_TIER_2)


def age_calculator(dates_of_birth, reference_dates):
    
    """
    Calculates age from date of birth using given reference dates.

    Args:
    -----------
    dates_of_birth (pd.Series): The column with dates of birth, which will be transformed to age.
    reference_dates (pd.Series): The column with reference dates (lead creation dates).
    
    Returns:
    ------------
    pd.Series: The column with calculated age.
    """
    
    dates_of_birth = pd.to_datetime(dates_of_birth)
    reference_dates = pd.to_datetime(reference_dates)
        
    dates_of_birth = dates_of_birth.apply(lambda x: x - pd.DateOffset(years = 100) 
                                          if x.year > 2014 
                                          else x)

    return (reference_dates - dates_of_birth).dt.days // 365


def data_preparing(data, basic_preparing = False):
    
    """
    Preprocesses and transforms the input DataFrame.

    Args:
    -------------
    data (pd.DataFrame): The input DataFrame containing raw data.
    basic_preparing: If True, performs only transformations provided as basic.

    Returns:
    -------------
    X (pd.DataFrame): A preprocessed DataFrame with modified and transformed columns.
    y (pd.Series): Target variable
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
                                  inplace = True)
                
        data[["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]].fillna(0, inplace = True)

        data["City"] = data["City"].apply(lambda x: 
                                          "Tier_1" if x in TIER_1 
                                          else "Tier_2" if x in TIER_2 
                                          else "Tier_3" if pd.notna(x) 
                                          else x)

        data.rename(columns = {"City": "City_Size"}, inplace = True)
        
    X = data.drop(["ID", "Lead_Creation_Date", "Salary_Account", "Device_Type", "LoggedIn", "Disbursed"], 
                  axis = 1)
    y = data.Disbursed

    return X, y


class RareAggregator(BaseEstimator, TransformerMixin):
    
    """
    Groups rare categories within categorical features. Categories with a frequency less than the specified threshold
    will be grouped into a single category. The transformer can be applied to all categorical features as it automatically
    selects features with more than two categories.
    The primary grouping is based on frequency, but different grouping options are available for cities.
    
    Parameters:
    -----------
    threshold: int, default = 1000
        The threshold below which categories are considered rare when grouping. Categories with a frequency
        less than this threshold will be grouped into a single 'Others' category.
    group_by: str, default = 'frequency'
        Supported values are 'frequency', 'tiers/freq', 'big_cities/tiers/freq'.
        - 'frequency': Groups rare categories based on their frequency.
        - 'tiers/freq': Groups all cities in the 'City' column using official Indian cities classification 
          (leaves missing values unchanged). Other features will be grouped using frequency.
        - 'big_cities/tiers/freq': Leaves the biggest cities and missing values in the 'City' column unchanged, 
          classifies smaller cities as Tier_2 and Tier_3. Other features will be grouped using frequency.
    rare_category_name: str, default = 'Others'
        The name used for grouping rare categories.

    Attributes:
    -----------
    features_to_group_by_frequency_: list
        A list of column names containing multilevel categorical features to group by frequency.
    frequent_categories_: dict
        A dictionary containing information about rare categories that were grouped. Keys represent column names,
        and values are lists of categories that were grouped into the 'Others' category for each respective column. 
    """

    def __init__(self,
                 threshold = 1000, 
                 group_by = "frequency",
                 rare_category_name = "Others"):

        self.threshold = threshold
        self.group_by = group_by
        self.rare_category_name = rare_category_name
        
    def frequency_fitting(self, X):
        
        self.frequent_categories_ = {}
        
        for col in self.features_to_group_by_frequency_:
            category_counts = X[col].value_counts()
            self.frequent_categories_[col] = category_counts.index[category_counts > self.threshold].to_list()

    def fit(self, X, y = None):
        
        self.features_to_group_by_frequency_ = [col for col in X.columns if X[col].nunique() > 2]
            
        if self.group_by == "tiers/freq" or self.group_by == "big_cities/tiers/freq":
            self.features_to_group_by_frequency_.remove("City")
            
        self.frequency_fitting(X)

        return self
    
    def frequency_encoding(self, X_transformed):
        
        for col in self.features_to_group_by_frequency_:
            X_transformed[col] = X_transformed[col].apply(lambda x: 
                                                          self.rare_category_name if x not in self.frequent_categories_[col] 
                                                          else x)
        return X_transformed

    def transform(self, X):
        
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
    how: str or NoneType, default = 'simple'
        The imputation strategy to be used. Supported values are 'simple', 'applied_submitted_compression' and None.
        - 'simple': Fills missing values with 0 for 'Loan_Amount_Applied', 'Loan_Tenure_Applied', and 'Existing_EMI',
              and fills other missing values with -1.
        - 'applied_submitted_compression': Fills missing values in 'Loan_Amount_Submitted' and 'Loan_Tenure_Submitted' 
              with corresponding values from 'Loan_Amount_Applied' and 'Loan_Tenure_Applied'. Removes 'Loan_Amount_Applied' 
              and 'Loan_Tenure_Applied' from the dataset. Fills missing 'Existing_EMI' with 0 and other missing values with -1.
        - None: imputation will be skipped.
    """
    
    def __init__(self, 
                 how = "simple"):
        
        self.how = how
        self.fillna_with_0 = ["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]

    def fit(self, X, y = None):
        
        return self

    def transform(self, X):

        X_transformed = X.copy()
        
        if self.how is None:
            return X_transformed
        
        X_transformed[self.fillna_with_0].fillna(0, inplace = True)

        if self.how == "simple":
            X_transformed.fillna(-1, inplace = True)

        if self.how == "applied_submitted_compression":
            
            X_transformed["Loan_Amount_Submitted"].fillna(X_transformed["Loan_Amount_Applied"],
                                                          inplace = True)
            X_transformed["Loan_Tenure_Submitted"].fillna(X_transformed["Loan_Tenure_Applied"],
                                                          inplace = True)
            
            X_transformed.drop(["Loan_Amount_Applied", "Loan_Tenure_Applied"], 
                               axis = 1, inplace = True)
            
            X_transformed.fillna(-1, inplace = True)
            
        return X_transformed


class ColumnRemover(BaseEstimator, TransformerMixin):
    
    """
    A transformer for removing specified features based on the results of feature importance tests.
    The feature names to drop should be specified as hard-coded strings based on the names used
    in the transformation pipeline.

    Parameters:
    -----------
    to_drop: int, default = 0 
        The number of least important features to drop (0-8 or more if longer list of features is given). 
        For example, if to_drop=2, it will drop the two least important features.
    least_important_features: list, default = ["num_pipe__Loan_Tenure_Applied",
                                             "num_pipe__EMI_Loan_Submitted",
                                             "cat_pipe__multi__Var1",
                                             "cat_pipe__multi__Var2",
                                             "cat_pipe__binary__Mobile_Verified_Y",
                                             "num_pipe__Loan_Amount_Submitted",
                                             "num_pipe__Interest_Rate",
                                             "num_pipe__Loan_Tenure_Submitted"]
        Features to drop (ascending order by feature importance).
    """
    
    def __init__(self, 
                 to_drop = 0,
                 least_important_features = ["num_pipe__Loan_Tenure_Applied",
                                             "num_pipe__EMI_Loan_Submitted",
                                             "cat_pipe__multi__Var1",
                                             "cat_pipe__multi__Var2",
                                             "cat_pipe__binary__Mobile_Verified_Y",
                                             "num_pipe__Loan_Amount_Submitted",
                                             "num_pipe__Interest_Rate",
                                             "num_pipe__Loan_Tenure_Submitted"]
                ):
        
        self.to_drop = to_drop
        self.least_important_features = least_important_features

    def fit(self, X, y = None): 
        
        return self

    def transform(self, X):
        
        features_to_drop = [col for col in self.least_important_features[:self.to_drop] 
                            if col in X.columns]

        return X.drop(features_to_drop, axis = 1)