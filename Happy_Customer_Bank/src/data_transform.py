from warnings import filterwarnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

filterwarnings("ignore")

from src.cities_lists import TIER_1, TIER_2


def age_calculator(dob_column, date_for_calc = "04-08-2015"):
    
    """
    Calculates age from date of birth.

    Args:
    -----------
    dob_column (pd.Series): The column with dates of birth.
    date_for_calc (str): default="04-08-2015"
        A reference date in the format "dd-mm-yyyy". By default, age is calculated relative to 
        the date the Happy Customer Bank task appeared.
    """
    
    dob_column = pd.to_datetime(dob_column)
        
    dob_column = dob_column.apply(
        lambda x: x - pd.DateOffset(years=100) if x.year > 2014 else x
    )
    reference_date = pd.to_datetime(date_for_calc)

    return (reference_date - dob_column).dt.days // 365


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
    
    X = data.drop(["ID", "Lead_Creation_Date", "Salary_Account", "Device_Type", "LoggedIn", "Disbursed"], axis = 1)
    y = data.Disbursed
    
    X["DOB"] = age_calculator(X["DOB"])
    X["Employer_Name"] = X["Employer_Name"].apply(lambda x: 
                                                  0 if pd.isna(x) or (str(x).isdigit()) 
                                                  else 1)
    
    X.rename(columns={"DOB": "Age", 
                      "Employer_Name": "Employer_Provided"},
             inplace = True)
    
    if not basic_preparing:
        
        for feature, value in zip(["Var1", "Var2", "Source"], [1000] * 3):
            rare_values = X[feature].value_counts()[data[feature].value_counts() < value].index.tolist()
            X[feature].replace(rare_values, "Others", inplace = True)
                
        X[["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]].fillna(0, inplace = True)

        X["City"] = X["City"].apply(lambda x: 
                                    "Tier_1" if x in TIER_1 
                                    else "Tier_2" if x in TIER_2 
                                    else "Tier_3" if pd.notna(x) 
                                    else x)

        X.rename(columns={"City": "City_Size"}, inplace = True)

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
        less than this threshold will be grouped into a single "Others" category.
    group_by: str, default = "frequency"
        Supported values are "frequency", "tiers/freq", "big_cities/tiers/freq".
        - "frequency": Groups rare categories based on their frequency.
        - "tiers/freq": Groups all cities in the "City" column using official Indian cities classification 
          (leaves missing values unchanged). Other features will be grouped using frequency.
        - "big_cities/tiers/freq": Leaves the biggest cities and missing values in the "City" column unchanged, 
          classifies smaller cities as Tier_2 and Tier_3. Other features will be grouped using frequency.
    rare_category_name: str, default = "Others"
        The name used for grouping rare categories.

    Attributes:
    -----------
    features_to_group_by_frequency_: list
        A list of column names containing multilevel categorical features to group by frequency.
    frequent_categories_: dict
        A dictionary containing information about rare categories that were grouped. Keys represent column names,
        and values are lists of categories that were grouped into the "Others" category for each respective column. 
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
            
        return self.frequent_categories_

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
    how: str or NoneType, default="simple"
        The imputation strategy to be used. Supported values are "simple", "applied_submitted_compression" and None.
        - "simple": Fills missing values with 0 for 'Loan_Amount_Applied', 'Loan_Tenure_Applied', and 'Existing_EMI',
              and fills other missing values with -1.
        - "applied_submitted_compression": Fills missing values in 'Loan_Amount_Submitted' and 'Loan_Tenure_Submitted' 
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
            
            X_transformed["Loan_Amount_Submitted"].fillna(X_transformed["Loan_Amount_Applied"], inplace=True)
            X_transformed["Loan_Tenure_Submitted"].fillna(X_transformed["Loan_Tenure_Applied"], inplace=True)
            
            X_transformed.drop(["Loan_Amount_Applied", "Loan_Tenure_Applied"], 
                               axis = 1, inplace = True)
            
            X_transformed.fillna(-1, inplace = True)
            
        return X_transformed


class ColumnRemover(BaseEstimator, TransformerMixin):
    
    """
    A transformer for removing specified features based on the results of feature importance tests.
    The feature names to drop are specified as hard-coded strings based on the names used
    in the transformation pipeline.

    Parameters:
    -----------
    to_drop: int, default = 0 
        The number of least important features to drop (0-8). For example, if to_drop=2, it will drop
        the two least important features.

    """
    
    def __init__(self, 
                 to_drop = 0):
        
        self.to_drop = to_drop

    def fit(self, X, y = None): 
        return self
    
    def drop_features(self, X_transformed, features):
        
        for feature in features:
            if feature not in X_transformed.columns:

                features.remove(feature)
        
        return X_transformed.drop(features, axis = 1, inplace = True)

    def transform(self, X):
        
        X_transformed = X.copy()
        
        features_to_drop = ["num_pipe__Loan_Tenure_Applied",
                            "num_pipe__EMI_Loan_Submitted",
                            "cat_pipe__multi__Var1",
                            "cat_pipe__multi__Var2",
                            "cat_pipe__binary__Mobile_Verified_Y",
                            "num_pipe__Loan_Amount_Submitted",
                            "num_pipe__Interest_Rate",
                            "num_pipe__Loan_Tenure_Submitted"]
        
        if self.to_drop > 0:
            self.drop_features(X_transformed,
                               features_to_drop[:self.to_drop])

        return X_transformed