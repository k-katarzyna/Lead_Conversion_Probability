import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# based on: https://www.loomsolar.com/blogs/collections/list-of-cities-in-india
biggest_cities = ["Bangalore", "Delhi", "Chennai", "Hyderabad", "Mumbai", "Pune", "Kolkata", "Ahmedabad"]


def data_preparing_v1(data):
    """
    Preprocess and transform the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing raw data.

    Returns:
        pd.DataFrame: A preprocessed DataFrame with modified and transformed columns.
    """
    data = data.copy()

    data.drop(["ID", "Lead_Creation_Date", "Device_Type", "Salary_Account"], axis = 1, inplace = True)
    data[["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]] = data[["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]].fillna(0)

    data["City"] = data["City"].apply(
        lambda x: "Y" if x in biggest_cities else "N" if pd.notna(x) else x)

    data["DOB"] = data["DOB"].apply(lambda x: int(x[-2:]))

    data["Employer_Name"] = data["Employer_Name"].apply(lambda x: 0 if pd.isna(x) or (str(x).isdigit()) else 1)

    for feature, value in zip(["Var1", "Var2", "Source"], [1000] * 3):
        rare_values = data[feature].value_counts()[data[feature].value_counts() < value].index.tolist()
        data[feature].replace(rare_values, "Others", inplace = True)

    data.rename(columns={"DOB": "Year_Of_Birth",
                         "City": "Is_Big_City",
                         "Employer_Name": "Employer_Provided"},
                inplace=True)

    return data


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    
    """
    Transformer for encoding categorical features and with rare categories grouping.

    This transformer encodes categories by assigning consecutive numerical values in the range (number of categories, 1) based on the frequency of each category. 
    If NaNs are present, they will be replaced by -1.

    Parameters:
    -----------
    rare_threshold : int, default=1000
        The threshold below which categories are considered rare when grouping. Categories with a frequency
        less than this threshold will be grouped into a single "Rare" category.
    group_only : bool, default=False
        If used, the 'group_rare' parameter must also be set to true.
        If True, the transformer will only perform rare category grouping and return the transformed column(s)
        without additional encoding. If set to False, the column(s) will be both grouped (if group_rare is True)
        and encoded.

    Attributes:
    -----------
    multi_level_columns_ : list
        A list of column names containing multilevel categorical features.
    rare_categories_ : dict
        A dictionary containing information about rare categories that were grouped. Keys represent column names,
        and values are lists of categories that were grouped into the "Others" category for each respective column.
    """

    def __init__(self,
                 rare_threshold = 1000, 
                 group_only = False,
                 group_by = "frequency"):

        self.rare_threshold = rare_threshold
        self.group_only = group_only
        self.group_by = group_by

    def fit(self, X, y = None):

        self.multi_level_columns_ = [col for col in X.columns if X[col].nunique() > 2]
        self.rare_categories_ = {}
            
        for col in self.multi_level_columns_:
            category_counts = X[col].value_counts()
            self.rare_categories_[col] = category_counts.index[category_counts <= self.rare_threshold]

        return self

    def transform(self, X):
        X_transformed = X.copy()

        for col in self.multi_level_columns_:
            X_transformed[col] = X_transformed[col].apply(
                lambda x: "Others" if x in self.rare_categories_[col] else x
            )
                
        if not self.group_only:
                
            for col in X_transformed.columns:
                category_counts = X_transformed[col].value_counts()
                category_mapping = {category: len(category_counts) - i
                                    for i, category in enumerate(category_counts.index)}
                X_transformed[col] = X_transformed[col].map(category_mapping)
    
                if X_transformed[col].isna().any():
                    X_transformed[col] = X_transformed[col].fillna(0).astype(int)
            
        return pd.DataFrame(X_transformed)


class MixedImputer(BaseEstimator, TransformerMixin):
    
    """
    Transformer for imputing missing values. It can handle both simple imputation, where missing values are filled with specified values,
    and imputation based on a compression of applied and submitted loan features.

    Parameters:
    -----------
    how : str, default="simple"
        The imputation strategy to be used. Supported values are "simple" and "applied_submitted_compression".
        - "simple": Fills missing values with 0 for 'Loan_Amount_Applied', 'Loan_Tenure_Applied', and 'Existing_EMI',
          and fills other missing values with -1.
        - "applied_submitted_compression": Fills missing values in 'Loan_Amount_Submitted' and 'Loan_Tenure_Submitted' 
          with corresponding values from 'Loan_Amount_Applied' and 'Loan_Tenure_Applied'. Removes 'Loan_Amount_Applied' 
          and 'Loan_Tenure_Applied' from the dataset. Fills missing 'Existing_EMI' with 0 and other missing values with -1.
    """
    
    def __init__(self, how = "simple"):
        self.how = how

    def fit(self, X, y = None):
        return self

    def transform(self, X):

        X_transformed = X.copy()

        if self.how == "simple":
            X_transformed[["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]].fillna(0, inplace = True)
            X_transformed.fillna(-1, inplace = True)

        if self.how == "applied_submitted_compression":
            
            X_transformed["Loan_Amount_Submitted"].fillna(X_transformed["Loan_Amount_Applied"], inplace=True)
            X_transformed["Loan_Tenure_Submitted"].fillna(X_transformed["Loan_Tenure_Applied"], inplace=True)
            
            X_transformed.drop(["Loan_Amount_Applied", "Loan_Tenure_Applied"], 
                               axis = 1, inplace = True)
            
            X_transformed["Existing_EMI"].fillna(0, inplace = True)
            X_transformed.fillna(-1, inplace = True)

        if self.how is None:
            pass
            
        return X_transformed


class DateEncoder(BaseEstimator, TransformerMixin):
    
    """
    Transformer for age calculation in date column.

    Parameters:
    -----------
    extract_from_date : str, default=None
        Specify the type of information to extract from the date. Options: "age".
    date_for_calc : str, default="04-08-2015"
        A reference date in the format "dd-mm-yyyy". By default, age is calculated relative to this date 
        (the date the Happy Customer Bank task appeared).

    """
    
    def __init__(self, 
				 extract_from_date = "age", 
				 date_for_calc = "04-08-2015"):
		
        self.extract_from_date = extract_from_date
        self.date_for_calc = date_for_calc
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        transformed = X.copy()
        transformed = pd.to_datetime(transformed)
        
        if self.extract_from_date == "age":
            transformed = transformed.apply(
				lambda x: x - pd.DateOffset(years=100) if x.year > 2014 else x
			)
            reference_date = pd.to_datetime(self.date_for_calc)
            return pd.DataFrame((reference_date - transformed).dt.days // 365)


class ZeroOneEncoder(BaseEstimator, TransformerMixin):

    """
    Transformer for categorical features with missing values, where only whether the value is defined or not may be relevant.
    It automatically detects columns with NaN values.
    It replaces NaN with 0 and non-NaN values with 1.

    Parameters:
    -----------
    None
    """

    def fit(self, X, y = None):
        self.columns_with_nan_ = [col for col in X.columns if X[col].isna().any()]
        return self

    def transform(self, X):
        transformed = X.copy()

        for col in self.columns_with_nan_:
            transformed[col] = transformed[col].apply(
                lambda x:
                0 if pd.isna(x) or (str(x).isdigit()) # "0" and other uncorrect digital values in Employer_Name
                else 1
            )

        return transformed
