import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from dateutil.relativedelta import relativedelta


class ZeroOneCategoryEncoder(BaseEstimator, TransformerMixin):

    """
    Transformer for:
	- encoding categorical features with many categories and NaN values, when it is assumed that the value may not be significant, but its presence is,
	- encoding categorical features with exactly 2 unique values.

    This transformer automatically detects columns with NaN values and binary categorical columns.
    For columns with NaN values, it replaces NaN with 0 and non-NaN values with 1.
    For binary categorical columns, it encodes one value as 0 and the other as 1.
    
    """
    
    def __init__(self):
        self.columns_with_nan = []
        self.binary_columns = []
    
    def fit(self, X, y=None):
        self.columns_with_nan = [col for col in X.columns if X[col].isna().any()]
        self.binary_columns = [col for col in X.columns if len(X[col].unique()) == 2]
        return self
    
    def transform(self, X):
        transformed = X.copy()
        
        for column in self.columns_with_nan:
            transformed[column] = transformed[column].apply(lambda x: 1 if pd.notna(x) else 0)
        
        for column in self.binary_columns:
            unique_values = transformed[column].unique()
            transformed[column] = transformed[column].apply(lambda x: 1 if x == unique_values[1] else 0)

        return transformed


class MultiLevelCategoryEncoder(BaseEstimator, TransformerMixin):
	
    """
    Transformer for encoding multilevel categorical features.

    This transformer detects columns with more than two unique values and encodes them by assigning
    consecutive numerical values in the range (1, number of categories) based on the frequency of each category. 
    If NaNs are present, they will be replaced by 0.

    Additionally, it handles binary categorical columns by encoding them as 0 and 1.
    """

    def __init__(self):
        self.multi_level_columns = []
        self.binary_columns = []

    def fit(self, X, y=None):
        self.multi_level_columns = [col for col in X.columns if X[col].nunique() > 2]
        self.binary_columns = [col for col in X.columns if X[col].nunique() == 2]
        return self

    def transform(self, X):
        transformed = X.copy()

        for column in self.multi_level_columns:
            category_counts = transformed[column].value_counts()
            category_mapping = {category: len(category_counts) - i for i, category in enumerate(category_counts.index)}
            transformed[column] = transformed[column].map(category_mapping)
            
            if transformed[column].isna().any():
                transformed[column] = transformed[column].fillna(0).astype(int)

        for column in self.binary_columns:
            unique_values = transformed[column].value_counts().index
            transformed[column] = transformed[column].apply(lambda x: 1 if x == unique_values[0] else 0)

        return transformed



class DateEncoder(BaseEstimator, TransformerMixin):
    
    """
    A transformer for encoding date columns.

    This transformer is designed to work with date columns. 
    It offers three encoding options: age calculation, day of the year extraction, and month extraction.

    Parameters:
    -----------
    extract_from_date : str, default=None
        Specify the type of information to extract from the date. Options: "age", "day_of_year", "month".
    date_for_calc : str, default="04-08-2015"
        A reference date in the format "dd-mm-yyyy". By default, age is calculated relative to this date 
        (the date the Happy Customer Bank task appeared).
    If extract_from_date is None, the transformer returns pandas datetime object.
    """
    
    def __init__(self, extract_from_date=None, date_for_calc="04-08-2015"):
        self.extract_from_date = extract_from_date
        self.date_for_calc = date_for_calc
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed = X.copy()
        transformed = pd.to_datetime(transformed)
        transformed = transformed.apply(lambda x: x - pd.DateOffset(years=100) if x.year > 2014 else x)
        
        if self.extract_from_date == "age":
            reference_date = pd.to_datetime(self.date_for_calc)
            return relativedelta(reference_date, transformed)
        
        elif self.extract_from_date == "day_of_year":
            return transformed.dt.dayofyear
			
        elif self.extract_from_date == "month":
            return transformed.dt.strftime("%B")
			
        else:
            return transformed
