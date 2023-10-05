import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ZeroOneEncoder(BaseEstimator, TransformerMixin):

    """
    Transformer for encoding categorical features with two unique values or handling NaN values.

    This transformer automatically detects columns with NaN values and binary categorical columns.
    For columns with NaN values, it replaces NaN with 0 and non-NaN values with 1.
    For binary categorical columns, it encodes one value as 0 and the other as 1.

    Parameters:
    -----------
    None
    """

    def fit(self, X, y = None):
        self.columns_with_nan_ = [col for col in X.columns if X[col].isna().any()]
        self.binary_columns_ = [col for col in X.columns if X[col].nunique() == 2]
        return self

    def transform(self, X):
        transformed = X.copy()

        for col in self.columns_with_nan:
            transformed[col] = transformed[col].apply(
                lambda x:
                0 if pd.isna(x) or (str(x).isdigit())
                else 1
            )

        for col in self.binary_columns:
            unique_values = transformed[col].unique()
            transformed[col] = transformed[col].apply(
                lambda x:
                0 if x == unique_values[0]
                else 1
            )

        return transformed


class MultiTaskEncoder(BaseEstimator, TransformerMixin):
    
    """
    Transformer for encoding multilevel categorical features and optional rare category grouping.

    This transformer detects columns with more than two unique values and encodes them by assigning
    consecutive numerical values in the range (number of categories, 1) based on the frequency of each category. 
    If NaNs are present, they will be replaced by 0.

    Additionally, it handles binary categorical columns by encoding them as 2 and 1, wchich is also based on frequency
    (2 for more numerous category).

    Parameters:
    -----------
    group_rare : bool, default=False
        If True, rare categories are grouped in the multilevel categorical columns. Rare categories are defined as
        categories with a frequency of less than the specified threshold (rare_threshold).
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
    binary_columns_ : list
        A list of column names containing binary categorical features.
    rare_categories_ : dict
        A dictionary containing information about rare categories that were grouped. Keys represent column names,
        and values are lists of categories that were grouped into the "Others" category for each respective column.
    """

    def __init__(self, 
                 group_rare = False, 
                 rare_threshold = 1000, 
                 group_only = False):
        
        self.group_rare = group_rare
        self.rare_threshold = rare_threshold
        self.group_only = group_only

    def fit(self, X, y = None):
        self.multi_level_columns_ = [col for col in X.columns if X[col].nunique() > 2]
        self.binary_columns_ = [col for col in X.columns if X[col].nunique() == 2]

        if self.group_rare:
            self.rare_categories_ = {}
			
            for col in self.multi_level_columns_:
                category_counts = X[col].value_counts()
                self.rare_categories_[col] = category_counts[category_counts < self.rare_threshold].index

        return self

    def transform(self, X):
        transformed = X.copy()

        for col in self.multi_level_columns_:
            if self.group_rare and col in self.rare_categories_:
                transformed[col] = transformed[col].apply(
                    lambda x:
                    "Others" if x in self.rare_categories_[col] 
					else x
                )
                
                if self.group_only:
                    return transformed

            category_counts = transformed[col].value_counts()
            category_mapping = {category: len(category_counts) - i
                                for i, category in enumerate(category_counts.index)}
            transformed[col] = transformed[col].map(category_mapping)

            if transformed[col].isna().any():
                transformed[col] = transformed[col].fillna(0).astype(int)

        for col in self.binary_columns_:
            unique_values = transformed[col].value_counts().index
            transformed[col] = transformed[col].apply(
                lambda x:
                1 if x == unique_values[0]
                else 0
            )

        return transformed


class DateEncoder(BaseEstimator, TransformerMixin):
    
    """
    Transformer for encoding date columns.

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
    
    def __init__(self, 
				 extract_from_date = None, 
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
            return (reference_date - transformed).dt.days // 365

        elif self.extract_from_date == "day_of_year":
            return transformed.dt.dayofyear
			
        elif self.extract_from_date == "month":
            return transformed.dt.strftime("%B")
			
        else:
            return transformed
