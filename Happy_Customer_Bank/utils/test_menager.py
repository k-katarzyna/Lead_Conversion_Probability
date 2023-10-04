import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                             HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler


params_to_save = ["class_weight", "min_samples_leaf", "max_depth", "n_estimators"]
missings_handling_models = ["DecisionTreeClassifier", "BaggingClassifier", "HistGradientBoostingClassifier"]


def collect_tests_results(models, X, y, test="imputation", preprocessors=None):
    """
    Collects test results for different machine learning models.

    Args:
        models (list): List of models to evaluate.
        X (pd.DataFrame): Input data.
        y (pd.Series): Target labels.
        test (str): Type of test to perform, either "imputation" or "encoders".
        preprocessors (list): List of preprocessors.

    Returns:
        list: List of dictionaries containing test results.
    """
    results = []

    num_features = X.select_dtypes(float, int).columns
    cat_features = X.select_dtypes(object).columns

    for model in models:
        model_name = model.__class__.__name__
        model_params = ", ".join([f"{param}: {getattr(model, param)}"
                                  for param in params_to_save
                                  if hasattr(model, param)])

        if preprocessors is None and test == "imputation":
            if model_name in missings_handling_models:
                preprocessor = ColumnTransformer([
                    ("categorical", 
                     OneHotEncoder(), 
                     cat_features)],
                    remainder="passthrough")

                result = scores(preprocessor, 
                                model, 
                                X, y, 
                                model_name, 
                                model_params, 
                                test="imputation", 
                                imputation="none")

                results.append(result)

        if preprocessors is not None and test == "imputation":
            for imputer_name, preprocessor in preprocessors:
                preprocessor = ColumnTransformer([
                    ("numerical", 
                     preprocessor, 
                     num_features),
                    ("categorical", 
                     OneHotEncoder(), 
                     cat_features)],
                    remainder="passthrough")

                result = scores(preprocessor,
                                model, 
                                X, y, 
                                model_name, 
                                model_params, 
                                test="imputation", 
                                imputation=imputer_name)

                results.append(result)

        if preprocessors is not None and test == "encoders":
            for preprocessor in preprocessors:
                preprocessor = ColumnTransformer([
                    ("numerical", 
                     SimpleImputer(strategy="constant", 
                                   fill_value=-1, 
                                   add_indicator=True), 
                     num_features),
                    ("categorical", 
                     preprocessor, 
                     cat_features)],
                    remainder="passthrough")

                encoder = preprocessor.transformers[1][1].__class__.__name__

                result = scores(preprocessor, 
                                model, 
                                X, y, 
                                model_name, 
                                model_params, 
                                test="encoders", 
                                encoder=encoder)

                results.append(result)

    return results


def scores(preprocessor, model, X, y, model_name, model_params, test, imputation=None, encoder=None):
    """
    Calculates cross validation scores and prepares a dictionary with the results.

    Args:
        preprocessor: Data preprocessor.
        model: Machine learning model to evaluate.
        X (pd.DataFrame): Input data.
        y (pd.Series): Target labels.
        model_name (str): Name of the machine learning model.
        model_params (str): Model parameters as a formatted string.
        test (str): Type of test, either "imputation" or "encoders".
        imputation (str): Imputation method (only for "imputation" test).
        encoder (str): Encoder used (only for "encoders" test).

    Returns:
        dict: Dictionary containing test results.
    """
    pipeline = make_pipeline(preprocessor, model)
    cv_scheme = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = cross_validate(pipeline, 
                                X, y, 
                                scoring="roc_auc", 
                                cv=cv_scheme, 
                                n_jobs=-1)

    if test == "imputation":
        return {"Model": model_name,
                "Parameters": model_params,
                "ROC_AUC": round(cv_results["test_score"].mean(), 4),
                "Imputation": imputation,
                "Time[s]": round(cv_results["fit_time"].mean(), 2)}

    if test == "encoders":
        return {"Model": model_name,
                "Parameters": model_params,
                "ROC_AUC": round(cv_results["test_score"].mean(), 4),
                "Encoder": encoder,
                "Time[s]": round(cv_results["fit_time"].mean(), 2)}


def create_results_dataframe(*args):
    """
    Create a results DataFrame from a list or lists of dictionaries.

    Args:
        *args: Variable number of dictionaries containing results data.

    Returns:
        pd.DataFrame: A DataFrame containing the results with the model names as the index.
    """
    if len(args) == 1:
        df = pd.DataFrame(args[0])
    elif len(args) > 1:
        df = pd.concat([pd.DataFrame(arg) for arg in args])

    df.set_index("Model", inplace=True)

    return df
