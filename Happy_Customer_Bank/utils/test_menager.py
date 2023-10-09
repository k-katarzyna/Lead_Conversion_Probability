import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                             HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.feature_selection import SelectFromModel

cv_scheme = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

def collect_tests_results(X, y, test, models = None, preprocessors = None, grid_search = None):
    """
    Collects test results for different machine learning models.

    Args:
        models (list): List of models to evaluate.
        X (pd.DataFrame): Input data.
        y (pd.Series): Target labels.
        test (str): Type of test to perform, either "imputation", "cat_encoding", "feature_selection", "searching_params",.
        preprocessors (list): List of preprocessors.
        grid_search (tuple): (model, param_grid), only for "searching_params" test.

    Returns:
        list: List of dictionaries containing test results.
    """

    num_features = X.select_dtypes(float, int).columns
    cat_features = X.select_dtypes(object).columns

    general_preprocessor = make_column_transformer((SimpleImputer(strategy = "constant",
                                                                  fill_value = -1),
                                                    num_features),
                                                   (TargetEncoder(random_state = 42),
                                                    cat_features),
                                                    remainder = "passthrough")

    if models is not None:

        results = []
        params_to_save = ["class_weight", "min_samples_leaf", "n_estimators", "max_features"]
        missings_handling_models = ["BaggingClassifier", "HistGradientBoostingClassifier"]
    
        for model in models:
            model_name = model.__class__.__name__
            model_params = ", ".join([f"{param}: {getattr(model, param)}"
                                      for param in params_to_save
                                      if hasattr(model, param)])
    
            if test == "imputation" and preprocessors is None:
                if model_name in missings_handling_models:
                    
                    preprocessor = make_column_transformer((OneHotEncoder(), cat_features),
                                                           remainder="passthrough")
                    
                    pipeline = make_pipeline(preprocessor, model)
                    
                    result = scores(pipeline,
                                    X, y, 
                                    model_name, 
                                    model_params, 
                                    test = test, 
                                    imputation = "none")
    
                    results.append(result)
    
            if test == "imputation" and preprocessors is not None:
                for imputer_name, preprocessor in preprocessors:
                    
                    preprocessor = make_column_transformer((preprocessor, num_features),
                                                           (OneHotEncoder(), cat_features),
                                                           remainder = "passthrough")
    
                    pipeline = make_pipeline(preprocessor, model)
                    
                    result = scores(pipeline,
                                    X, y, 
                                    model_name, 
                                    model_params, 
                                    test = test, 
                                    imputation = imputer_name)
    
                    results.append(result)
    
            if test == "cat_encoding":
                for preprocessor in preprocessors:
                    
                    preprocessor = make_column_transformer((SimpleImputer(strategy = "constant",
                                                                          fill_value = -1),
                                                            num_features),
                                                           (preprocessor, cat_features),
                                                           remainder = "passthrough")
    
                    encoder = preprocessor.transformers[1][1].__class__.__name__
                    pipeline = make_pipeline(preprocessor, model)
                    
                    result = scores(pipeline,
                                    X, y, 
                                    model_name, 
                                    model_params, 
                                    test = test, 
                                    encoder = encoder)
    
                    results.append(result)
    
            if test == "feature_selection":
        
                preprocessor = general_preprocessor

                selection_thresholds = [0.005, 0.007, 0.01, 0.012, 0.014, 0.015, 0.017, 0.019, 0.024]
                
                for threshold in selection_thresholds:
                    
                    pipeline = make_pipeline(
                        preprocessor,
                        SelectFromModel(estimator = RandomForestClassifier(n_estimators = 100,
                                                                           class_weight = {1:75}, 
                                                                           min_samples_leaf = 50,
                                                                           max_features = None,
                                                                           max_samples = 0.3,
                                                                           random_state = 42, n_jobs = -1),
                                        threshold = threshold),
                        model)
    
                    result = scores(pipeline,
                                    X, y,
                                    model_name,
                                    model_params,
                                    test = test,
                                    threshold = threshold)
    
                    results.append(result)

        return results

    if test == "searching_params":

        if preprocessors is None:
            preprocessor = general_preprocessor
        else:
            preprocessor = preprocessors
        
        model, param_grid = grid_search

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        optimizer = GridSearchCV(pipeline, 
                                 param_grid, 
                                 cv = cv_scheme, 
                                 scoring = "roc_auc", 
                                 n_jobs = -1)
        optimizer.fit(X, y)
        best_params = optimizer.best_params_
        best_params["best_auc_score"] = np.round(optimizer.best_score_, 4)
        
        return best_params


def scores(pipeline, X, y, model_name, model_params, test, imputation = None, encoder = None, threshold = None):
    """
    Calculates cross validation scores and prepares a dictionary with the results.

    Args:
        preprocessor: Data preprocessor.
        model: Machine learning model to evaluate.
        X (pd.DataFrame): Input data.
        y (pd.Series): Target labels.
        model_name (str): Name of the machine learning model.
        model_params (str): Model parameters as a formatted string.
        test (str): Type of test, either "imputation", "encoders", "feature_selection".
        imputation (str): Imputation method (only for "imputation" test).
        encoder (str): Encoder used (only for "encoders" test).
        threshold (float): Feature selection threshold (only for "feature_selection" test).

    Returns:
        dict: Dictionary containing test results.
    """
    
    cv_results = cross_validate(pipeline, 
                                X, y, 
                                scoring = "roc_auc",
                                cv = cv_scheme, 
                                n_jobs = -1)
    
    roc_auc = np.round(cv_results["test_score"].mean(), 4)
    time = np.round((cv_results["fit_time"] + cv_results["score_time"]).mean(), 2)

    if test == "imputation":
        return {"Model": model_name,
                "Parameters": model_params,
                "ROC_AUC": roc_auc,
                "Imputation": imputation,
                "Time[s]": time}

    if test == "cat_encoding":
        return {"Model": model_name,
                "Parameters": model_params,
                "ROC_AUC": roc_auc,
                "Encoder": encoder,
                "Time[s]": time}

    if test == "feature_selection":

        pipeline.fit(X, y)
        selected = np.round(pipeline.steps[1][1].get_support().mean() * 100)
        rejected = np.where(~pipeline.steps[1][1].get_support())[0]
        
        return {"Model": model_name,
                "Parameters": model_params,
                "ROC_AUC": roc_auc,
                "Threshold": np.round(threshold, 4), 
                "Selected[%]": selected,
                "Rejected[idx]": rejected,
                "Time[s]": time}


def create_results_dataframe(*args):
    """
    Create a results DataFrame from a list or lists of dictionaries.

    Args:
        *args: Variable number of dictionaries containing results data.

    Returns:
        pd.DataFrame: A DataFrame containing the results with the model names as the index.
    """
    if len(args) == 1:
        return pd.DataFrame(args[0])
    elif len(args) > 1:
        return pd.concat([pd.DataFrame(arg) for arg in args])


def summarize_results(dataframe, column_to_group_by):
    """
    Display summary statistics for a DataFrame grouped by a specified column.

    This function takes a DataFrame and a column name as input and computes summary statistics
    (count, mean, min, max) for two specified columns ("ROC_AUC" and "Time[s]") after grouping
    the DataFrame by the specified column. Returns a styled DataFrame with 
    background gradient applied to the "mean_roc_auc," "max_roc_auc," "mean_time[s]," and "max_time[s]" columns.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data to be summarized.
    column : str
        The name of the column by which the DataFrame should be grouped.

    Returns:
    --------
    pandas.io.formats.style.Styler
        A styled DataFrame with summary statistics and background gradient for specific columns.
    """
   
    results = (dataframe
               .groupby(column_to_group_by)
               .agg(
                   {"ROC_AUC": [np.size, np.mean, np.min, np.max],
                    "Time[s]": [np.mean, np.min, np.max]
                    })
               .set_axis(["count", "mean_roc_auc", "min_roc_auc", "max_roc_auc", "mean_time[s]", "min_time[s]", "max_time[s]"],
                         axis = 1)
               .round(4))

    results[["mean_time[s]"]] = results[["mean_time[s]"]].round(2)
    results = results.sort_values(by = ["max_roc_auc", "mean_roc_auc"], ascending = False)

    return results.style.background_gradient(subset = ["mean_roc_auc", "max_roc_auc", "mean_time[s]", "max_time[s]"])
