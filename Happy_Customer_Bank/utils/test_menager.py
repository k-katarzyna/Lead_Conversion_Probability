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


def scores(pipeline, X, y, model_name, model_params, imputation):

    cv_scheme = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = cross_validate(pipeline,
                        X, y,
                        scoring="roc_auc",
                        cv=cv_scheme,
                        n_jobs=-1)

    return {"Model": model_name,
            "Parameters": model_params,
            "ROC_AUC": round(cv_results["test_score"].mean(), 4),
            "Imputation": imputation,
            "Time[s]": round(cv_results["fit_time"].mean(), 2)}


def collect_imputation_results(models, X, y, num_features, cat_features, preprocessors):

    results = []

    for model in models:
        
        model_name = model.__class__.__name__
        model_params = ", ".join([f"{param}: {getattr(model, param)}" for param in params_to_save if hasattr(model, param)])

        for imputer_name, preprocessor in preprocessors:

            preprocessor = ColumnTransformer([
                    ("numerical", preprocessor, num_features),
                    ("categorical", OneHotEncoder(), cat_features)],
                    remainder="passthrough")
            
            pipeline = make_pipeline(preprocessor, model)
            
            imputation = imputer_name
            result = scores(pipeline, X, y, model_name, model_params, imputation)
            results.append(result)

    return results


def collect_without_imputation_results(models, X, y, cat_features):

    results = []

    for model in models:
        
        model_name = model.__class__.__name__
        model_params = ", ".join([f"{param}: {getattr(model, param)}" for param in params_to_save if hasattr(model, param)])

        if model_name in missings_handling_models:
        
            preprocessor = ColumnTransformer([("categorical", OneHotEncoder(), cat_features)],
                        remainder="passthrough")
            imputation = "None"
            pipeline = make_pipeline(preprocessor, model)
            result = scores(pipeline, X, y, model_name, model_params, imputation)
            results.append(result)

    return results


def create_results_dataframe(*args):
    
    if len(args) == 1:
        df = pd.DataFrame(args[0])
        
    elif len(args) > 1:
        df = pd.concat([pd.DataFrame(arg) for arg in args])
        
    df.set_index("Model", inplace=True)
    
    return df
