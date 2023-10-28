import pandas as pd, numpy as np

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn import set_config

from joblib import Parallel, delayed, dump

from warnings import filterwarnings
filterwarnings("ignore")

from src.utils import to_labels
from src.visuals import thresholds_results_plot


cv_scheme = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)


def run_test(X, y, test, models = None, preprocessors = None, search = None, feat_sel_estimator = None):
    
    """
    Runs tests and collects results for various experiments.

    Args:
        models (list): List of models to evaluate.
        X (pd.DataFrame): Input data.
        y (pd.Series): Target labels.
        test (str): Type of test to perform, either "imputation", "cat_encoding", "feature_selection",
        "grid_search", "randomized_search".
        preprocessors (list): List of preprocessors.
        search (tuple): (model, param_grid), only for "grid_search" and "randomized_search" test.
        feat_sel_estimator: estimator for feature selection test

    Returns:
        list: List of dictionaries containing test results.
    """

    num_features = X.select_dtypes("number").columns
    cat_features = X.select_dtypes("object").columns

    general_preprocessor = make_column_transformer((SimpleImputer(strategy = "constant",
                                                                  fill_value = -1),
                                                    num_features),
                                                   (TargetEncoder(random_state = 42),
                                                    cat_features))

    if models is not None:

        results = []
        params_to_save = ["n_estimators", "class_weight", "min_samples_leaf", "max_samples"]
    
        for model in models:
            model_name = model.__class__.__name__
            
            if model_name == "BalancedBaggingClassifier":
                if getattr(model, "sampler").__class__.__name__ == "RandomUnderSampler":
                    model_name = "BalancedBagging_UnderSampling"
                else:
                    model_name = "BalancedBagging_OverSampling"
                
                
            model_params = ", ".join([f"{param}: {getattr(model, param)}"
                                      for param in params_to_save
                                      if hasattr(model, param)])

    
            if test == "imputation":

                try:         
                    for imputer, preprocessor in preprocessors:
    
                        preprocessor = make_column_transformer((preprocessor, num_features),
                                                               (OneHotEncoder(), cat_features))
        
                        pipeline = make_pipeline(preprocessor, model)
                        
                        result = cv_scores(pipeline,
                                           X, y, 
                                           model_name, 
                                           model_params, 
                                           test = test, 
                                           imputation = imputer)
    
                        results.append(result)

                except:
                    pass
    
            if test == "cat_encoding":
                for encoder, preprocessor in preprocessors:
                    
                    preprocessor = make_column_transformer((SimpleImputer(strategy = "constant",
                                                                          fill_value = -1),
                                                            num_features),
                                                           (preprocessor, cat_features))
    
                    pipeline = make_pipeline(preprocessor, model)
                    
                    result = cv_scores(pipeline,
                                       X, y, 
                                       model_name, 
                                       model_params, 
                                       test = test, 
                                       encoder = encoder)
    
                    results.append(result)
    
            if test == "feature_selection":
        
                preprocessor = general_preprocessor

                selection_thresholds = [0.01, 0.015, 0.016, 0.0165, 0.017, 0.018, 0.02, 0.025]
                
                for threshold in selection_thresholds:
                    
                    pipeline = make_pipeline(
                        preprocessor,
                        SelectFromModel(estimator = feat_sel_estimator,
                                        threshold = threshold),
                        model)
    
                    result = cv_scores(pipeline,
                                       X, y,
                                       model_name,
                                       model_params,
                                       test = test,
                                       threshold = threshold)
    
                    results.append(result)

        results = create_results_dataframe(results)
        results.to_csv(f"results_data/data_v1_results_{test}.csv", index = False)
        
        return results

    if test == "grid_search":

        preprocessor = general_preprocessor
        model, param_grid = search

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
        
        dump(optimizer.best_estimator_.steps[1][1], "results_data/pickles/fs_forest.pkl")
        
        return np.round(optimizer.best_score_, 4)

    if test == "randomized_search":

        models, grids = search

        results = []
        
        for model, param_grid in zip(models, grids):
        
            pipeline = Pipeline([
                ("preprocessor", preprocessors[0]),
                ("remover", preprocessors[1]),
                ("model", model)
            ])
            
            param_grid = param_grid
        
            optimizer = RandomizedSearchCV(pipeline, 
                                           param_grid,
                                           n_iter = 500,
                                           cv = cv_scheme,
                                           scoring = "roc_auc",
                                           n_jobs = -1)
            optimizer.fit(X, y)

            model_name = model.__class__.__name__
            roc_auc = np.round(optimizer.best_score_, 4)
  
            idx = np.where(optimizer.cv_results_["rank_test_score"] == 1)
            time = np.round((optimizer.cv_results_["mean_fit_time"] + optimizer.cv_results_["mean_score_time"])[idx][0], 2)
        
            result = {"Model": model_name,
                      "ROC_AUC": roc_auc,
                      "Time[s]": time}

            results.append(result)
            
            dump(optimizer.best_estimator_, f"results_data/pickles/{model_name}.pkl")
        
        results = create_results_dataframe(results)
        results.to_csv("results_data/best_estimators.csv", index = False)
        
        return results


def cv_scores(pipeline, X, y, model_name, model_params, test, imputation = None, encoder = None, threshold = None):
    
    """
    Calculates cross validation scores and prepares a dictionary with the results.

    Args:
        preprocessor: Data preprocessor.
        model: Machine learning model to evaluate.
        X (pd.DataFrame): Input data.
        y (np.array): Target labels.
        model_name (str): Name of the machine learning model.
        model_params (str): Model parameters as a formatted string.
        test (str): Type of test, either "imputation", "cat_encoding", "feature_selection".
        imputation (str): Imputation method (only for "imputation" test).
        encoder (str): Encoder used (only for "cat_encoding" test).
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
    Create a results DataFrame from any number of dictionaries or dataframes.
    
    Args:
        *args: Variable number of dictionaries containing results data.
    Returns:
        pd.DataFrame: A DataFrame containing the results.
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

    Args:
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

    results["mean_time[s]"] = results["mean_time[s]"].round(2)
    results = results.sort_values(by = ["max_roc_auc", "mean_roc_auc"], ascending = False)

    return results.style.background_gradient(subset = ["mean_roc_auc", "max_roc_auc", "mean_time[s]", "max_time[s]"])


def process_fold(train_idx, test_idx, X, y, estimator, t):
    
    """
    Processes a single fold of cross-validation. It calculates various classification metrics
    for the provided estimator using the specified threshold and returns F1 Score, precision,
    recall, and the geometric mean of TPF and FPR.
    
    Args:
    train_idx (array-like): The indices of the training data.
    test_idx (array-like): The indices of the testing data.
    X (pd.DataFrame): The input features.
    y (np.array): The target labels.
    estimator: A scikit-learn classifier.
    t (float): The threshold for class labels.

    Returns:
        f1 (float): F1 score.
        precision (float): Precision.
        recall (float): Recall.
        g_mean (float): Geometric mean.
    """
    
    X_train, X_test, y_train, y_test = (X.iloc[train_idx],
                                        X.iloc[test_idx],
                                        y[train_idx],
                                        y[test_idx])
    
    set_config(transform_output="pandas")

    estimator_clone = estimator
    estimator_clone.fit(X_train, y_train)
    y_proba = estimator_clone.predict_proba(X_test)[:, 1]
    y_pred = to_labels(y_proba, t)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    g_mean = np.sqrt(tpr * (1-fpr))

    return f1, precision, recall, g_mean


def cv_scores_for_thresholds(estimators, X, y, thresholds):
    
    """
    Perform cross-validation for multiple estimators and different thresholds.

    Parameters:
        estimators (list of tuples): List of (estimator_name, estimator) pairs.
        X (pd.DataFrame): The input features.
        y (pd.Series): The target labels.
        thresholds (np.array): List of threshold values to evaluate.

    Returns:
        results (dict): A dictionary of results for each estimator.
        optimal_thresholds (list): List of optimal threshold values.
    """
    
    results = {}
    optimal_thresholds = []

    X, y = X.copy(), y.copy()

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    for estimator_name, estimator in estimators:

        f1_scores = []
        precision_scores = []
        recall_scores = []
        g_mean_scores = []

        for t in thresholds:
            f1_cv = []
            precision_cv = []
            recall_cv = []
            g_mean_cv = []

            results_per_fold = Parallel(n_jobs=-1)(
                delayed(process_fold)(train_idx, test_idx, X, y, estimator, t)
                for train_idx, test_idx in cv_scheme.split(X, y)
            )

            for f1, precision, recall, g_mean in results_per_fold:
                f1_cv.append(f1)
                precision_cv.append(precision)
                recall_cv.append(recall)
                g_mean_cv.append(g_mean)

            f1_scores.append(np.mean(f1_cv))
            precision_scores.append(np.mean(precision_cv))
            recall_scores.append(np.mean(recall_cv))
            g_mean_scores.append(np.mean(g_mean_cv))

        optimal_threshold = thresholds[np.argmax(g_mean_scores)]
        optimal_thresholds.append(optimal_threshold)

        results[estimator_name] = (f1_scores, precision_scores, recall_scores)

    thresholds_results_plot(results, thresholds, optimal_thresholds)

    return results, optimal_thresholds