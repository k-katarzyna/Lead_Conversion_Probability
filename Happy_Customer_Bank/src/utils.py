from joblib import load

def load_estimators():
    
    HGBclf = load("results_data/pickles/HistGradientBoostingClassifier.pkl")
    BRFclf = load("results_data/pickles/BalancedRandomForestClassifier.pkl")
    BBclf = load("results_data/pickles/BalancedBaggingClassifier.pkl")
    RFclf = load("results_data/pickles/RandomForestClassifier.pkl")

    return HGBclf, BRFclf, BBclf, RFclf


def to_labels(y_proba, threshold):
    
    return (y_proba >= threshold).astype("int")

