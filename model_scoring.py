from joblib import load
import os

def load_model(model_name):
    if model_name == "RandomForestClassifier":
        with open("RandomForestClassifier.joblib", 'rb') as f:
            model = load(f)
    elif model_name == "DecisionTreeClassifier":
        with open("DecisionTreeClassifier.joblib", 'rb') as f:
            model = load(f)
    elif model_name == "MultiRandomForestClassifier":
        with open("MultiRandomForestClassifier.joblib", 'rb') as f:
            model = load(f)
    
    return model

def run_scoring(model, input_record):
    res = model.predict_proba(input_record)
    return res
