import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame using a path relative to this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading test data from:", full_path)
    df = pd.read_csv(full_path)
    print("Loaded test data with", df.shape[0], "rows and", df.shape[1], "columns.")
    return df

def reindex_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Reindex X so that it has exactly the features used during training.
    If the model has the attribute 'feature_names_in_', use it; otherwise, X is returned as is.
    """
    if hasattr(model, "feature_names_in_"):
        training_features = list(model.feature_names_in_)
        print(f"Reindexing test data to {len(training_features)} features...")
        X = X.reindex(columns=training_features, fill_value=0)
    else:
        print("Model does not have 'feature_names_in_'; using X as is.")
    return X

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a model on the test set and return performance metrics.
    """
    X_test = reindex_features(X_test, model)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0)
    }
    return metrics

def load_models(models_dir: str) -> dict:
    """
    Load all trained models from the specified directory.
    Expects model files: RandomForest_model.pkl, XGBoost_model.pkl, LightGBM_model.pkl.
    """
    loaded_models = {}
    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        model_file = os.path.join(models_dir, f"{model_name}_model.pkl")
        if os.path.exists(model_file):
            loaded_models[model_name] = joblib.load(model_file)
            print(f"Loaded {model_name} model from {model_file}")
        else:
            print(f"Warning: {model_file} not found!")
    if not loaded_models:
        print("No models were loaded. Please verify your models folder.")
    return loaded_models

def main():
    # Load test data.
    test_data_path = os.path.join("..", "data", "processed", "bmt_test.csv")
    df_test = load_data(test_data_path)
    print("Test data shape:", df_test.shape)
    
    # Check that target column exists.
    if "survival_status" not in df_test.columns:
        raise ValueError("Target column 'survival_status' not found in test data.")
    
    X_test = df_test.drop(columns=["survival_status"])
    y_test = df_test["survival_status"]
    
    # Apply one-hot encoding.
    X_test = pd.get_dummies(X_test, drop_first=True)
    print("Feature shape after one-hot encoding:", X_test.shape)
    
    # Load models.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    models = load_models(models_dir)
    
    # If no models were loaded, exit.
    if not models:
        return
    
    # Evaluate each model.
    results = {}
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} on test data...")
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics
        print(f"{model_name} metrics: Accuracy={metrics['Accuracy']:.4f}, ROC-AUC={metrics['ROC-AUC']:.4f}, "
              f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}")
    
    results_df = pd.DataFrame(results).T
    print("\nSummary of test performance:")
    print(results_df)
    
    # Save results to CSV in the "plots" folder.
    plots_dir = os.path.join(script_dir, "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    results_csv_path = os.path.join(plots_dir, "model_test_performance.csv")
    results_df.to_csv(results_csv_path)
    print(f"Test performance results saved to: {results_csv_path}")

if __name__ == "__main__":
    main()
