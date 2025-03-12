import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame using a path relative to this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading training data from:", full_path)
    return pd.read_csv(full_path)

def train_models(X, y):
    """
    Train three models on the training data and print training performance.
    Returns a dictionary mapping model names to trained model objects.
    """
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name} on {X.shape[1]} features...")
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        if hasattr(model, "predict_proba"):
            roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        else:
            roc_auc = 0.0
        print(f"{name} Training Performance: Accuracy={acc:.4f}, ROC-AUC={roc_auc:.4f}")
        trained_models[name] = model
    return trained_models

def save_models(models: dict, output_dir: str):
    """
    Save each trained model as a pickle file in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        filename = os.path.join(output_dir, f"{name}_model.pkl")
        joblib.dump(model, filename)
        print(f"Saved {name} model to {filename}")

def main():
    # Load oversampled and processed training data.
    train_csv = os.path.join("..", "data", "processed", "bmt_train.csv")
    df = load_data(train_csv)
    print("Training data shape:", df.shape)
    
    # Ensure target column exists.
    if "survival_status" not in df.columns:
        raise ValueError("Target column 'survival_status' not found in training data.")
    
    # Separate features and target.
    X = df.drop(columns=["survival_status"])
    y = df["survival_status"]
    
    # If needed, uncomment the next line if your data is not already one-hot encoded:
    # X = pd.get_dummies(X, drop_first=True)
    
    print("Training data has", X.shape[1], "features.")
    
    # Train models on the oversampled data.
    trained_models = train_models(X, y)
    
    # Save the trained models.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    save_models(trained_models, models_dir)

if __name__ == "__main__":
    main()
