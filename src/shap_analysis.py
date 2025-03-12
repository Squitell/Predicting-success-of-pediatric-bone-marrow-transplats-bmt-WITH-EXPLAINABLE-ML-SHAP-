import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure Matplotlib does not block execution
plt.ioff()


def load_data(relative_path: str) -> pd.DataFrame:
    """ Load a CSV file into a pandas DataFrame. """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading training data from:", full_path)
    return pd.read_csv(full_path)


def load_models(models_dir: str) -> dict:
    """ Load trained models from the specified directory. """
    models = {}
    for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
        model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
            print(f"Loaded {model_name} model from {model_path}")
        else:
            print(f"Warning: {model_name} model not found in {models_dir}")
    return models


def create_shap_directory():
    """ Creates a directory for SHAP analysis results. """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shap_dir = os.path.join(script_dir, "..", "shap")
    os.makedirs(shap_dir, exist_ok=True)
    return shap_dir


def reindex_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """ Align feature names in X with those used during model training. """
    model_features = model.feature_names_in_
    print(f"Using model.feature_names_in_ with {len(model_features)} features.")

    if set(model_features) != set(X.columns):
        print("Reindexing features to match the model...")
        X = X.reindex(columns=model_features, fill_value=0)
    
    print(f"Reindexed X shape: {X.shape}")
    return X


def shap_explain(model, X: pd.DataFrame, model_name: str):
    """ Compute and visualize SHAP values for the given model and dataset. """
    X = reindex_features(X, model)
    
    # Create SHAP directory
    shap_dir = create_shap_directory()
    model_shap_dir = os.path.join(shap_dir, model_name)
    os.makedirs(model_shap_dir, exist_ok=True)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # If binary classification, use the second element of SHAP values list
    if isinstance(shap_values, list):  # XGBoost & LightGBM return lists
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Ensure feature names are aligned
    feature_names = list(X.columns)

    # Compute mean absolute SHAP values and sort them
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    # ✅ **Fix: Extract the most important feature properly**
    top_feature_index = int(np.argmax(mean_abs_shap))  # Ensure it's an integer
    top_feature = feature_names[top_feature_index]  # Get feature name

    print(f"Top feature selected for dependence plot: {top_feature}")

    # 1. SHAP Summary Bar Plot
    plt.figure()
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    bar_plot_path = os.path.join(model_shap_dir, "shap_summary_bar.png")
    plt.savefig(bar_plot_path, bbox_inches="tight")
    print(f"SHAP summary bar plot for {model_name} saved to: {bar_plot_path}")
    plt.close()

    # 2. SHAP Beeswarm Plot
    plt.figure()
    shap.summary_plot(shap_vals, X, show=False)
    beeswarm_plot_path = os.path.join(model_shap_dir, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_plot_path, bbox_inches="tight")
    print(f"SHAP summary beeswarm plot for {model_name} saved to: {beeswarm_plot_path}")
    plt.close()

    # ✅ **Fix: Skip SHAP dependence plot if top feature is categorical**
    if X[top_feature].dtype == "object" or X[top_feature].nunique() < 10:
        print(f"Skipping SHAP dependence plot for categorical feature: {top_feature}")
    else:
        # 3. SHAP Dependence Plot (for most important feature)
        plt.figure()
        shap.dependence_plot(top_feature, shap_vals, X, interaction_index=None, show=False)
        dependence_plot_path = os.path.join(model_shap_dir, f"shap_dependence_{top_feature}.png")
        plt.savefig(dependence_plot_path, bbox_inches="tight")
        print(f"SHAP dependence plot for feature '{top_feature}' of {model_name} saved to: {dependence_plot_path}")
        plt.close()


def main():
    # Load the training dataset
    train_data_path = os.path.join("..", "data", "processed", "bmt_train.csv")
    df_train = load_data(train_data_path)
    print("Training data shape:", df_train.shape)

    # Separate features and target (assumes "survival_status" is the target column)
    if "survival_status" not in df_train.columns:
        raise ValueError("Target column 'survival_status' not found in training data.")
    
    X_train = df_train.drop(columns=["survival_status"])
    y_train = df_train["survival_status"]

    # Load trained models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    models = load_models(models_dir)

    # Perform SHAP analysis for each model
    for model_name, model in models.items():
        print(f"\nPerforming SHAP analysis for {model_name}...")
        shap_explain(model, X_train, model_name)


if __name__ == "__main__":
    main()
