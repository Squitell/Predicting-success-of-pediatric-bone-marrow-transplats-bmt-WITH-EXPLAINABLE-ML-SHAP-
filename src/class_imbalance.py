import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Import SMOTE from imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("Please install imbalanced-learn (pip install imbalanced-learn) to use SMOTE.")

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def load_data(relative_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame using a path relative to this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)
    print("Loading data from:", full_path)
    return pd.read_csv(full_path)

def create_plots_dir():
    """
    Create a directory for saving imbalance-related plots.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "..", "imbalance_plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_class_distribution(y: pd.Series, title: str, filename: str, folder: str):
    """
    Plot and save the class distribution as a bar chart.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Plot saved to: {filepath}")
    plt.close()

def balance_with_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Oversample the minority class using SMOTE.
    """
    print("Applying SMOTE for oversampling...")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def compute_class_weights(y: pd.Series) -> dict:
    """
    Compute balanced class weights for the target variable.
    """
    classes = np.unique(y)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, weights))
    return weight_dict

def split_and_save_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split the data into training and testing sets and save them to CSV files.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    train_df = X_train.copy()
    train_df["survival_status"] = y_train
    test_df = X_test.copy()
    test_df["survival_status"] = y_test

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "..", "data", "processed", "bmt_train.csv")
    test_path = os.path.join(script_dir, "..", "data", "processed", "bmt_test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("Training data saved to:", train_path)
    print("Testing data saved to:", test_path)

def main():
    # Load processed data.
    input_csv = os.path.join("..", "data", "processed", "bmt_dataset_processed.csv")
    df = load_data(input_csv)
    print("Original processed data shape:", df.shape)
    
    # Check that target column exists.
    if "survival_status" not in df.columns:
        raise ValueError("Target column 'survival_status' not found in data.")
    
    # Separate features and target.
    X = df.drop(columns=["survival_status"])
    y = df["survival_status"]
    
    # Apply one-hot encoding to ensure all features are numeric.
    X = pd.get_dummies(X, drop_first=True)
    print("After one-hot encoding, feature shape:", X.shape)
    
    # Create directory for imbalance-related plots.
    plots_dir = create_plots_dir()
    
    # Plot and save the original class distribution.
    print("Original class distribution:", Counter(y))
    plot_class_distribution(y, title="Original Class Distribution", 
                            filename="original_class_distribution.png", folder=plots_dir)
    
    # Use SMOTE for oversampling (chosen method).
    X_smote, y_smote = balance_with_smote(X, y, random_state=42)
    print("After SMOTE, class distribution:", Counter(y_smote))
    plot_class_distribution(y_smote, title="SMOTE Oversampled Distribution", 
                            filename="smote_distribution.png", folder=plots_dir)
    
    # Compute class weights (informational).
    weights = compute_class_weights(y)
    print("Computed class weights:", weights)
    
    # Use the SMOTE-balanced data for splitting.
    X_balanced, y_balanced = X_smote, y_smote
    
    # Split the balanced data into training and testing sets and save them.
    split_and_save_train_test(X_balanced, y_balanced, test_size=0.2, random_state=42)

if __name__ == "__main__":
    main()

#Why SMOTE?
#For a moderately imbalanced dataset like ours (60% vs. 40%), SMOTE is a robust method because it synthesizes new minority samples instead of merely duplicating or dropping data. This helps preserve the overall information in the dataset and is generally preferable over undersampling when the dataset is not extremely imbalanced.
