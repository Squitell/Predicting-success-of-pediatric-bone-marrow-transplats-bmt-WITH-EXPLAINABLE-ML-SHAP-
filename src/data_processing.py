"""
data_processing.py

This module performs data preprocessing tasks:
1. Handle missing values.
2. Remove highly correlated features.
3. Optimize memory usage by downcasting numeric types.
4. Save the processed data to a new CSV file.
"""

import os
import numpy as np
import pandas as pd


def handle_missing_values(df: pd.DataFrame,
                          numeric_strategy: str = "median",
                          categorical_strategy: str = "most_frequent") -> pd.DataFrame:
    """
    Handle missing values for both numeric and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    numeric_strategy : str, default "median"
        Strategy for numeric columns: "mean", "median", or "drop".
    categorical_strategy : str, default "most_frequent"
        Strategy for categorical columns: "most_frequent" or "drop".

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Handle numeric columns
    if numeric_strategy == "mean":
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
    elif numeric_strategy == "median":
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
    elif numeric_strategy == "drop":
        df.dropna(subset=numeric_cols, inplace=True)

    # Handle categorical columns
    if categorical_strategy == "most_frequent":
        for col in categorical_cols:
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)
    elif categorical_strategy == "drop":
        df.dropna(subset=categorical_cols, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def remove_highly_correlated_features(df: pd.DataFrame,
                                      threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove features that are highly correlated with each other beyond a given threshold.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame (only numeric columns are considered).
    threshold : float, default 0.9
        Correlatiaon threshold above which features are dropped.

    Returns
    -------
    pd.DataFrame
        DataFrame with highly correlated features removed.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().abs()

    # Only consider the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    df.drop(columns=to_drop, inplace=True, errors='ignore')
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage by downcasting numeric columns to more efficient types.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with optimized memory usage.
    """
    df = df.copy()
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def save_processed_data(df: pd.DataFrame, relative_output_path: str) -> None:
    """
    Save the processed DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The processed DataFrame.
    relative_output_path : str
        Relative path to save the new CSV file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, relative_output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Processed data saved to:", output_path)


def main():
    # Relative path to the original CSV file from the location of this script.
    input_csv = os.path.join("..", "data", "processed", "bmt_dataset.csv")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_input_path = os.path.join(script_dir, input_csv)
    print("Loading raw data from:", full_input_path)
    
    df = pd.read_csv(full_input_path)
    print("Original data shape:", df.shape)
    
    # 1. Handle missing values.
    df = handle_missing_values(df, numeric_strategy="median", categorical_strategy="most_frequent")
    print("After handling missing values, shape:", df.shape)
    
    # 2. Remove highly correlated features.
    df = remove_highly_correlated_features(df, threshold=0.9)
    print("After removing highly correlated features, shape:", df.shape)
    
    # 3. Optimize memory usage.
    df = optimize_memory(df)
    print("Memory optimization complete.")
    
    # 4. Save the processed DataFrame to a new CSV file.
    output_csv = os.path.join("..", "data", "processed", "bmt_dataset_processed.csv")
    save_processed_data(df, output_csv)


if __name__ == "__main__":
    main()
