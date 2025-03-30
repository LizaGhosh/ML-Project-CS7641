"""
Data loading and basic inspection utility functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load data from a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def inspect_data(df):
    """
    Perform basic data inspection
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    None
    """
    # Display basic information
    print("Data Overview:")
    print(f"Shape: {df.shape}")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe().T)

def check_column_consistency(df):
    """
    Check for column consistency
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary with consistency check results
    """
    results = {}
    
    # Check for unique column names
    results['unique_columns'] = len(df.columns) == len(set(df.columns))
    
    # Check for completeness in rows
    results['complete_rows'] = df.notnull().all(axis=1).all()
    
    # Check for outliers in numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    results['potential_outliers'] = {}
    
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        results['potential_outliers'][col] = outliers
    
    return results

def clean_column_names(df):
    """
    Standardize column names
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with cleaned column names
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove unnamed or index columns
    unnamed_cols = [col for col in df_clean.columns if 'Unnamed' in col or col == '']
    if unnamed_cols:
        df_clean = df_clean.drop(columns=unnamed_cols)
    
    # Convert column names to lowercase, replace spaces with underscores
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    
    return df_clean

def split_train_test(df, target_column, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    data_path = "your_dataset.csv"
    df = load_data(data_path)
    
    if df is not None:
        inspect_data(df)
        consistency_results = check_column_consistency(df)
        print("\nConsistency Check Results:")
        for key, value in consistency_results.items():
            print(f"{key}: {value}")