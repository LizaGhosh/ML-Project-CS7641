"""
Feature Engineering utilities
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def extract_numerical_from_string(text, pattern, default=0):
    """
    Extract numerical values from text using regex pattern
    
    Parameters:
    -----------
    text : str
        Input text
    pattern : str
        Regex pattern with a capturing group for the number
    default : numeric, default=0
        Default value if no match is found
        
    Returns:
    --------
    numeric
        Extracted numerical value
    """
    if pd.isna(text) or not isinstance(text, str):
        return default
    
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return default

def extract_categorical_from_string(text, patterns, default='Other'):
    """
    Extract categorical value from text using multiple regex patterns
    
    Parameters:
    -----------
    text : str
        Input text
    patterns : dict
        Dictionary mapping categories to regex patterns
    default : str, default='Other'
        Default value if no match is found
        
    Returns:
    --------
    str
        Extracted category
    """
    if pd.isna(text) or not isinstance(text, str):
        return default
    
    for category, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return category
    
    return default

def extract_binary_feature(text, pattern):
    """
    Extract binary feature from text using regex pattern
    
    Parameters:
    -----------
    text : str
        Input text
    pattern : str
        Regex pattern to search for
        
    Returns:
    --------
    int
        1 if pattern is found, 0 otherwise
    """
    if pd.isna(text) or not isinstance(text, str):
        return 0
    
    return 1 if re.search(pattern, text, re.IGNORECASE) else 0

def create_interaction_features(df, column_pairs=None):
    """
    Create interaction features between pairs of columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    column_pairs : list of tuples, optional
        List of column pairs to create interactions for
        If None, will create interactions for all numeric columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with interaction features
    """
    df_result = df.copy()
    
    # If no column pairs specified, create pairs from numeric columns
    if column_pairs is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        column_pairs = [(numeric_cols[i], numeric_cols[j]) 
                        for i in range(len(numeric_cols)) 
                        for j in range(i+1, len(numeric_cols))]
    
    # Create interactions for each pair
    for col1, col2 in column_pairs:
        # Skip if columns don't exist
        if col1 not in df.columns or col2 not in df.columns:
            continue
            
        # Ensure columns are numeric before performing operations
        series1 = pd.to_numeric(df[col1], errors='coerce')
        series2 = pd.to_numeric(df[col2], errors='coerce')
        
        # Skip if conversion resulted in all NaN
        if series1.isna().all() or series2.isna().all():
            continue
            
        # Create multiplication feature
        df_result[f'{col1}_x_{col2}'] = series1 * series2
        
        # Division features (handle division by zero)
        # col1 / col2
        with np.errstate(divide='ignore', invalid='ignore'):
            division = series1 / series2
            df_result[f'{col1}_div_{col2}'] = division
            df_result[f'{col1}_div_{col2}'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # col2 / col1
        with np.errstate(divide='ignore', invalid='ignore'):
            division = series2 / series1
            df_result[f'{col2}_div_{col1}'] = division
            df_result[f'{col2}_div_{col1}'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Addition and subtraction
        df_result[f'{col1}_plus_{col2}'] = series1 + series2
        df_result[f'{col1}_minus_{col2}'] = series1 - series2
        df_result[f'{col2}_minus_{col1}'] = series2 - series1
    
    return df_result

def create_polynomial_features(df, columns, degree=2):
    """
    Create polynomial features for specified columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        List of columns to create polynomial features for
    degree : int, default=2
        Degree of polynomial features
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with polynomial features added
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    df_result = df.copy()
    
    for col in columns:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[[col]])
        
        # Create feature names
        feature_names = [f'{col}^{i+1}' for i in range(degree)]
        
        # Add polynomial features to dataframe
        for i, name in enumerate(feature_names):
            if i > 0:  # Skip the first feature (original feature)
                df_result[name] = poly_features[:, i]
    
    return df_result

def create_binned_features(df, columns, n_bins=5, strategy='quantile', labels=None):
    """
    Create binned features for specified columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        List of columns to create binned features for
    n_bins : int, default=5
        Number of bins
    strategy : str, default='quantile'
        Binning strategy ('uniform', 'quantile', or 'kmeans')
    labels : list, default=None
        Labels for the bins. If None, use the bin number.
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with binned features added
    """
    from sklearn.preprocessing import KBinsDiscretizer
    
    df_result = df.copy()
    
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    
    for col in columns:
        # Reshape for sklearn
        binned = discretizer.fit_transform(df[[col]])
        
        # Create bin edges for uniform strategy
        if strategy == 'uniform':
            bin_edges = np.linspace(df[col].min(), df[col].max(), n_bins + 1)
        elif strategy == 'quantile':
            bin_edges = [df[col].quantile(q) for q in np.linspace(0, 1, n_bins + 1)]
        else:
            bin_edges = None
        
        # Create custom labels if not provided
        if labels is None and bin_edges is not None:
            custom_labels = [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(n_bins)]
        else:
            custom_labels = labels
        
        # Add binned feature
        df_result[f'{col}_binned'] = binned
        
        # Map to custom labels if available
        if custom_labels:
            df_result[f'{col}_binned'] = df_result[f'{col}_binned'].map(
                {i: label for i, label in enumerate(custom_labels)}
            )
    
    return df_result

def create_ratio_features(df, numerator_cols, denominator_cols, suffix='_ratio'):
    """
    Create ratio features between columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    numerator_cols : list
        List of columns to use as numerators
    denominator_cols : list
        List of columns to use as denominators
    suffix : str, default='_ratio'
        Suffix to add to ratio feature names
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with ratio features added
    """
    df_result = df.copy()
    
    for num_col in numerator_cols:
        for denom_col in denominator_cols:
            # Skip if numerator and denominator are the same
            if num_col == denom_col:
                continue
            
            # Create ratio feature with safeguard against division by zero
            feature_name = f'{num_col}_to_{denom_col}{suffix}'
            df_result[feature_name] = df[num_col] / df[denom_col].replace(0, np.nan)
    
    return df_result

def create_group_statistics(df, group_cols, value_cols, operations=['mean', 'std', 'min', 'max']):
    """
    Create statistical aggregations within groups
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    group_cols : list
        Columns to group by
    value_cols : list
        Columns to aggregate
    operations : list, default=['mean', 'std', 'min', 'max']
        Statistical operations to apply
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with group statistics added
    """
    df_result = df.copy()
    
    # Convert group_cols to list if it's a single string
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    for group_col in group_cols:
        for value_col in value_cols:
            # Calculate group statistics
            group_stats = df.groupby(group_col)[value_col].agg(operations)
            
            # Create feature names
            feature_names = {op: f'{value_col}_by_{group_col}_{op}' for op in operations}
            group_stats = group_stats.rename(columns=feature_names)
            
            # Merge with original dataframe
            df_result = df_result.merge(
                group_stats.reset_index(), 
                on=group_col, 
                how='left'
            )
    
    return df_result

def create_time_features(df, datetime_col):
    """
    Create time-based features from a datetime column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    datetime_col : str
        Name of the datetime column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with time features added
    """
    df_result = df.copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df_result[datetime_col]):
        try:
            df_result[datetime_col] = pd.to_datetime(df_result[datetime_col])
        except:
            print(f"Could not convert {datetime_col} to datetime. Please check the format.")
            return df_result
    
    # Extract basic time components
    df_result[f'{datetime_col}_year'] = df_result[datetime_col].dt.year
    df_result[f'{datetime_col}_month'] = df_result[datetime_col].dt.month
    df_result[f'{datetime_col}_day'] = df_result[datetime_col].dt.day
    df_result[f'{datetime_col}_dayofweek'] = df_result[datetime_col].dt.dayofweek
    df_result[f'{datetime_col}_quarter'] = df_result[datetime_col].dt.quarter
    
    # Extract hour, minute, second if time component exists
    if (df_result[datetime_col].dt.hour > 0).any():
        df_result[f'{datetime_col}_hour'] = df_result[datetime_col].dt.hour
        df_result[f'{datetime_col}_minute'] = df_result[datetime_col].dt.minute
        df_result[f'{datetime_col}_second'] = df_result[datetime_col].dt.second
    
    # Create cyclical features for month, day of week, hour
    df_result[f'{datetime_col}_month_sin'] = np.sin(2 * np.pi * df_result[f'{datetime_col}_month'] / 12)
    df_result[f'{datetime_col}_month_cos'] = np.cos(2 * np.pi * df_result[f'{datetime_col}_month'] / 12)
    
    df_result[f'{datetime_col}_dayofweek_sin'] = np.sin(2 * np.pi * df_result[f'{datetime_col}_dayofweek'] / 7)
    df_result[f'{datetime_col}_dayofweek_cos'] = np.cos(2 * np.pi * df_result[f'{datetime_col}_dayofweek'] / 7)
    
    if (df_result[datetime_col].dt.hour > 0).any():
        df_result[f'{datetime_col}_hour_sin'] = np.sin(2 * np.pi * df_result[f'{datetime_col}_hour'] / 24)
        df_result[f'{datetime_col}_hour_cos'] = np.cos(2 * np.pi * df_result[f'{datetime_col}_hour'] / 24)
    
    # Create is_weekend feature
    df_result[f'{datetime_col}_is_weekend'] = df_result[f'{datetime_col}_dayofweek'].isin([5, 6]).astype(int)
    
    return df_result

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create a sample dataframe
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'date_column': ['2022-01-01', '2022-02-15', '2022-03-30', '2022-04-10', '2022-05-25'],
        'text_column': ['8GB RAM', '16GB RAM', '4GB RAM', '32GB RAM', '2GB RAM']
    }
    df = pd.DataFrame(data)
    
    # Convert date column to datetime
    df['date_column'] = pd.to_datetime(df['date_column'])
    
    print("Original DataFrame:")
    print(df)
    
    # Extract numerical values from text
    print("\nExtracting RAM from text:")
    df['ram_gb'] = df['text_column'].apply(
        lambda x: extract_numerical_from_string(x, r'(\d+)GB', default=0)
    )
    print(df)
    
    # Create polynomial features
    print("\nCreating polynomial features:")
    df_poly = create_polynomial_features(df, ['feature1', 'feature2'], degree=2)
    print(df_poly)
    
    # Create interaction features
    print("\nCreating interaction features:")
    df_interact = create_interaction_features(df, [('feature1', 'feature2')])
    print(df_interact)
    
    # Create time features
    print("\nCreating time features:")
    df_time = create_time_features(df, 'date_column')
    print(df_time)
    
    # Demonstrate multiple feature engineering steps
    print("\nCombined feature engineering:")
    df_engineered = df.copy()
    # Extract RAM
    df_engineered['ram_gb'] = df_engineered['text_column'].apply(
        lambda x: extract_numerical_from_string(x, r'(\d+)GB', default=0)
    )
    # Create time features
    df_engineered = create_time_features(df_engineered, 'date_column')
    # Create polynomial features for RAM
    df_engineered = create_polynomial_features(df_engineered, ['ram_gb'], degree=2)
    # Create interaction between RAM and month
    df_engineered = create_interaction_features(
        df_engineered, [('ram_gb', 'date_column_month')]
    )
    print(df_engineered[['ram_gb', 'ram_gb^2', 'ram_gb_x_date_column_month']].head())