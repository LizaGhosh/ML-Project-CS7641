"""
Data preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def identify_column_types(df, numeric_threshold=0.7, categorical_max_unique=20):
    """
    Automatically identify column types
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    numeric_threshold : float, default=0.7
        Threshold to consider a column as numeric
    categorical_max_unique : int, default=20
        Maximum number of unique values for categorical columns
        
    Returns:
    --------
    dict
        Dictionary with column types
    """
    column_types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'binary': [],
        'id': []
    }
    
    for col in df.columns:
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types['datetime'].append(col)
            continue
        
        # Try to convert to datetime
        try:
            pd.to_datetime(df[col], errors='raise')
            column_types['datetime'].append(col)
            continue
        except:
            pass
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's binary
            if len(df[col].unique()) <= 2:
                column_types['binary'].append(col)
            # Check if it might be an ID column
            elif col.lower().endswith('id') or col.lower() == 'id':
                column_types['id'].append(col)
            else:
                column_types['numeric'].append(col)
            continue
        
        # For string/object columns
        if pd.api.types.is_object_dtype(df[col]):
            # Check if it can be converted to numeric
            numeric_count = pd.to_numeric(df[col], errors='coerce').notna().mean()
            if numeric_count >= numeric_threshold:
                column_types['numeric'].append(col)
                continue
            
            # Count unique values
            n_unique = df[col].nunique()
            
            # Check if it's binary
            if n_unique <= 2:
                column_types['binary'].append(col)
            # Check if it's categorical
            elif n_unique <= categorical_max_unique:
                column_types['categorical'].append(col)
            # Check if it might be an ID column
            elif col.lower().endswith('id') or col.lower() == 'id':
                column_types['id'].append(col)
            # Otherwise consider it text
            else:
                column_types['text'].append(col)
    
    return column_types

def handle_missing_values(df, strategy='auto', numeric_strategy='mean', 
                         categorical_strategy='most_frequent', threshold=0.5):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    strategy : str, default='auto'
        Strategy for handling missing values
        'auto': Automatically choose based on column type
        'impute': Impute missing values
        'drop_columns': Drop columns with missing values above threshold
        'drop_rows': Drop rows with missing values
    numeric_strategy : str, default='mean'
        Strategy for numerical columns ('mean', 'median', 'constant')
    categorical_strategy : str, default='most_frequent'
        Strategy for categorical columns ('most_frequent', 'constant')
    threshold : float, default=0.5
        Threshold for dropping columns
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with missing values handled
    """
    # Check if there are any missing values
    if not df.isnull().values.any():
        print("No missing values found.")
        return df
    
    df_result = df.copy()
    
    # Drop columns with missing values above threshold
    if strategy in ['auto', 'drop_columns']:
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns with missing values above {threshold*100}% threshold")
            df_result = df_result.drop(columns=cols_to_drop)
    
    # Drop rows with missing values
    if strategy == 'drop_rows':
        original_shape = df_result.shape
        df_result = df_result.dropna()
        print(f"Dropped {original_shape[0] - df_result.shape[0]} rows with missing values")
        return df_result
    
    # Impute missing values
    if strategy in ['auto', 'impute']:
        # Identify column types
        column_types = identify_column_types(df_result)
        
        # Impute numeric columns
        for col in column_types['numeric']:
            if df_result[col].isnull().sum() > 0:
                if numeric_strategy == 'mean':
                    df_result[col] = df_result[col].fillna(df_result[col].mean())
                elif numeric_strategy == 'median':
                    df_result[col] = df_result[col].fillna(df_result[col].median())
                elif numeric_strategy == 'constant':
                    df_result[col] = df_result[col].fillna(0)
        
        # Impute categorical and binary columns
        for col in column_types['categorical'] + column_types['binary']:
            if df_result[col].isnull().sum() > 0:
                if categorical_strategy == 'most_frequent':
                    df_result[col] = df_result[col].fillna(df_result[col].mode()[0])
                elif categorical_strategy == 'constant':
                    df_result[col] = df_result[col].fillna('Unknown')
    
    return df_result

def encode_categorical_features(df, method='auto', drop_first=True):
    """
    Encode categorical features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    method : str, default='auto'
        Encoding method
        'auto': One-hot for low cardinality, label for high cardinality
        'onehot': One-hot encoding for all categorical columns
        'label': Label encoding for all categorical columns
    drop_first : bool, default=True
        Whether to drop the first category in one-hot encoding
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with encoded categorical features
    """
    # Identify categorical columns
    column_types = identify_column_types(df)
    categorical_cols = column_types['categorical'] + column_types['binary']
    
    if not categorical_cols:
        print("No categorical columns found.")
        return df
    
    df_result = df.copy()
    
    # Apply encoding based on method
    for col in categorical_cols:
        # Skip columns with no variation
        if df_result[col].nunique() <= 1:
            continue
        
        # For binary columns, try to convert directly to int
        if col in column_types['binary'] or df_result[col].nunique() == 2:
            try:
                df_result[col] = df_result[col].astype(int)
                continue
            except:
                pass
        
        # Determine encoding method
        col_method = method
        if method == 'auto':
            col_method = 'onehot' if df_result[col].nunique() < 10 else 'label'
        
        # Apply encoding
        if col_method == 'onehot':
            # Create dummy variables
            dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first)
            df_result = pd.concat([df_result, dummies], axis=1)
            df_result.drop(col, axis=1, inplace=True)
        elif col_method == 'label':
            le = LabelEncoder()
            df_result[col] = le.fit_transform(df_result[col].astype(str))
    
    return df_result

def scale_numerical_features(df, method='standard', columns=None):
    """
    Scale numerical features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    method : str, default='standard'
        Scaling method ('standard', 'minmax', 'robust')
    columns : list, default=None
        Columns to scale (if None, use all numeric columns)
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with scaled features
    tuple
        (dataframe, scaler) if return_scaler is True
    """
    df_result = df.copy()
    
    # Identify numerical columns if not specified
    if columns is None:
        column_types = identify_column_types(df)
        columns = column_types['numeric']
    
    if not columns:
        print("No numerical columns found.")
        return df_result
    
    # Select scaler based on method
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    # Apply scaling
    df_result[columns] = scaler.fit_transform(df_result[columns])
    
    return df_result, scaler

def handle_outliers(df, columns=None, method='clip', threshold=3):
    """
    Handle outliers in numerical columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list, default=None
        Columns to handle (if None, use all numeric columns)
    method : str, default='clip'
        Method for handling outliers ('clip', 'remove', 'winsorize')
    threshold : float, default=3
        Z-score threshold for outlier detection
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with outliers handled
    """
    df_result = df.copy()
    
    # Identify numerical columns if not specified
    if columns is None:
        column_types = identify_column_types(df)
        columns = column_types['numeric']
    
    if not columns:
        print("No numerical columns found.")
        return df_result
    
    # Process each column
    for col in columns:
        # Calculate z-scores
        z_scores = np.abs((df_result[col] - df_result[col].mean()) / df_result[col].std())
        outliers = z_scores > threshold
        
        if outliers.sum() == 0:
            continue
        
        print(f"Found {outliers.sum()} outliers in column '{col}'")
        
        if method == 'clip':
            # Clip values to threshold
            upper_bound = df_result[col].mean() + threshold * df_result[col].std()
            lower_bound = df_result[col].mean() - threshold * df_result[col].std()
            df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            # Remove rows with outliers
            df_result = df_result[~outliers]
        elif method == 'winsorize':
            # Replace outliers with percentile values
            q1 = df_result[col].quantile(0.25)
            q3 = df_result[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_result

def create_preprocessing_pipeline(numeric_features, categorical_features,
                                numeric_strategy='mean', categorical_strategy='most_frequent',
                                scaling='standard', encoding='onehot'):
    """
    Create a preprocessing pipeline for numeric and categorical features
    
    Parameters:
    -----------
    numeric_features : list
        List of numeric feature names
    categorical_features : list
        List of categorical feature names
    numeric_strategy : str, default='mean'
        Strategy for imputing numeric features
    categorical_strategy : str, default='most_frequent'
        Strategy for imputing categorical features
    scaling : str, default='standard'
        Scaling method for numeric features
    encoding : str, default='onehot'
        Encoding method for categorical features
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Preprocessing pipeline
    """
    # Numeric transformer
    if scaling == 'standard':
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numeric_strategy)),
            ('scaler', StandardScaler())
        ])
    elif scaling == 'minmax':
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numeric_strategy)),
            ('scaler', MinMaxScaler())
        ])
    elif scaling == 'robust':
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numeric_strategy)),
            ('scaler', RobustScaler())
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numeric_strategy))
        ])
    
    # Categorical transformer
    if encoding == 'onehot':
        try:
            # Try with the newer parameter name (sparse_output)
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=categorical_strategy)),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        except TypeError:
            # Fall back to old parameter name (sparse) for older scikit-learn versions
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=categorical_strategy)),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
    else:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=categorical_strategy)),
            ('encoder', LabelEncoder())
        ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def detect_and_remove_duplicates(df, subset=None, keep='first'):
    """
    Detect and remove duplicate rows
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    subset : list, default=None
        Columns to consider for identifying duplicates (if None, use all columns)
    keep : str, default='first'
        Which duplicates to keep ('first', 'last', False)
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with duplicates removed
    """
    # Check for duplicates
    duplicates = df.duplicated(subset=subset, keep=False).sum()
    
    if duplicates == 0:
        print("No duplicate rows found.")
        return df
    
    print(f"Found {duplicates} duplicate rows")
    
    # Remove duplicates
    df_result = df.drop_duplicates(subset=subset, keep=keep)
    print(f"Removed {len(df) - len(df_result)} rows")
    
    return df_result

def process_date_columns(df, date_columns=None, drop_original=True):
    """
    Process date columns into useful features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    date_columns : list, default=None
        List of date columns (if None, try to identify date columns)
    drop_original : bool, default=True
        Whether to drop original date columns
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with processed date features
    """
    df_result = df.copy()
    
    # Identify date columns if not specified
    if date_columns is None:
        date_columns = []
        for col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_columns.append(col)
                else:
                    pd.to_datetime(df[col], errors='raise')
                    date_columns.append(col)
            except:
                continue
    
    if not date_columns:
        print("No date columns found.")
        return df_result
    
    # Process each date column
    for col in date_columns:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_result[col]):
            try:
                df_result[col] = pd.to_datetime(df_result[col])
            except:
                print(f"Could not convert column '{col}' to datetime.")
                continue
        
        # Extract date components
        df_result[f'{col}_year'] = df_result[col].dt.year
        df_result[f'{col}_month'] = df_result[col].dt.month
        df_result[f'{col}_day'] = df_result[col].dt.day
        df_result[f'{col}_dayofweek'] = df_result[col].dt.dayofweek
        df_result[f'{col}_quarter'] = df_result[col].dt.quarter
        
        # Create is_weekend feature
        df_result[f'{col}_is_weekend'] = df_result[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Check if time information exists
        has_time = (df_result[col].dt.hour != 0).any() or (df_result[col].dt.minute != 0).any()
        
        if has_time:
            df_result[f'{col}_hour'] = df_result[col].dt.hour
            df_result[f'{col}_minute'] = df_result[col].dt.minute
        
        # Drop original column if requested
        if drop_original:
            df_result.drop(col, axis=1, inplace=True)
    
    return df_result

def prepare_train_test_set(df, target_column, test_size=0.2, random_state=42, stratify=None):
    """
    Prepare train and test sets with preprocessing
    
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
    stratify : array-like, default=None
        Data to use for stratified sampling (typically the target for classification)
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if stratify is not None:
        stratify = df[stratify] if isinstance(stratify, str) else stratify
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Identify column types
    column_types = identify_column_types(X_train)
    numeric_features = column_types['numeric']
    categorical_features = column_types['categorical'] + column_types['binary']
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )
    
    # Fit preprocessor on training data
    preprocessor.fit(X_train)
    
    return X_train, X_test, y_train, y_test, preprocessor

def apply_preprocessing(X_train, X_test, preprocessor):
    """
    Apply preprocessing to train and test data
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessor pipeline
        
    Returns:
    --------
    tuple
        (X_train_processed, X_test_processed)
    """
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = []
    
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == 'num':
            feature_names.extend(columns)
        elif transformer_name == 'cat':
            try:
                # For OneHotEncoder
                encoder = transformer.named_steps['encoder']
                # Get feature names for each categorical column
                for i, col in enumerate(columns):
                    categories = encoder.categories_[i]
                    feature_names.extend([f"{col}_{cat}" for cat in categories])
            except:
                # For other encoders or in case of error
                feature_names.extend(columns)
    
    return X_train_processed, X_test_processed, feature_names

if __name__ == "__main__":
    # Example usage
    import data_loader
    
    data_path = "your_dataset.csv"
    df = data_loader.load_data(data_path)
    
    if df is not None:
        # Handle missing values
        df_clean = handle_missing_values(df, strategy='auto')
        
        # Scale numerical features
        column_types = identify_column_types(df_clean)
        df_scaled, scaler = scale_numerical_features(df_clean, method='standard', columns=column_types['numeric'])
        
        # Encode categorical features
        df_encoded = encode_categorical_features(df_scaled, method='auto')
        
        # Prepare train/test split
        target_column = 'price'  # Replace with your target column
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_set(
            df_encoded, target_column, test_size=0.2
        )
        
        # Apply preprocessing
        X_train_processed, X_test_processed, feature_names = apply_preprocessing(
            X_train, X_test, preprocessor
        )
        
        print("Preprocessing complete!")
        print(f"Processed training data shape: {X_train_processed.shape}")
        print(f"Processed testing data shape: {X_test_processed.shape}")