"""
Exploratory Data Analysis utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_distribution(df, column, figsize=(10, 6), bins=30, kde=True, log_transform=False):
    """
    Plot distribution of a numerical column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name to plot
    figsize : tuple, default=(10, 6)
        Figure size
    bins : int, default=30
        Number of histogram bins
    kde : bool, default=True
        Whether to plot kernel density estimate
    log_transform : bool, default=False
        Apply log transformation for skewed data
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    
    data = df[column]
    title = f'Distribution of {column}'
    
    if log_transform and (data > 0).all():
        data = np.log(data)
        title = f'Distribution of log({column})'
    
    sns.histplot(data, bins=bins, kde=kde)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics for {column}:")
    stats_df = pd.DataFrame({
        'Mean': [data.mean()],
        'Median': [data.median()],
        'Std Dev': [data.std()],
        'Min': [data.min()],
        'Max': [data.max()],
        'Skewness': [stats.skew(data.dropna())],
        'Kurtosis': [stats.kurtosis(data.dropna())]
    })
    print(stats_df)

def plot_categorical_distribution(df, column, figsize=(12, 6), top_n=None, sort_by_count=True):
    """
    Plot distribution of a categorical column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name to plot
    figsize : tuple, default=(12, 6)
        Figure size
    top_n : int, default=None
        Limit to top N categories
    sort_by_count : bool, default=True
        Sort categories by count
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    
    value_counts = df[column].value_counts()
    
    if sort_by_count:
        value_counts = value_counts.sort_values(ascending=False)
    
    if top_n is not None:
        value_counts = value_counts.head(top_n)
        title = f'Top {top_n} Categories in {column}'
    else:
        title = f'Distribution of {column}'
    
    value_counts.plot(kind='bar')
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print category percentages
    print(f"\nCategory Percentages for {column}:")
    percentage_df = pd.DataFrame({
        'Count': value_counts,
        'Percentage': (value_counts / len(df) * 100).round(2)
    })
    print(percentage_df)

def plot_target_by_category(df, category_col, target_col, figsize=(12, 6), 
                           plot_type='box', top_n=None, sort_by_target=True):
    """
    Plot target variable by categorical column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    category_col : str
        Categorical column name
    target_col : str
        Target column name
    figsize : tuple, default=(12, 6)
        Figure size
    plot_type : str, default='box'
        Type of plot ('box', 'violin', or 'bar')
    top_n : int, default=None
        Limit to top N categories
    sort_by_target : bool, default=True
        Sort categories by target mean
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    
    # Calculate means for sorting if needed
    if sort_by_target:
        category_means = df.groupby(category_col)[target_col].mean().sort_values(ascending=False)
        categories = category_means.index.tolist()
        if top_n:
            categories = categories[:top_n]
    else:
        categories = df[category_col].value_counts().head(top_n).index.tolist() if top_n else None
    
    # Create plot
    if plot_type == 'box':
        sns.boxplot(x=category_col, y=target_col, data=df, order=categories)
    elif plot_type == 'violin':
        sns.violinplot(x=category_col, y=target_col, data=df, order=categories)
    elif plot_type == 'bar':
        sns.barplot(x=category_col, y=target_col, data=df, order=categories, estimator=np.mean)
    
    plt.title(f'{target_col} by {category_col}')
    plt.xlabel(category_col)
    plt.ylabel(target_col)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{target_col} Statistics by {category_col}:")
    summary_df = df.groupby(category_col)[target_col].agg(['count', 'mean', 'std', 'min', 'max'])
    if sort_by_target:
        summary_df = summary_df.sort_values('mean', ascending=False)
    if top_n:
        summary_df = summary_df.head(top_n)
    print(summary_df)

def plot_correlation_matrix(df, figsize=(12, 10), method='pearson', annot=True, 
                           target_col=None, numeric_only=True):
    """
    Plot correlation matrix
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    figsize : tuple, default=(12, 10)
        Figure size
    method : str, default='pearson'
        Correlation method ('pearson', 'kendall', 'spearman')
    annot : bool, default=True
        Whether to annotate the heatmap with correlation values
    target_col : str, default=None
        Highlight correlations with this target column
    numeric_only : bool, default=True
        Only include numeric columns
        
    Returns:
    --------
    pandas.DataFrame
        Correlation matrix
    """
    # Select numeric columns if requested
    if numeric_only:
        data = df.select_dtypes(include=['int64', 'float64'])
    else:
        data = df
    
    # Calculate correlation matrix
    corr_matrix = data.corr(method=method)
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f', 
               cmap='coolwarm', center=0, square=True, linewidths=.5)
    
    plt.title(f'{method.capitalize()} Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Show correlations with target if specified
    if target_col and target_col in corr_matrix.columns:
        print(f"\nCorrelations with {target_col}:")
        target_corr = corr_matrix[target_col].sort_values(ascending=False)
        print(target_corr)
    
    return corr_matrix

def plot_scatter_matrix(df, columns=None, figsize=(15, 15), target_col=None):
    """
    Plot scatter matrix (pairs plot)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list, default=None
        List of columns to include (uses all numeric if None)
    figsize : tuple, default=(15, 15)
        Figure size
    target_col : str, default=None
        Color points by this categorical column
        
    Returns:
    --------
    None
    """
    # Select columns
    if columns is None:
        plot_df = df.select_dtypes(include=['int64', 'float64'])
    else:
        plot_df = df[columns]
    
    # Create plot
    if target_col and target_col in df.columns:
        g = sns.pairplot(df, vars=plot_df.columns, hue=target_col, corner=True)
    else:
        g = sns.pairplot(plot_df, corner=True)
    
    g.fig.set_size_inches(figsize)
    plt.tight_layout()
    plt.show()

def identify_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Identify outliers in numerical columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list, default=None
        List of columns to check (uses all numeric if None)
    method : str, default='iqr'
        Method to identify outliers ('iqr' or 'zscore')
    threshold : float, default=1.5
        Threshold for IQR method (typically 1.5) or Z-score method (typically 3)
        
    Returns:
    --------
    dict
        Dictionary with outlier information for each column
    """
    # Select columns
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    outlier_info = {}
    
    for col in columns:
        if method == 'iqr':
            # IQR method
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df.iloc[np.where(z_scores > threshold)]
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'min': df[col].min(),
            'max': df[col].max(),
            'outlier_indices': outliers.index.tolist()
        }
        
        # Add bounds for IQR method
        if method == 'iqr':
            outlier_info[col]['lower_bound'] = lower_bound
            outlier_info[col]['upper_bound'] = upper_bound
    
    return outlier_info

def plot_time_series(df, time_col, value_col, figsize=(15, 6), freq=None, title=None):
    """
    Plot time series data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    time_col : str
        Column with time/date information
    value_col : str
        Column with values to plot
    figsize : tuple, default=(15, 6)
        Figure size
    freq : str, default=None
        Resampling frequency (e.g., 'M' for month, 'Q' for quarter)
    title : str, default=None
        Plot title
        
    Returns:
    --------
    None
    """
    # Ensure the time column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            time_series_df = df.copy()
            time_series_df[time_col] = pd.to_datetime(time_series_df[time_col])
        except:
            print(f"Could not convert {time_col} to datetime. Please check the format.")
            return
    else:
        time_series_df = df
    
    # Set the time column as index
    time_series_df = time_series_df.set_index(time_col)
    
    # Resample if frequency is specified
    if freq:
        time_series_df = time_series_df[value_col].resample(freq).mean()
    else:
        time_series_df = time_series_df[value_col]
    
    # Plot
    plt.figure(figsize=figsize)
    time_series_df.plot()
    
    plt.title(title or f'Time Series of {value_col}')
    plt.xlabel('Time')
    plt.ylabel(value_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_missing_values(df, figsize=(10, 6)):
    """
    Visualize missing values in the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    None
    """
    # Calculate percentage of missing values
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        print("No missing values found.")
        return
    
    plt.figure(figsize=figsize)
    missing.plot(kind='bar')
    plt.title('Percentage of Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Values (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print missing values information
    print("\nMissing Values Information:")
    missing_df = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('Missing Count', ascending=False)
    print(missing_df[missing_df['Missing Count'] > 0])

if __name__ == "__main__":
    # Example usage
    import data_loader
    
    data_path = "your_dataset.csv"
    df = data_loader.load_data(data_path)
    
    if df is not None:
        # Perform EDA
        target_column = 'price'  # Replace with your target column
        plot_distribution(df, target_column)
        plot_distribution(df, target_column, log_transform=True)
        
        # Categorical analysis
        category_col = 'category'  # Replace with your categorical column
        plot_categorical_distribution(df, category_col, top_n=10)
        plot_target_by_category(df, category_col, target_column, plot_type='box')
        
        # Correlation analysis
        plot_correlation_matrix(df, target_col=target_column)