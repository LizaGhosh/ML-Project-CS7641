"""
Model evaluation utilities for regression tasks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    explained_variance_score,
    mean_absolute_percentage_error,
    max_error
)
from sklearn.model_selection import cross_val_score, KFold, learning_curve
import time
from scipy import stats

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate standard regression evaluation metrics
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary with calculated metrics
    """
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred)
    }
    
    # Calculate MAPE only if there are no zeros in y_true
    if not np.any(y_true == 0):
        metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    else:
        # Calculate MAPE excluding zeros
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            metrics['MAPE (non-zero)'] = mean_absolute_percentage_error(
                y_true[non_zero_mask], y_pred[non_zero_mask]) * 100
    
    # Calculate adjusted R²
    n = len(y_true)
    p = 1  # Number of predictors, use 1 as default
    metrics['Adjusted R²'] = 1 - (1 - metrics['R²']) * (n - 1) / (n - p - 1)
    
    return metrics

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate a model on both training and test data
    
    Parameters:
    -----------
    model : object
        Trained model with predict method
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training target values
    y_test : array-like
        Test target values
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics for both sets
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_regression_metrics(y_train, y_train_pred)
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
    
    # Add prefixes to distinguish between train and test metrics
    train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    
    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}
    
    return all_metrics, y_train_pred, y_test_pred

def print_metrics(metrics, model_name="Model"):
    """
    Print evaluation metrics in a readable format
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with evaluation metrics
    model_name : str, default="Model"
        Name of the model for display
        
    Returns:
    --------
    None
    """
    print(f"\nEvaluation metrics for {model_name}:")
    print("-" * 50)
    
    # Group and print training metrics
    print("\nTraining Metrics:")
    train_metrics = {k.replace('train_', ''): v for k, v in metrics.items() if k.startswith('train_')}
    for metric, value in train_metrics.items():
        print(f"{metric:20}: {value:.4f}")
    
    # Group and print test metrics
    print("\nTest Metrics:")
    test_metrics = {k.replace('test_', ''): v for k, v in metrics.items() if k.startswith('test_')}
    for metric, value in test_metrics.items():
        print(f"{metric:20}: {value:.4f}")

def cross_validate_model(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
    """
    Perform cross-validation on a model
    
    Parameters:
    -----------
    model : object
        Model to cross-validate
    X : array-like
        Features
    y : array-like
        Target values
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='neg_mean_squared_error'
        Scoring metric to use
    n_jobs : int, default=-1
        Number of jobs to run in parallel
        
    Returns:
    --------
    tuple
        (mean_score, std_score)
    """
    scores = cross_val_score(
        model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs
    )
    
    # Convert negative scores to positive if needed
    if scoring.startswith('neg_'):
        scores = -scores
        metric_name = scoring.replace('neg_', '')
    else:
        metric_name = scoring
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\nCross-validation results ({cv} folds):")
    print(f"{metric_name}: {mean_score:.4f} ± {std_score:.4f}")
    
    return mean_score, std_score

def plot_residuals(y_true, y_pred, model_name="Model", figsize=(16, 12)):
    """
    Create residual plots for model diagnostics
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str, default="Model"
        Name of the model for display
    figsize : tuple, default=(16, 12)
        Figure size
        
    Returns:
    --------
    None
    """
    residuals = y_true - y_pred
    
    # Create subplot figure
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16)
    
    # Plot 1: Residuals vs. Predicted Values
    axs[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axs[0, 0].axhline(y=0, color='r', linestyle='--')
    axs[0, 0].set_title('Residuals vs Predicted Values')
    axs[0, 0].set_xlabel('Predicted Values')
    axs[0, 0].set_ylabel('Residuals')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of Residuals
    axs[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axs[0, 1].axvline(x=0, color='r', linestyle='--')
    axs[0, 1].set_title('Histogram of Residuals')
    axs[0, 1].set_xlabel('Residual Value')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Q-Q Plot
    stats.probplot(residuals, plot=axs[1, 0])
    axs[1, 0].set_title('Q-Q Plot')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals vs. Actual Values
    axs[1, 1].scatter(y_true, residuals, alpha=0.5)
    axs[1, 1].axhline(y=0, color='r', linestyle='--')
    axs[1, 1].set_title('Residuals vs Actual Values')
    axs[1, 1].set_xlabel('Actual Values')
    axs[1, 1].set_ylabel('Residuals')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    # Print residual statistics
    print("\nResidual Statistics:")
    print(f"Mean: {np.mean(residuals):.4f}")
    print(f"Standard Deviation: {np.std(residuals):.4f}")
    print(f"Min: {np.min(residuals):.4f}")
    print(f"Max: {np.max(residuals):.4f}")
    
    # Normality test
    _, p_value = stats.normaltest(residuals)
    print(f"\nNormality Test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("The residuals are NOT normally distributed (p < 0.05)")
    else:
        print("The residuals appear to be normally distributed (p >= 0.05)")

def plot_prediction_actual(y_true, y_pred, model_name="Model", figsize=(10, 8)):
    """
    Plot predicted vs actual values
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str, default="Model"
        Name of the model for display
    figsize : tuple, default=(10, 8)
        Figure size
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    
    # Scatter plot of actual vs predicted
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs Predicted Values - {model_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print metrics
    metrics = calculate_regression_metrics(y_true, y_pred)
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

def plot_prediction_distribution(y_true, y_pred, model_name="Model", figsize=(12, 6)):
    """
    Plot distribution of actual and predicted values
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str, default="Model"
        Name of the model for display
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    
    sns.kdeplot(y_true, label='Actual', shade=True)
    sns.kdeplot(y_pred, label='Predicted', shade=True)
    
    plt.title(f'Distribution of Actual and Predicted Values - {model_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print descriptive statistics
    print("\nDescriptive Statistics:")
    
    actual_stats = {
        'Mean': np.mean(y_true),
        'Median': np.median(y_true),
        'Std Dev': np.std(y_true),
        'Min': np.min(y_true),
        'Max': np.max(y_true)
    }
    
    predicted_stats = {
        'Mean': np.mean(y_pred),
        'Median': np.median(y_pred),
        'Std Dev': np.std(y_pred),
        'Min': np.min(y_pred),
        'Max': np.max(y_pred)
    }
    
    stats_df = pd.DataFrame({
        'Actual': actual_stats,
        'Predicted': predicted_stats
    })
    
    print(stats_df)

def plot_error_distribution(y_true, y_pred, model_name="Model", figsize=(12, 6), 
                           error_type='absolute'):
    """
    Plot distribution of prediction errors
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str, default="Model"
        Name of the model for display
    figsize : tuple, default=(12, 6)
        Figure size
    error_type : str, default='absolute'
        Type of error to plot ('absolute', 'relative', 'squared')
        
    Returns:
    --------
    None
    """
    if error_type == 'absolute':
        errors = np.abs(y_true - y_pred)
        title = 'Absolute Error Distribution'
        xlabel = 'Absolute Error'
    elif error_type == 'relative':
        # Avoid division by zero
        mask = y_true != 0
        errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
        title = 'Relative Error Distribution (%)'
        xlabel = 'Relative Error (%)'
    elif error_type == 'squared':
        errors = (y_true - y_pred) ** 2
        title = 'Squared Error Distribution'
        xlabel = 'Squared Error'
    else:
        raise ValueError("error_type must be 'absolute', 'relative', or 'squared'")
    
    plt.figure(figsize=figsize)
    
    sns.histplot(errors, kde=True, bins=30)
    
    plt.title(f'{title} - {model_name}')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print error statistics
    print(f"\n{error_type.capitalize()} Error Statistics:")
    print(f"Mean: {np.mean(errors):.4f}")
    print(f"Median: {np.median(errors):.4f}")
    print(f"Std Dev: {np.std(errors):.4f}")
    print(f"Min: {np.min(errors):.4f}")
    print(f"Max: {np.max(errors):.4f}")
    print(f"90th Percentile: {np.percentile(errors, 90):.4f}")
    print(f"95th Percentile: {np.percentile(errors, 95):.4f}")

def plot_feature_importance(model, feature_names, top_n=20, figsize=(12, 10)):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int, default=20
        Number of top features to show
    figsize : tuple, default=(12, 10)
        Figure size
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature importances
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None
    
    # Create DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Limit to top N features
    if len(importance_df) > top_n:
        plot_df = importance_df.head(top_n).copy()
    else:
        plot_df = importance_df.copy()
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=plot_df)
    plt.title(f'Top {len(plot_df)} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return importance_df

def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
                       scoring='neg_mean_squared_error', figsize=(10, 6)):
    """
    Plot learning curve for a model
    
    Parameters:
    -----------
    model : object
        Model to evaluate
    X : array-like
        Features
    y : array-like
        Target values
    cv : int, default=5
        Number of cross-validation folds
    train_sizes : array-like, default=np.linspace(0.1, 1.0, 10)
        Training set sizes to plot
    scoring : str, default='neg_mean_squared_error'
        Scoring metric to use
    figsize : tuple, default=(10, 6)
        Figure size
        
    Returns:
    --------
    tuple
        (train_sizes, train_scores, test_scores)
    """
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring,
        n_jobs=-1, shuffle=True, random_state=42
    )
    
    # Calculate mean and std of training and test scores
    train_mean = -np.mean(train_scores, axis=1) if scoring.startswith('neg_') else np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1) if scoring.startswith('neg_') else np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create metric name for display
    if scoring.startswith('neg_'):
        metric_name = scoring.replace('neg_', '').replace('_', ' ').title()
    else:
        metric_name = scoring.replace('_', ' ').title()
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation Score')
    
    # Add error bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    plt.title(f'Learning Curve - {metric_name}')
    plt.xlabel('Training Examples')
    plt.ylabel(metric_name)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return train_sizes, train_scores, test_scores

def compare_models(models, X_train, X_test, y_train, y_test, metric='test_RMSE'):
    """
    Compare multiple models
    
    Parameters:
    -----------
    models : dict
        Dictionary of {model_name: model_object}
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training target values
    y_test : array-like
        Test target values
    metric : str, default='test_RMSE'
        Metric to use for comparison
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with model comparison results
    """
    results = []
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        start_time = time.time()
        
        # Train the model if not already fitted
        if not hasattr(model, 'predict'):
            model.fit(X_train, y_train)
            models[model_name] = model  # Update with fitted model
        
        # Evaluate the model
        metrics, _, _ = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Add training time and store results
        training_time = time.time() - start_time
        metrics['training_time'] = training_time
        metrics['model_name'] = model_name
        
        results.append(metrics)
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put model_name first
    cols = ['model_name'] + [col for col in results_df.columns if col != 'model_name']
    results_df = results_df[cols]
    
    # Sort by the specified metric
    if metric in results_df.columns:
        # Sort in ascending or descending order based on metric type
        ascending = metric.startswith(('train_RMSE', 'test_RMSE', 'train_MAE', 'test_MAE'))
        results_df = results_df.sort_values(metric, ascending=ascending)
    
    return results_df

def plot_model_comparison(results_df, metrics=None, figsize=(12, 8)):
    """
    Plot model comparison
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with model comparison results
    metrics : list, default=None
        List of metrics to plot (if None, use common regression metrics)
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    None
    """
    if metrics is None:
        metrics = ['test_RMSE', 'test_MAE', 'test_R²']
    
    # Check if metrics exist in results
    available_metrics = [m for m in metrics if m in results_df.columns]
    if not available_metrics:
        print("No specified metrics found in results")
        return
    
    # Create figure
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # Handle case with only one metric
    if n_metrics == 1:
        axes = [axes]
    
    # Sort models by first metric
    if available_metrics:
        ascending = available_metrics[0].startswith(('train_RMSE', 'test_RMSE', 'train_MAE', 'test_MAE'))
        sorted_results = results_df.sort_values(available_metrics[0], ascending=ascending)
    else:
        sorted_results = results_df
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        # Determine if lower is better for this metric
        is_lower_better = metric.startswith(('train_RMSE', 'test_RMSE', 'train_MAE', 'test_MAE'))
        
        # Create bar chart
        sns.barplot(x='model_name', y=metric, data=sorted_results, ax=axes[i])
        
        # Add value labels on bars
        for j, bar in enumerate(axes[i].patches):
            axes[i].text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + (sorted_results[metric].max() * 0.01),
                f"{bar.get_height():.4f}",
                ha='center', va='bottom', rotation=90
            )
        
        # Set title and labels
        metric_name = metric.replace('train_', 'Train ').replace('test_', 'Test ')
        axes[i].set_title(metric_name)
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def calculate_prediction_intervals(model, X, y, confidence=0.95, method='bootstrap', n_samples=1000):
    """
    Calculate prediction intervals
    
    Parameters:
    -----------
    model : object
        Trained model
    X : array-like
        Features
    y : array-like
        Target values
    confidence : float, default=0.95
        Confidence level for intervals
    method : str, default='bootstrap'
        Method to use ('bootstrap', 'residual')
    n_samples : int, default=1000
        Number of bootstrap samples
        
    Returns:
    --------
    tuple
        (predictions, lower_bounds, upper_bounds)
    """
    y_pred = model.predict(X)
    
    if method == 'bootstrap':
        # Bootstrap method
        predictions = []
        
        for _ in range(n_samples):
            # Sample with replacement
            sample_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            # Train model on bootstrap sample
            bootstrap_model = clone(model)
            bootstrap_model.fit(X_sample, y_sample)
            
            # Make predictions
            predictions.append(bootstrap_model.predict(X))
        
        # Calculate bounds
        predictions = np.array(predictions)
        lower_bound = np.percentile(predictions, (1 - confidence) / 2 * 100, axis=0)
        upper_bound = np.percentile(predictions, (1 + confidence) / 2 * 100, axis=0)
        
    elif method == 'residual':
        # Residual method
        residuals = y - y_pred
        
        # Calculate standard deviation of residuals
        residual_std = np.std(residuals)
        
        # Calculate z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Calculate bounds
        lower_bound = y_pred - z_score * residual_std
        upper_bound = y_pred + z_score * residual_std
        
    else:
        raise ValueError("method must be 'bootstrap' or 'residual'")
    
    return y_pred, lower_bound, upper_bound

def plot_prediction_intervals(y_true, y_pred, lower_bound, upper_bound, 
                            model_name="Model", figsize=(10, 8)):
    """
    Plot predictions with intervals
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    lower_bound : array-like
        Lower bound of prediction interval
    upper_bound : array-like
        Upper bound of prediction interval
    model_name : str, default="Model"
        Name of the model for display
    figsize : tuple, default=(10, 8)
        Figure size
        
    Returns:
    --------
    None
    """
    # Sort by predicted value for clearer visualization
    sorted_indices = np.argsort(y_pred)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    lower_sorted = lower_bound[sorted_indices]
    upper_sorted = upper_bound[sorted_indices]
    
    # Create range for x-axis
    x_range = np.arange(len(y_pred_sorted))
    
    plt.figure(figsize=figsize)
    
    # Plot predictions
    plt.plot(x_range, y_pred_sorted, 'bo-', label='Predictions', alpha=0.6)
    
    # Plot actual values
    plt.plot(x_range, y_true_sorted, 'ro', label='Actual Values', alpha=0.4)
    
    # Plot prediction intervals
    plt.fill_between(x_range, lower_sorted, upper_sorted, color='blue', alpha=0.2, label='Prediction Interval')
    
    plt.title(f'Predictions with Intervals - {model_name}')
    plt.xlabel('Sample Index (sorted by predicted value)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate interval coverage
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    print(f"\nPrediction interval coverage: {coverage:.4f}")
    
    # Calculate average interval width
    avg_width = np.mean(upper_bound - lower_bound)
    print(f"Average interval width: {avg_width:.4f}")

if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = 3*X[:,0] + 2*X[:,1] - X[:,2] + np.random.normal(0, 0.5, 100)
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    metrics, y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    print_metrics(metrics, model_name="Random Forest")
    
    # Plot residuals
    plot_residuals(y_test, y_test_pred, model_name="Random Forest")
    
    # Plot actual vs predicted
    plot_prediction_actual(y_test, y_test_pred, model_name="Random Forest")
    
    # Compare multiple models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results_df = compare_models(models, X_train, X_test, y_train, y_test)
    print("\nModel Comparison Results:")
    print(results_df)