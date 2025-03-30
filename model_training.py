"""
Model training utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import joblib

def train_linear_model(X_train, y_train, model_type='linear', alpha=1.0, l1_ratio=0.5):
    """
    Train a linear regression model
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    model_type : str, default='linear'
        Type of linear model ('linear', 'ridge', 'lasso', 'elastic_net')
    alpha : float, default=1.0
        Regularization strength
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter
        
    Returns:
    --------
    object
        Trained model
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha)
    elif model_type == 'elastic_net':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    else:
        raise ValueError("model_type must be 'linear', 'ridge', 'lasso', or 'elastic_net'")
    
    # Train model
    model.fit(X_train, y_train)
    
    return model

def train_tree_model(X_train, y_train, model_type='decision_tree', 
                    n_estimators=100, max_depth=None, random_state=42):
    """
    Train a tree-based model
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    model_type : str, default='decision_tree'
        Type of tree model ('decision_tree', 'random_forest', 'gradient_boosting')
    n_estimators : int, default=100
        Number of estimators (for ensemble methods)
    max_depth : int, default=None
        Maximum depth of the tree
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    object
        Trained model
    """
    if model_type == 'decision_tree':
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
    elif model_type == 'adaboost':
        model = AdaBoostRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
    else:
        raise ValueError(
            "model_type must be 'decision_tree', 'random_forest', 'gradient_boosting', or 'adaboost'"
        )
    
    # Train model
    model.fit(X_train, y_train)
    
    return model

def train_other_models(X_train, y_train, model_type='svr', **kwargs):
    """
    Train other types of models
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    model_type : str, default='svr'
        Type of model ('svr', 'knn', 'mlp')
    **kwargs : dict
        Additional parameters for the model
        
    Returns:
    --------
    object
        Trained model
    """
    if model_type == 'svr':
        model = SVR(**kwargs)
    elif model_type == 'knn':
        model = KNeighborsRegressor(**kwargs)
    elif model_type == 'mlp':
        model = MLPRegressor(**kwargs)
    else:
        raise ValueError("model_type must be 'svr', 'knn', or 'mlp'")
    
    # Train model
    model.fit(X_train, y_train)
    
    return model

def train_xgboost_model(X_train, y_train, **kwargs):
    """
    Train an XGBoost model
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Parameters for XGBoost
        
    Returns:
    --------
    object
        Trained model
    """
    try:
        from xgboost import XGBRegressor
        
        # Create model with default or provided parameters
        model = XGBRegressor(**kwargs)
        
        # Train model
        model.fit(X_train, y_train)
        
        return model
    except ImportError:
        print("XGBoost is not installed. Please install it with: pip install xgboost")
        return None

def train_lightgbm_model(X_train, y_train, **kwargs):
    """
    Train a LightGBM model
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Parameters for LightGBM
        
    Returns:
    --------
    object
        Trained model
    """
    try:
        from lightgbm import LGBMRegressor
        
        # Create model with default or provided parameters
        model = LGBMRegressor(**kwargs)
        
        # Train model
        model.fit(X_train, y_train)
        
        return model
    except ImportError:
        print("LightGBM is not installed. Please install it with: pip install lightgbm")
        return None

def train_catboost_model(X_train, y_train, **kwargs):
    """
    Train a CatBoost model
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Parameters for CatBoost
        
    Returns:
    --------
    object
        Trained model
    """
    try:
        from catboost import CatBoostRegressor
        
        # Create model with default or provided parameters
        model = CatBoostRegressor(silent=True, **kwargs)
        
        # Train model
        model.fit(X_train, y_train)
        
        return model
    except ImportError:
        print("CatBoost is not installed. Please install it with: pip install catboost")
        return None

def train_multiple_models(X_train, y_train, X_test, y_test, models_to_train=None):
    """
    Train multiple models and compare their performance
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_test : array-like
        Testing features
    y_test : array-like
        Testing target
    models_to_train : list, default=None
        List of model names to train (if None, train all available models)
        
    Returns:
    --------
    dict
        Dictionary of trained models
    pandas.DataFrame
        DataFrame with model performance metrics
    """
    all_models = {
        'linear_regression': (LinearRegression, {}),
        'ridge': (Ridge, {'alpha': 1.0}),
        'lasso': (Lasso, {'alpha': 0.1}),
        'elastic_net': (ElasticNet, {'alpha': 0.1, 'l1_ratio': 0.5}),
        'decision_tree': (DecisionTreeRegressor, {'max_depth': 10, 'random_state': 42}),
        'random_forest': (RandomForestRegressor, {'n_estimators': 100, 'random_state': 42}),
        'gradient_boosting': (GradientBoostingRegressor, {'n_estimators': 100, 'random_state': 42}),
        'svr': (SVR, {'kernel': 'rbf', 'C': 1.0}),
        'knn': (KNeighborsRegressor, {'n_neighbors': 5})
    }
    
    # Try to import optional models
    try:
        from xgboost import XGBRegressor
        all_models['xgboost'] = (XGBRegressor, {'n_estimators': 100, 'random_state': 42})
    except ImportError:
        pass
    
    try:
        from lightgbm import LGBMRegressor
        all_models['lightgbm'] = (LGBMRegressor, {'n_estimators': 100, 'random_state': 42})
    except ImportError:
        pass
    
    try:
        from catboost import CatBoostRegressor
        all_models['catboost'] = (CatBoostRegressor, {'iterations': 100, 'random_seed': 42, 'silent': True})
    except ImportError:
        pass
    
    # Filter models to train
    if models_to_train is not None:
        models_to_train = {k: v for k, v in all_models.items() if k in models_to_train}
    else:
        models_to_train = all_models
    
    # Train models and store results
    trained_models = {}
    results = []
    
    for name, (model_class, params) in models_to_train.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        # Initialize and train model
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Store results
        trained_models[name] = model
        results.append({
            'Model': name,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Training Time (s)': elapsed_time
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    return trained_models, results_df

def plot_feature_importance(model, feature_names, top_n=20, figsize=(12, 8)):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : list
        List of feature names
    top_n : int, default=20
        Number of top features to show
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature importances
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature importances.")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return importance_df

def plot_model_comparison(results_df, metric='Test R²', figsize=(12, 8)):
    """
    Plot model comparison
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with model results
    metric : str, default='Test R²'
        Metric to compare models
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    sns.barplot(x=metric, y='Model', data=results_df.sort_values(metric, ascending=False))
    plt.title(f'Model Comparison by {metric}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_prediction_vs_actual(y_true, y_pred, model_name='Model', figsize=(10, 8)):
    """
    Plot predicted vs actual values
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str, default='Model'
        Name of the model
    figsize : tuple, default=(10, 8)
        Figure size
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

def save_model(model, filename='model.joblib'):
    """
    Save model to file
    
    Parameters:
    -----------
    model : object
        Trained model
    filename : str, default='model.joblib'
        Output filename
        
    Returns:
    --------
    str
        Path to saved model
    """
    try:
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def load_model(filename='model.joblib'):
    """
    Load model from file
    
    Parameters:
    -----------
    filename : str, default='model.joblib'
        Input filename
        
    Returns:
    --------
    object
        Loaded model
    """
    try:
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    import data_loader
    import data_preprocessing
    
    data_path = "your_dataset.csv"
    df = data_loader.load_data(data_path)
    
    if df is not None:
        # Preprocess data
        target_column = 'price'  # Replace with your target column
        df_clean = data_preprocessing.handle_missing_values(df)
        
        # Split data
        X_train, X_test, y_train, y_test, preprocessor = data_preprocessing.prepare_train_test_set(
            df_clean, target_column=target_column
        )
        
        # Apply preprocessing
        X_train_processed, X_test_processed, feature_names = data_preprocessing.apply_preprocessing(
            X_train, X_test, preprocessor
        )
        
        # Train multiple models
        trained_models, results_df = train_multiple_models(
            X_train_processed, y_train, X_test_processed, y_test
        )
        
        print("\nModel Performance Comparison:")
        print(results_df)
        
        # Plot model comparison
        plot_model_comparison(results_df)
        
        # Get best model
        best_model_name = results_df.iloc[0]['Model']
        best_model = trained_models[best_model_name]
        
        # Plot feature importance for best model if applicable
        if hasattr(best_model, 'feature_importances_'):
            importance_df = plot_feature_importance(best_model, feature_names)
            
        # Plot predictions vs actual for best model
        y_test_pred = best_model.predict(X_test_processed)
        plot_prediction_vs_actual(y_test, y_test_pred, model_name=best_model_name)
        
        # Save best model
        save_model(best_model, filename=f'{best_model_name.lower()}_model.joblib')