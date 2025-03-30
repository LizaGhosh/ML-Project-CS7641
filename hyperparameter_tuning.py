"""
Hyperparameter tuning utilities for machine learning models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV, 
    cross_val_score,
    KFold
)
import time
from scipy.stats import uniform, randint, loguniform
import itertools

def grid_search_tuning(model, param_grid, X_train, y_train, X_test=None, y_test=None, 
                      cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1):
    """
    Perform grid search hyperparameter tuning
    
    Parameters:
    -----------
    model : estimator object
        Model to tune
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter values
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like, default=None
        Test features (if provided, will evaluate best model on test set)
    y_test : array-like, default=None
        Test target values
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='neg_mean_squared_error'
        Scoring metric
    n_jobs : int, default=-1
        Number of jobs to run in parallel
    verbose : int, default=1
        Verbosity level
        
    Returns:
    --------
    tuple
        (grid_search object, best_params, results_df)
    """
    start_time = time.time()
    
    # Create grid search object
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Convert negative score to positive if needed
    if scoring.startswith('neg_'):
        best_score = -best_score
        score_name = scoring.replace('neg_', '')
    else:
        score_name = scoring
    
    print(f"\nGrid search completed in {elapsed_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best CV {score_name}: {best_score:.6f}")
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        best_model = grid_search.best_estimator_
        test_score = -cross_val_score(
            best_model, X_test, y_test, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring
        ).mean() if scoring.startswith('neg_') else cross_val_score(
            best_model, X_test, y_test, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring
        ).mean()
        
        print(f"Test set {score_name}: {test_score:.6f}")
    
    # Create DataFrame with results
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Sort by rank
    results = results.sort_values('rank_test_score')
    
    return grid_search, best_params, results

def randomized_search_tuning(model, param_distributions, X_train, y_train, X_test=None, y_test=None,
                            n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1):
    """
    Perform randomized search hyperparameter tuning
    
    Parameters:
    -----------
    model : estimator object
        Model to tune
    param_distributions : dict
        Dictionary with parameters names as keys and distributions or lists of parameters
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like, default=None
        Test features (if provided, will evaluate best model on test set)
    y_test : array-like, default=None
        Test target values
    n_iter : int, default=100
        Number of parameter settings sampled
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='neg_mean_squared_error'
        Scoring metric
    n_jobs : int, default=-1
        Number of jobs to run in parallel
    verbose : int, default=1
        Verbosity level
        
    Returns:
    --------
    tuple
        (randomized_search object, best_params, results_df)
    """
    start_time = time.time()
    
    # Create randomized search object
    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        random_state=42
    )
    
    # Fit randomized search
    randomized_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    
    # Get best parameters and score
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_
    
    # Convert negative score to positive if needed
    if scoring.startswith('neg_'):
        best_score = -best_score
        score_name = scoring.replace('neg_', '')
    else:
        score_name = scoring
    
    print(f"\nRandomized search completed in {elapsed_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best CV {score_name}: {best_score:.6f}")
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        best_model = randomized_search.best_estimator_
        test_score = -cross_val_score(
            best_model, X_test, y_test, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring
        ).mean() if scoring.startswith('neg_') else cross_val_score(
            best_model, X_test, y_test, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring
        ).mean()
        
        print(f"Test set {score_name}: {test_score:.6f}")
    
    # Create DataFrame with results
    results = pd.DataFrame(randomized_search.cv_results_)
    
    # Sort by rank
    results = results.sort_values('rank_test_score')
    
    return randomized_search, best_params, results

def plot_search_results(results_df, param_name, score_metric='mean_test_score', n_top=10,
                       is_negative_score=True, figsize=(12, 6)):
    """
    Plot grid search or randomized search results for a specific parameter
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with search results
    param_name : str
        Parameter name to plot
    score_metric : str, default='mean_test_score'
        Metric to use for scoring
    n_top : int, default=10
        Number of top results to display
    is_negative_score : bool, default=True
        Whether the score is negative (e.g., neg_mean_squared_error)
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    None
    """
    # Construct the param column name
    param_col = f'param_{param_name}'
    
    # Check if parameter exists in results
    if param_col not in results_df.columns:
        print(f"Parameter '{param_name}' not found in results.")
        return
    
    # Extract parameter values and scores
    param_values = results_df[param_col].astype(str)
    scores = results_df[score_metric]
    
    # Convert negative scores to positive if needed
    if is_negative_score:
        scores = -scores
        score_name = score_metric.replace('mean_test_', '').replace('neg_', '')
    else:
        score_name = score_metric.replace('mean_test_', '')
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Parameter Value': param_values,
        'Score': scores
    })
    
    # Group by parameter value and calculate mean score
    plot_df = plot_df.groupby('Parameter Value').mean().reset_index()
    
    # Sort by score
    plot_df = plot_df.sort_values('Score')
    
    # Limit to top N results
    if len(plot_df) > n_top:
        plot_df = plot_df.head(n_top)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(x='Parameter Value', y='Score', data=plot_df)
    plt.title(f'Impact of {param_name} on {score_name}')
    plt.xlabel(param_name)
    plt.ylabel(score_name)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_parameter_interactions(results_df, param1, param2, score_metric='mean_test_score',
                               is_negative_score=True, figsize=(12, 10)):
    """
    Plot interaction between two parameters
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with search results
    param1 : str
        First parameter name
    param2 : str
        Second parameter name
    score_metric : str, default='mean_test_score'
        Metric to use for scoring
    is_negative_score : bool, default=True
        Whether the score is negative (e.g., neg_mean_squared_error)
    figsize : tuple, default=(12, 10)
        Figure size
        
    Returns:
    --------
    None
    """
    # Construct the param column names
    param1_col = f'param_{param1}'
    param2_col = f'param_{param2}'
    
    # Check if parameters exist in results
    if param1_col not in results_df.columns or param2_col not in results_df.columns:
        print(f"One or both parameters not found in results.")
        return
    
    # Extract parameter values and scores
    param1_values = results_df[param1_col].astype(str)
    param2_values = results_df[param2_col].astype(str)
    scores = results_df[score_metric]
    
    # Convert negative scores to positive if needed
    if is_negative_score:
        scores = -scores
        score_name = score_metric.replace('mean_test_', '').replace('neg_', '')
    else:
        score_name = score_metric.replace('mean_test_', '')
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        param1: param1_values,
        param2: param2_values,
        'Score': scores
    })
    
    # Group by both parameters and calculate mean score
    plot_df = plot_df.groupby([param1, param2]).mean().reset_index()
    
    # Create pivot table for heatmap
    pivot_df = plot_df.pivot(index=param1, columns=param2, values='Score')
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'Interaction between {param1} and {param2} ({score_name})')
    plt.tight_layout()
    plt.show()
    
    # Also create a line plot with param1 on x-axis, different lines for param2
    plt.figure(figsize=figsize)
    for p2 in plot_df[param2].unique():
        subset = plot_df[plot_df[param2] == p2]
        plt.plot(subset[param1], subset['Score'], 'o-', label=str(p2))
    
    plt.title(f'Interaction between {param1} and {param2} ({score_name})')
    plt.xlabel(param1)
    plt.ylabel(score_name)
    plt.legend(title=param2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_tuning_params_for_model(model_type):
    """
    Get hyperparameter grids for common model types
    
    Parameters:
    -----------
    model_type : str
        Type of model ('linear', 'ridge', 'lasso', 'elastic_net', 'decision_tree',
        'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')
        
    Returns:
    --------
    tuple
        (grid_search_params, randomized_search_params)
    """
    # Dictionary of parameter grids for grid search
    grid_params = {}
    
    # Dictionary of parameter distributions for randomized search
    random_params = {}
    
    if model_type == 'linear':
        # Linear Regression has few tunable hyperparameters
        grid_params = {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
        random_params = {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
    
    elif model_type == 'ridge':
        grid_params = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
        random_params = {
            'alpha': loguniform(0.001, 100),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    
    elif model_type == 'lasso':
        grid_params = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'max_iter': [1000, 3000, 5000],
            'selection': ['cyclic', 'random']
        }
        random_params = {
            'alpha': loguniform(0.001, 10),
            'max_iter': randint(1000, 10000),
            'selection': ['cyclic', 'random']
        }
    
    elif model_type == 'elastic_net':
        grid_params = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [1000, 5000],
            'selection': ['cyclic', 'random']
        }
        random_params = {
            'alpha': loguniform(0.001, 10),
            'l1_ratio': uniform(0, 1),
            'max_iter': randint(1000, 10000),
            'selection': ['cyclic', 'random']
        }
    
    elif model_type == 'decision_tree':
        grid_params = {
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2', None]
        }
        random_params = {
            'max_depth': [None] + list(randint(3, 30).rvs(10)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['auto', 'sqrt', 'log2', None]
        }
    
    elif model_type == 'random_forest':
        grid_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }
        random_params = {
            'n_estimators': randint(10, 300),
            'max_depth': [None] + list(randint(5, 50).rvs(10)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['auto', 'sqrt', 'log2']
        }
    
    elif model_type == 'gradient_boosting':
        grid_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        random_params = {
            'n_estimators': randint(50, 300),
            'learning_rate': loguniform(0.01, 0.5),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'subsample': uniform(0.7, 0.3)
        }
    
    elif model_type == 'xgboost':
        grid_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        random_params = {
            'n_estimators': randint(50, 300),
            'learning_rate': loguniform(0.01, 0.5),
            'max_depth': randint(3, 15),
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 0.5),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3)
        }
    
    elif model_type == 'lightgbm':
        grid_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'num_leaves': [31, 50, 70, 90],
            'min_child_samples': [5, 10, 20],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        random_params = {
            'n_estimators': randint(50, 300),
            'learning_rate': loguniform(0.01, 0.5),
            'max_depth': randint(3, 15),
            'num_leaves': randint(20, 100),
            'min_child_samples': randint(5, 30),
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3)
        }
    
    else:
        print(f"Model type '{model_type}' not recognized")
    
    return grid_params, random_params

def tune_multiple_models(models_to_tune, X_train, y_train, X_test=None, y_test=None,
                       scoring='neg_mean_squared_error', cv=5, method='grid', n_jobs=-1):
    """
    Tune multiple models and compare their performance
    
    Parameters:
    -----------
    models_to_tune : dict
        Dictionary of {model_name: model_object}
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like, default=None
        Test features
    y_test : array-like, default=None
        Test target values
    scoring : str, default='neg_mean_squared_error'
        Scoring metric
    cv : int, default=5
        Number of cross-validation folds
    method : str, default='grid'
        Tuning method ('grid', 'random', 'both')
    n_jobs : int, default=-1
        Number of jobs to run in parallel
        
    Returns:
    --------
    dict
        Dictionary with tuning results for each model
    """
    results = {}
    
    for model_name, model in models_to_tune.items():
        print(f"\n{'-'*50}")
        print(f"Tuning {model_name}...")
        
        # Determine model type
        model_type = model.__class__.__name__.lower()
        if 'linear' in model_type:
            model_type = 'linear'
        elif 'ridge' in model_type:
            model_type = 'ridge'
        elif 'lasso' in model_type:
            model_type = 'lasso'
        elif 'elasticnet' in model_type:
            model_type = 'elastic_net'
        elif 'decisiontree' in model_type:
            model_type = 'decision_tree'
        elif 'randomforest' in model_type:
            model_type = 'random_forest'
        elif 'gradientboosting' in model_type:
            model_type = 'gradient_boosting'
        elif 'xgb' in model_type:
            model_type = 'xgboost'
        elif 'lgbm' in model_type:
            model_type = 'lightgbm'
        
        # Get hyperparameter grids
        grid_params, random_params = get_tuning_params_for_model(model_type)
        
        model_results = {}
        
        # Perform grid search
        if method in ['grid', 'both'] and grid_params:
            print("\nPerforming Grid Search...")
            grid_search, grid_best_params, grid_results = grid_search_tuning(
                model=model,
                param_grid=grid_params,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs
            )
            model_results['grid_search'] = {
                'search_object': grid_search,
                'best_params': grid_best_params,
                'results': grid_results,
                'best_estimator': grid_search.best_estimator_
            }
        
        # Perform randomized search
        if method in ['random', 'both'] and random_params:
            print("\nPerforming Randomized Search...")
            random_search, random_best_params, random_results = randomized_search_tuning(
                model=model,
                param_distributions=random_params,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs
            )
            model_results['random_search'] = {
                'search_object': random_search,
                'best_params': random_best_params,
                'results': random_results,
                'best_estimator': random_search.best_estimator_
            }
        
        results[model_name] = model_results
    
    return results

def get_best_models(tuning_results, method='grid'):
    """
    Extract best models from tuning results
    
    Parameters:
    -----------
    tuning_results : dict
        Dictionary with tuning results
    method : str, default='grid'
        Which method to use for best models ('grid', 'random')
        
    Returns:
    --------
    dict
        Dictionary of {model_name: best_estimator}
    """
    best_models = {}
    
    for model_name, results in tuning_results.items():
        if method == 'grid' and 'grid_search' in results:
            best_models[model_name] = results['grid_search']['best_estimator']
        elif method == 'random' and 'random_search' in results:
            best_models[model_name] = results['random_search']['best_estimator']
        elif method == 'both':
            # Use grid search if available, otherwise use randomized search
            if 'grid_search' in results:
                best_models[model_name] = results['grid_search']['best_estimator']
            elif 'random_search' in results:
                best_models[model_name] = results['random_search']['best_estimator']
    
    return best_models

def plot_tuning_results_comparison(tuning_results, scoring='mean_test_score',
                                 is_negative_score=True, figsize=(12, 8)):
    """
    Plot comparison of tuning results for multiple models
    
    Parameters:
    -----------
    tuning_results : dict
        Dictionary with tuning results
    scoring : str, default='mean_test_score'
        Scoring metric
    is_negative_score : bool, default=True
        Whether the score is negative (e.g., neg_mean_squared_error)
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with best scores for each model and method
    """
    results = []
    
    for model_name, model_results in tuning_results.items():
        model_row = {'Model': model_name}
        
        if 'grid_search' in model_results:
            grid_search = model_results['grid_search']['search_object']
            best_score = grid_search.best_score_
            
            if is_negative_score:
                best_score = -best_score
            
            model_row['Grid Search Score'] = best_score
            model_row['Grid Search Params'] = str(grid_search.best_params_)
        
        if 'random_search' in model_results:
            random_search = model_results['random_search']['search_object']
            best_score = random_search.best_score_
            
            if is_negative_score:
                best_score = -best_score
            
            model_row['Random Search Score'] = best_score
            model_row['Random Search Params'] = str(random_search.best_params_)
        
        results.append(model_row)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Check which scores are available
    has_grid = 'Grid Search Score' in results_df.columns
    has_random = 'Random Search Score' in results_df.columns
    
    # Calculate positions
    models = results_df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    # Plot bars
    if has_grid:
        plt.bar(x - width/2 if has_random else x, results_df['Grid Search Score'], 
               width=width, label='Grid Search')
    
    if has_random:
        plt.bar(x + width/2 if has_grid else x, results_df['Random Search Score'], 
               width=width, label='Random Search')
    
    # Add labels and legend
    plt.xlabel('Model')
    
    # Format score name for display
    if is_negative_score:
        score_name = scoring.replace('mean_test_', '').replace('neg_', '')
    else:
        score_name = scoring.replace('mean_test_', '')
    
    plt.ylabel(score_name.replace('_', ' ').title())
    plt.title('Comparison of Tuning Results')
    plt.xticks(x, models, rotation=45, ha='right')
    
    if has_grid or has_random:
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results_df

def perform_sequential_tuning(model, X_train, y_train, X_test=None, y_test=None,
                            param_grids, cv=5, scoring='neg_mean_squared_error', n_jobs=-1):
    """
    Perform sequential hyperparameter tuning
    
    Parameters:
    -----------
    model : estimator object
        Model to tune
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like, default=None
        Test features
    y_test : array-like, default=None
        Test target values
    param_grids : list of dict
        List of parameter grids to use in sequence
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='neg_mean_squared_error'
        Scoring metric
    n_jobs : int, default=-1
        Number of jobs to run in parallel
        
    Returns:
    --------
    tuple
        (final_model, best_params, tuning_results)
    """
    current_model = model
    best_params = {}
    tuning_results = []
    
    print("\nPerforming sequential hyperparameter tuning...")
    
    for i, param_grid in enumerate(param_grids):
        print(f"\nTuning round {i+1}/{len(param_grids)}")
        print(f"Parameters: {list(param_grid.keys())}")
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=current_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters and update model
        round_best_params = grid_search.best_params_
        best_params.update(round_best_params)
        current_model = grid_search.best_estimator_
        
        # Store results
        tuning_results.append({
            'round': i+1,
            'params_tuned': list(param_grid.keys()),
            'best_params': round_best_params,
            'best_score': grid_search.best_score_
        })
        
        print(f"Best parameters for this round: {round_best_params}")
        
        # Convert negative score to positive if needed
        best_score = grid_search.best_score_
        if scoring.startswith('neg_'):
            best_score = -best_score
            score_name = scoring.replace('neg_', '')
        else:
            score_name = scoring
        
        print(f"Best {score_name}: {best_score:.6f}")
    
    # Evaluate final model on test set if provided
    if X_test is not None and y_test is not None:
        test_score = -cross_val_score(
            current_model, X_test, y_test, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring
        ).mean() if scoring.startswith('neg_') else cross_val_score(
            current_model, X_test, y_test, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring
        ).mean()
        
        # Convert negative score to positive if needed
        if scoring.startswith('neg_'):
            score_name = scoring.replace('neg_', '')
        else:
            score_name = scoring
        
        print(f"\nFinal model performance on test set ({score_name}): {test_score:.6f}")
    
    print("\nFinal best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return current_model, best_params, tuning_results

def create_bayesian_optimization(model, X_train, y_train, X_test=None, y_test=None,
                               param_space, cv=5, scoring='neg_mean_squared_error',
                               n_iter=50, random_state=42):
    """
    Perform Bayesian hyperparameter optimization using Scikit-Optimize
    
    Parameters:
    -----------
    model : estimator object
        Model to tune
    X_train : array-like
        Training features
    y_train : array-like
        Training target values
    X_test : array-like, default=None
        Test features
    y_test : array-like, default=None
        Test target values
    param_space : dict
        Dictionary with parameter names as keys and search spaces as values
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='neg_mean_squared_error'
        Scoring metric
    n_iter : int, default=50
        Number of iterations for optimization
    random_state : int, default=42
        Random seed
        
    Returns:
    --------
    tuple
        (optimized_model, best_params, all_results)
    """
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
    except ImportError:
        print("Scikit-Optimize is not installed. Please install it with:")
        print("pip install scikit-optimize")
        return None, None, None
    
    print("\nPerforming Bayesian hyperparameter optimization...")
    
    # Create Bayesian search object
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )
    
    # Fit search
    bayes_search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = bayes_search.best_params_
    best_model = bayes_search.best_estimator_
    
    # Convert negative score to positive if needed
    best_score = bayes_search.best_score_
    if scoring.startswith('neg_'):
        best_score = -best_score
        score_name = scoring.replace('neg_', '')
    else:
        score_name = scoring
    
    print(f"\nBest {score_name}: {best_score:.6f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        test_score = -cross_val_score(
            best_model, X_test, y_test, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring
        ).mean() if scoring.startswith('neg_') else cross_val_score(
            best_model, X_test, y_test, 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring
        ).mean()
        
        print(f"\nTest set {score_name}: {test_score:.6f}")
    
    # Process all results
    all_results = pd.DataFrame(bayes_search.cv_results_)
    
    return best_model, best_params, all_results

def plot_bayesian_optimization_results(results, param_name, is_negative_score=True, figsize=(12, 6)):
    """
    Plot results from Bayesian optimization for a specific parameter
    
    Parameters:
    -----------
    results : pandas.DataFrame
        DataFrame with optimization results
    param_name : str
        Parameter name to plot
    is_negative_score : bool, default=True
        Whether the score is negative (e.g., neg_mean_squared_error)
    figsize : tuple, default=(12, 6)
        Figure size
        
    Returns:
    --------
    None
    """
    # Check if parameter exists in results
    param_col = f'param_{param_name}'
    if param_col not in results.columns:
        print(f"Parameter '{param_name}' not found in results.")
        return
    
    # Extract values
    param_values = results[param_col]
    scores = results['mean_test_score']
    
    # Convert negative scores to positive if needed
    if is_negative_score:
        scores = -scores
        score_name = 'mean_test_score'.replace('mean_test_', '').replace('neg_', '')
    else:
        score_name = 'mean_test_score'.replace('mean_test_', '')
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.scatter(param_values, scores, alpha=0.6)
    
    # Add trend line if parameter is numeric
    if pd.api.types.is_numeric_dtype(param_values):
        try:
            # Sort for line plot
            sorted_indices = np.argsort(param_values)
            plt.plot(param_values.iloc[sorted_indices], scores.iloc[sorted_indices], 'r-')
        except:
            pass
    
    plt.title(f'Bayesian Optimization Results: {param_name} vs {score_name}')
    plt.xlabel(param_name)
    plt.ylabel(score_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model
    model = RandomForestRegressor(random_state=42)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search
    grid_search, best_params, results = grid_search_tuning(
        model=model,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    # Plot results for specific parameter
    plot_search_results(results, 'max_depth')