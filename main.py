"""
Main script for price prediction pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import logging
import time
import joblib

# Import utility modules
from data_loader import load_data, clean_column_names
from eda import (
    plot_distribution, 
    plot_correlation_matrix, 
    plot_target_by_category,
    identify_outliers
)
from feature_engineering import (
    group_rare_categories,
    create_interaction_features,
    create_polynomial_features,
    create_ratio_features
)
from data_preprocessing import (
    identify_column_types,
    handle_missing_values,
    encode_categorical_features,
    scale_numerical_features,
    handle_outliers,
    prepare_train_test_set,
    apply_preprocessing
)
from model_training import (
    train_linear_model,
    train_tree_model,
    train_xgboost_model,
    train_lightgbm_model,
    train_multiple_models,
    save_model
)
from model_evaluation import (
    evaluate_model,
    print_metrics,
    plot_residuals,
    plot_prediction_actual,
    plot_feature_importance,
    compare_models,
    plot_model_comparison
)
from hyperparameter_tuning import (
    get_tuning_params_for_model,
    grid_search_tuning,
    tune_multiple_models,
    get_best_models
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Price Prediction Pipeline')
    
    parser.add_argument('--input', type=str, required=True, help='Path to input data file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--target', type=str, default='price', help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--model', type=str, default='all', 
                       help='Model to use (linear, ridge, lasso, decision_tree, random_forest, xgboost, lightgbm, all)')
    
    return parser.parse_args()

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create subdirectories
    for subdir in ['models', 'plots', 'results']:
        path = os.path.join(output_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)

def save_plot(fig, filename, output_dir):
    """Save matplotlib figure to file"""
    path = os.path.join(output_dir, 'plots', filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved plot to {path}")

def save_results(results, filename, output_dir):
    """Save results to CSV file"""
    path = os.path.join(output_dir, 'results', filename)
    results.to_csv(path, index=False)
    logger.info(f"Saved results to {path}")

def run_pipeline(args):
    """Run the full prediction pipeline"""
    start_time = time.time()
    
    # Create output directory
    create_output_dir(args.output)
    
    # Load and clean data
    logger.info("Loading data...")
    df = load_data(args.input)
    df = clean_column_names(df)
    
    # Print basic information
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Target column: {args.target}")
    
    # Run EDA
    logger.info("Running EDA...")
    
    # Plot target distribution
    plt.figure(figsize=(10, 6))
    plot_distribution(df, args.target)
    plt.savefig(os.path.join(args.output, 'plots', 'target_distribution.png'))
    plt.close()
    
    # Log-transform target if it's skewed
    skewness = df[args.target].skew()
    logger.info(f"Target skewness: {skewness:.4f}")
    
    if abs(skewness) > 1:
        logger.info("Target is skewed, plotting log-transformed distribution")
        plt.figure(figsize=(10, 6))
        plot_distribution(df, args.target, log_transform=True)
        plt.savefig(os.path.join(args.output, 'plots', 'log_target_distribution.png'))
        plt.close()
    
    # Identify column types
    column_types = identify_column_types(df)
    
    # Count column types
    logger.info("Column type counts:")
    for type_name, columns in column_types.items():
        logger.info(f"  {type_name}: {len(columns)}")
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    plot_correlation_matrix(df, target_col=args.target)
    plt.savefig(os.path.join(args.output, 'plots', 'correlation_matrix.png'))
    plt.close()
    
    # Plot target by categorical variables
    for col in column_types['categorical'][:5]:  # Limit to 5 categories
        if df[col].nunique() < 15:  # Avoid too many categories
            plt.figure(figsize=(12, 6))
            plot_target_by_category(df, col, args.target)
            plt.savefig(os.path.join(args.output, 'plots', f'{col}_vs_target.png'))
            plt.close()
    
    # Identify outliers
    outliers = identify_outliers(df, column_types['numeric'] + [args.target])
    
    # Feature Engineering
    logger.info("Performing feature engineering...")
    
    # Group rare categories
    categorical_cols = column_types['categorical'] + column_types['binary']
    df = group_rare_categories(df, categorical_cols, threshold=0.01)
    
    # Create interaction features between top correlated features
    corr_matrix = df[column_types['numeric']].corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    
    # Get top correlations
    corr_pairs = []
    for i in range(min(5, len(corr_matrix))):
        max_idx = corr_matrix.values.argmax()
        row, col = max_idx // len(corr_matrix), max_idx % len(corr_matrix)
        feature1, feature2 = corr_matrix.index[row], corr_matrix.columns[col]
        corr_pairs.append((feature1, feature2))
        corr_matrix.iloc[row, col] = 0  # Set to 0 to find next pair
    
    if corr_pairs:
        logger.info(f"Creating interaction features for: {corr_pairs}")
        df = create_interaction_features(df, corr_pairs)
    
    # Create polynomial features for top numeric features
    corr_with_target = df[column_types['numeric']].corrwith(df[args.target]).abs().sort_values(ascending=False)
    top_features = corr_with_target.index[:3].tolist()  # Top 3 features
    
    if top_features:
        logger.info(f"Creating polynomial features for: {top_features}")
        df = create_polynomial_features(df, top_features, degree=2)
    
    # Create ratio features if applicable
    if len(column_types['numeric']) >= 2:
        logger.info("Creating ratio features")
        df = create_ratio_features(df, column_types['numeric'][:2], column_types['numeric'][2:4])
    
    # Data Preprocessing
    logger.info("Preprocessing data...")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df, method='clip')
    
    # Split data into train/test sets
    logger.info(f"Splitting data with test_size={args.test_size}")
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_set(
        df, 
        args.target, 
        test_size=args.test_size, 
        random_state=args.random_state
    )
    
    # Apply preprocessing
    X_train_processed, X_test_processed, feature_names = apply_preprocessing(
        X_train, X_test, preprocessor
    )
    
    logger.info(f"Processed training data shape: {X_train_processed.shape}")
    logger.info(f"Processed testing data shape: {X_test_processed.shape}")
    
    # Model Training
    logger.info("Training models...")
    
    # Determine which models to train
    if args.model == 'all':
        model_types = ['linear', 'ridge', 'lasso', 'decision_tree', 'random_forest']
        # Add optional models if available
        try:
            import xgboost
            model_types.append('xgboost')
        except ImportError:
            logger.warning("XGBoost not available, skipping")
        
        try:
            import lightgbm
            model_types.append('lightgbm')
        except ImportError:
            logger.warning("LightGBM not available, skipping")
    else:
        model_types = [args.model]
    
    logger.info(f"Models to train: {model_types}")
    
    # Initialize models
    models = {}
    
    for model_type in model_types:
        if model_type in ['linear', 'ridge', 'lasso', 'elastic_net']:
            alpha = 1.0 if model_type != 'linear' else None
            models[model_type] = train_linear_model(
                X_train_processed, y_train, model_type=model_type, alpha=alpha
            )
        elif model_type in ['decision_tree', 'random_forest', 'gradient_boosting']:
            models[model_type] = train_tree_model(
                X_train_processed, y_train, model_type=model_type
            )
        elif model_type == 'xgboost':
            models[model_type] = train_xgboost_model(
                X_train_processed, y_train
            )
        elif model_type == 'lightgbm':
            models[model_type] = train_lightgbm_model(
                X_train_processed, y_train
            )
    
    # Training all models at once
    if not models:
        logger.warning("No models specified, using train_multiple_models")
        models, results_df = train_multiple_models(
            X_train_processed, y_train, X_test_processed, y_test
        )
    else:
        # Evaluate individual models
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Evaluate model
            metrics, _, y_test_pred = evaluate_model(
                model, X_train_processed, X_test_processed, y_train, y_test
            )
            
            # Add to results
            metrics['model_name'] = model_name
            results.append(metrics)
            
            # Print metrics
            print_metrics(metrics, model_name=model_name)
            
            # Plot residuals and actual vs predicted
            plt.figure(figsize=(16, 12))
            plot_residuals(y_test, y_test_pred, model_name=model_name)
            plt.savefig(os.path.join(args.output, 'plots', f'{model_name}_residuals.png'))
            plt.close()
            
            plt.figure(figsize=(10, 8))
            plot_prediction_actual(y_test, y_test_pred, model_name=model_name)
            plt.savefig(os.path.join(args.output, 'plots', f'{model_name}_predictions.png'))
            plt.close()
            
            # Plot feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 10))
                importance_df = plot_feature_importance(model, feature_names)
                plt.savefig(os.path.join(args.output, 'plots', f'{model_name}_feature_importance.png'))
                plt.close()
                
                # Save importance to CSV
                importance_path = os.path.join(args.output, 'results', f'{model_name}_feature_importance.csv')
                importance_df.to_csv(importance_path, index=False)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['model_name'] + [col for col in results_df.columns if col != 'model_name']
        results_df = results_df[cols]
        
        # Sort by test R²
        results_df = results_df.sort_values('test_R²', ascending=False)
    
    # Save results
    save_results(results_df, 'model_results.csv', args.output)
    
    # Plot model comparison
    plt.figure(figsize=(12, 8))
    plot_model_comparison(results_df)
    plt.savefig(os.path.join(args.output, 'plots', 'model_comparison.png'))
    plt.close()
    
    # Hyperparameter tuning
    if args.tune:
        logger.info("Performing hyperparameter tuning...")
        
        # Get top 3 models
        top_models = {}
        for idx, row in results_df.head(3).iterrows():
            model_name = row['model_name']
            if model_name in models:
                top_models[model_name] = models[model_name]
        
        # Tune models
        tuning_results = tune_multiple_models(
            top_models, 
            X_train_processed, y_train, 
            X_test_processed, y_test,
            method='grid'
        )
        
        # Get best models
        best_models = get_best_models(tuning_results)
        
        # Evaluate best models
        best_results = []
        
        for model_name, model in best_models.items():
            logger.info(f"Evaluating tuned {model_name}...")
            
            # Evaluate model
            metrics, _, y_test_pred = evaluate_model(
                model, X_train_processed, X_test_processed, y_train, y_test
            )
            
            # Add to results
            metrics['model_name'] = f"{model_name} (tuned)"
            best_results.append(metrics)
            
            # Print metrics
            print_metrics(metrics, model_name=f"{model_name} (tuned)")
            
            # Save model
            save_model(model, os.path.join(args.output, 'models', f'{model_name}_tuned.joblib'))
        
        # Create best results DataFrame
        best_results_df = pd.DataFrame(best_results)
        
        # Reorder columns
        cols = ['model_name'] + [col for col in best_results_df.columns if col != 'model_name']
        best_results_df = best_results_df[cols]
        
        # Save best results
        save_results(best_results_df, 'tuned_model_results.csv', args.output)
    
    # Select and save best model
    best_model_name = results_df.iloc[0]['model_name']
    best_model = models[best_model_name]
    
    logger.info(f"Best model: {best_model_name}")
    
    # Save best model
    model_path = save_model(best_model, os.path.join(args.output, 'models', f'{best_model_name}.joblib'))
    logger.info(f"Saved best model to {model_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(args.output, 'models', 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    # Calculate execution time
    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)