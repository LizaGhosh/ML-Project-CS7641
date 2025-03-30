"""
Script to make predictions with trained models
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib

# Import utility modules
from data_loader import load_data, clean_column_names

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Price Prediction')
    
    parser.add_argument('--input', type=str, required=True, help='Path to input data file')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model file')
    parser.add_argument('--preprocessor', type=str, required=True, help='Path to saved preprocessor file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to output file')
    parser.add_argument('--include_features', action='store_true', help='Include features in output')
    
    return parser.parse_args()

def load_model_and_preprocessor(model_path, preprocessor_path):
    """Load model and preprocessor from files"""
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
        
        return model, preprocessor
    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {e}")
        return None, None

def preprocess_data(df, preprocessor):
    """Preprocess data using saved preprocessor"""
    try:
        X_processed = preprocessor.transform(df)
        logger.info(f"Preprocessed data with shape: {X_processed.shape}")
        return X_processed
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None

def make_predictions(model, X):
    """Make predictions using trained model"""
    try:
        predictions = model.predict(X)
        logger.info(f"Made {len(predictions)} predictions")
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None

def save_predictions(df, predictions, output_path, include_features=False):
    """Save predictions to CSV file"""
    try:
        # Create results DataFrame
        results_df = pd.DataFrame({'predicted_price': predictions})
        
        # Include original features if requested
        if include_features:
            results_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        return False

def run_prediction(args):
    """Run the prediction pipeline"""
    # Load and clean data
    logger.info("Loading data...")
    df = load_data(args.input)
    df = clean_column_names(df)
    
    # Print basic information
    logger.info(f"Dataset shape: {df.shape}")
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor(args.model, args.preprocessor)
    if model is None or preprocessor is None:
        return
    
    # Preprocess data
    X_processed = preprocess_data(df, preprocessor)
    if X_processed is None:
        return
    
    # Make predictions
    predictions = make_predictions(model, X_processed)
    if predictions is None:
        return
    
    # Save predictions
    success = save_predictions(df, predictions, args.output, args.include_features)
    if success:
        logger.info("Prediction completed successfully")

if __name__ == "__main__":
    args = parse_arguments()
    run_prediction(args)