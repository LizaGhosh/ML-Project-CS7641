"""
AI Orchestrator for Data Science Pipeline
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import google.generativeai as genai
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
from data_loader import load_data, clean_column_names, check_column_consistency
from eda import (
    plot_distribution, 
    plot_correlation_matrix, 
    plot_target_by_category,
    identify_outliers,
    plot_scatter_matrix
)
from feature_engineering import (
    extract_numerical_from_string,
    create_interaction_features,
    create_polynomial_features,
    create_ratio_features
)
from data_preprocessing import (
    identify_column_types,
    handle_missing_values,
    encode_categorical_features,
    scale_numerical_features,
    prepare_train_test_set
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AIOrchestrator")

class DataScienceOrchestrator:
    """
    AI Orchestrator that coordinates data science workflow using Gemini API
    for decision making and existing functions for execution.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-pro", output_dir: str = "orchestrator_output"):
        """
        Initialize the orchestrator
        
        Parameters:
        -----------
        api_key : str
            Gemini API key
        model : str, default="gemini-pro"
            Gemini model to use
        output_dir : str, default="orchestrator_output"
            Directory for output files
        """
        self.setup_gemini(api_key, model)
        self.conversation_history = []
        self.results = {
            "dataset_info": {},
            "eda_results": {},
            "feature_engineering": {},
            "preprocessing": {},
            "modeling": {},
            "evaluation": {}
        }
        self.df = None
        self.target_column = None
        self.output_dir = output_dir
        
        # Create output directory and subdirectories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)
        logger.info(f"Created output directories in {self.output_dir}")
    
    def setup_gemini(self, api_key: str, model: str):
        """Set up Gemini API client"""
        genai.configure(api_key=api_key)
        
        # Configure the generative model
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 4096,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # Try the updated model names for Gemini
        try:
            # First attempt with current model name
            self.model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Test with a simple prompt
            _ = self.model.generate_content("Hello")
            
        except Exception as e:
            logger.warning(f"Couldn't use model '{model}', trying alternatives: {str(e)}")
            
            # Try alternative model names
            model_alternatives = [
                "gemini-1.5-pro",
                "gemini-1.0-pro",
                "models/gemini-pro",
                "models/gemini-1.5-pro"
            ]
            
            for alt_model in model_alternatives:
                try:
                    logger.info(f"Trying alternative model: {alt_model}")
                    self.model = genai.GenerativeModel(
                        model_name=alt_model,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    # Test with a simple prompt
                    _ = self.model.generate_content("Hello")
                    logger.info(f"Successfully connected using model: {alt_model}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed with model '{alt_model}': {str(e)}")
            else:
                # If we get here, none of the alternatives worked
                raise ValueError(f"Could not connect to any Gemini model. Please check your API key and available models.")
        
        self.chat = self.model.start_chat(history=[])
        logger.info(f"Initialized Gemini model: {model}")
            
    def query_gemini(self, prompt: str) -> str:
        """
        Query the Gemini model
        
        Parameters:
        -----------
        prompt : str
            Prompt to send to Gemini
            
        Returns:
        --------
        str
            Response from Gemini
        """
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "parts": [prompt]})
            
            # Get response - try both chat interface and direct generation
            try:
                # First try chat interface
                response = self.chat.send_message(prompt)
                response_text = response.text
            except AttributeError:
                # If chat interface fails, try direct generation
                response = self.model.generate_content(prompt)
                response_text = response.text
            
            # Add to conversation history
            self.conversation_history.append({"role": "model", "parts": [response_text]})
            
            return response_text
        
        except Exception as e:
            logger.error(f"Error querying Gemini: {e}")
            return f"Error: {str(e)}"

    def get_dataset_recommendation(self, filepath: str) -> Dict[str, Any]:
        """
        Get recommendations for dataset analysis
        
        Parameters:
        -----------
        filepath : str
            Path to the dataset
            
        Returns:
        --------
        dict
            Recommendations for dataset analysis
        """
        # First, load dataset summary
        if self.df is None:
            self.df = load_data(filepath)
            self.df = clean_column_names(self.df)
        
        # Get dataset summary
        dataset_summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": self.df.isnull().sum().to_dict(),
            "sample_data": self.df.head(5).to_dict()
        }
        
        # Update results
        self.results["dataset_info"] = {
            "filepath": filepath,
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": self.df.isnull().sum().sum()
        }
        
        # Query Gemini for recommendations
        prompt = f"""
        I have a dataset with the following characteristics:
        - Shape: {dataset_summary['shape']}
        - Columns: {dataset_summary['columns']}
        - Data types: {json.dumps(dataset_summary['dtypes'], indent=2)}
        - Missing values: {json.dumps(dataset_summary['missing_values'], indent=2)}
        
        Here's a sample of the data:
        {json.dumps(dataset_summary['sample_data'], indent=2)}
        
        Based on this information, please provide recommendations for:
        1. Which column should be the target variable for price prediction?
        2. Which columns need special preprocessing or feature engineering?
        3. Which columns might contain issues that need to be addressed?
        4. What types of EDA visualizations would be most informative?
        
        Format your response as a JSON object with the following keys without using any markdown formatting:
        {{"target_column": "recommended_target_column",
            "columns_for_preprocessing": ["col1", "col2", "..."],
            "columns_with_issues": ["col3", "col4", "..."],
            "recommended_visualizations": ["visualization1", "visualization2", "..."],
            "reasoning": "Your explanation here without any markdown formatting, asterisks, or special characters"
        }}
        
        Make sure your response can be directly parsed as valid JSON without any code blocks, markdown, or formatting.
        """
        
        response = self.query_gemini(prompt)
        
        try:
            # Extract JSON from response - handle possible code blocks or markdown
            if "```json" in response:
                json_start = response.find('{', response.find('```json'))
            else:
                json_start = response.find('{')
                
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            # Clean any markdown formatting that might be present in the JSON string
            # This is a failsafe in case the model doesn't follow instructions
            json_str = json_str.replace('**', '')
            json_str = json_str.replace('*', '')
            json_str = json_str.replace('\n\n', ' ')
            
            # Try to fix common JSON parsing issues
            try:
                recommendations = json.loads(json_str)
            except json.JSONDecodeError:
                # If still failing, try a more aggressive approach
                import re
                
                # Fix escape characters
                json_str = json_str.replace('\\n', '\\\\n')
                
                # Try to fix invalid line breaks inside strings
                pattern = r'("[^"]*)\n([^"]*")'
                json_str = re.sub(pattern, r'\1\\n\2', json_str)
                
                # Try again
                recommendations = json.loads(json_str)
            
            # Store target column
            if "target_column" in recommendations:
                self.target_column = recommendations["target_column"]
            
            logger.info(f"Got dataset recommendations: {recommendations}")
            return recommendations
        
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            logger.error(f"Response: {response}")
            
            # Last resort: try to manually extract key information
            try:
                # Extract information manually
                target_col = None
                if "target_column" in response:
                    target_match = re.search(r'"target_column":\s*"([^"]+)"', response)
                    if target_match:
                        target_col = target_match.group(1)
                
                # Build default recommendations with any extracted info
                default_recs = {
                    "target_column": target_col or self.df.columns[-1],  # Extracted or default
                    "columns_for_preprocessing": [],
                    "columns_with_issues": [],
                    "recommended_visualizations": ["distribution", "correlation", "scatter"],
                    "reasoning": "Manually extracted from Gemini response due to parsing errors."
                }
                
                return default_recs
                
            except Exception:
                # Return completely default recommendations
                return {
                    "target_column": self.df.columns[-1],  # Assume last column is target
                    "columns_for_preprocessing": [],
                    "columns_with_issues": [],
                    "recommended_visualizations": ["distribution", "correlation", "scatter"],
                    "reasoning": "Failed to parse Gemini response, using defaults."
                }

    def run_eda(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run exploratory data analysis based on recommendations
        
        Parameters:
        -----------
        recommendations : dict
            Recommendations from Gemini
            
        Returns:
        --------
        dict
            EDA results
        """
        eda_results = {}
        
        # Get target column
        target_column = recommendations.get("target_column", self.target_column)
        if target_column is None or target_column not in self.df.columns:
            # If no target column, try to guess based on column name
            potential_targets = [col for col in self.df.columns if 'price' in col.lower()]
            if potential_targets:
                target_column = potential_targets[0]
            else:
                target_column = self.df.columns[-1]  # Default to last column
        
        self.target_column = target_column
        logger.info(f"Target column: {target_column}")
        
        # Get column types
        column_types = identify_column_types(self.df)
        eda_results["column_types"] = {k: v for k, v in column_types.items()}
        
        # Run visualizations based on recommendations
        recommended_vis = recommendations.get("recommended_visualizations", [])
        
        # Create visualizations
        plots_created = []
        
        # 1. Distribution of target variable
        if "distribution" in recommended_vis or not recommended_vis:
            plt.figure(figsize=(10, 6))
            plot_distribution(self.df, target_column)
            plt.savefig(os.path.join(self.output_dir, "plots", "target_distribution.png"))
            plt.close()
            plots_created.append("target_distribution.png")
            
            # Log-transformed distribution if target is skewed
            if self.df[target_column].skew() > 1:
                plt.figure(figsize=(10, 6))
                plot_distribution(self.df, target_column, log_transform=True)
                plt.savefig(os.path.join(self.output_dir, "plots", "target_log_distribution.png"))
                plt.close()
                plots_created.append("target_log_distribution.png")
        
        # 2. Correlation matrix
        if "correlation" in recommended_vis or not recommended_vis:
            plt.figure(figsize=(12, 10))
            plot_correlation_matrix(self.df, target_col=target_column)
            plt.savefig(os.path.join(self.output_dir, "plots", "correlation_matrix.png"))
            plt.close()
            plots_created.append("correlation_matrix.png")
        
        # 3. Target by categorical variables
        if "categorical" in recommended_vis or not recommended_vis:
            for col in column_types.get("categorical", [])[:5]:  # Limit to 5 categories
                if self.df[col].nunique() < 15:  # Avoid too many categories
                    plt.figure(figsize=(12, 6))
                    plot_target_by_category(self.df, col, target_column)
                    plt.savefig(os.path.join(self.output_dir, "plots", f"{col}_vs_target.png"))
                    plt.close()
                    plots_created.append(f"{col}_vs_target.png")
        
        # 4. Pairplot/Scatter matrix for key numerical features
        if "scatter" in recommended_vis or not recommended_vis:
            # Find top correlated features with target
            numeric_cols = column_types.get("numeric", [])
            if numeric_cols and target_column in self.df.columns:
                corr_with_target = self.df[numeric_cols].corrwith(self.df[target_column]).abs()
                top_features = corr_with_target.sort_values(ascending=False).head(5).index.tolist()
                
                if top_features:
                    plot_features = top_features + [target_column] if target_column not in top_features else top_features
                    plt.figure(figsize=(15, 15))
                    plot_scatter_matrix(self.df, columns=plot_features)
                    plt.savefig(os.path.join(self.output_dir, "plots", "scatter_matrix.png"))
                    plt.close()
                    plots_created.append("scatter_matrix.png")
        
        # 5. Identify outliers
        if "outliers" in recommended_vis or not recommended_vis:
            outliers = identify_outliers(self.df, column_types.get("numeric", []) + [target_column])
            eda_results["outliers"] = outliers
        
        eda_results["plots_created"] = plots_created
        eda_results["target_column"] = target_column
        
        # Update overall results
        self.results["eda_results"] = eda_results
        
        return eda_results
    
    def get_feature_engineering_plan(self, eda_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommendations for feature engineering
        
        Parameters:
        -----------
        eda_results : dict
            Results from EDA
            
        Returns:
        --------
        dict
            Feature engineering plan
        """
        # Summarize EDA results for Gemini
        column_types = eda_results.get("column_types", {})
        numeric_columns = column_types.get("numeric", [])
        categorical_columns = column_types.get("categorical", [])
        text_columns = column_types.get("text", [])
        datetime_columns = column_types.get("datetime", [])
        
        correlation_info = ""
        if numeric_columns and self.target_column:
            # Get top correlations with target
            corr_with_target = self.df[numeric_columns].corrwith(self.df[self.target_column]).abs()
            top_correlations = corr_with_target.sort_values(ascending=False).head(10)
            correlation_info = f"Top correlations with {self.target_column}:\n"
            for col, corr in top_correlations.items():
                correlation_info += f"- {col}: {corr:.4f}\n"
        
        outlier_info = ""
        if "outliers" in eda_results:
            outlier_info = "Columns with significant outliers:\n"
            for col, info in eda_results["outliers"].items():
                if info["count"] > 0:
                    outlier_info += f"- {col}: {info['count']} outliers\n"
        
        # Query Gemini for feature engineering recommendations
        prompt = f"""
        Based on the exploratory data analysis of my dataset, I need recommendations for feature engineering.
        
        Dataset information:
        - Target column: {self.target_column}
        - Numeric columns: {numeric_columns}
        - Categorical columns: {categorical_columns}
        - Text columns: {text_columns}
        - Datetime columns: {datetime_columns}
        
        {correlation_info}
        
        {outlier_info}
        
        Based on this information, please recommend feature engineering steps that would improve a price prediction model.
        Consider the following types of feature engineering:
        1. Interaction features (which pairs of features to combine)
        2. Polynomial features (which features to create polynomials for, and what degree)
        3. Ratio features (which features to create ratios between)
        4. Extracting information from text columns
        5. Creating binned features
        6. Any other transformations that might help
        
        Format your response as a JSON object with the following keys:
        {{
            "interaction_features": [["col1", "col2"], ["col3", "col4"]],
            "polynomial_features": {{"columns": ["col1", "col2"], "degree": 2}},
            "ratio_features": {{"numerators": ["col1", "col2"], "denominators": ["col3", "col4"]}},
            "text_extraction": [{{"column": "col1", "patterns": {{"feature_name": "regex_pattern"}}}}],
            "binning": [{{"column": "col1", "n_bins": 5, "strategy": "quantile"}}],
            "other_transformations": ["description1", "description2"],
            "reasoning": "Your explanation here without any markdown formatting"
        }}
        
        Make sure your response can be directly parsed as valid JSON without any code blocks, markdown, or formatting.
        """
        
        response = self.query_gemini(prompt)
        
        try:
            # Extract JSON from response - handle possible code blocks or markdown
            if "```json" in response:
                json_start = response.find('{', response.find('```json'))
            else:
                json_start = response.find('{')
                
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            # Clean any markdown formatting that might be present in the JSON string
            json_str = json_str.replace('**', '')
            json_str = json_str.replace('*', '')
            json_str = json_str.replace('\n\n', ' ')
            
            # Try to fix common JSON parsing issues
            try:
                fe_plan = json.loads(json_str)
            except json.JSONDecodeError:
                # If still failing, try a more aggressive approach
                import re
                
                # Fix escape characters
                json_str = json_str.replace('\\n', '\\\\n')
                
                # Try to fix invalid line breaks inside strings
                pattern = r'("[^"]*)\n([^"]*")'
                json_str = re.sub(pattern, r'\1\\n\2', json_str)
                
                # Try again
                fe_plan = json.loads(json_str)
            
            logger.info(f"Got feature engineering plan: {fe_plan}")
            return fe_plan
            
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            logger.error(f"Response: {response}")
            # Return default plan
            return {
                "interaction_features": [],
                "polynomial_features": {"columns": [], "degree": 2},
                "ratio_features": {"numerators": [], "denominators": []},
                "text_extraction": [],
                "binning": [],
                "other_transformations": [],
                "reasoning": "Failed to parse Gemini response, using defaults."
            }

    def apply_feature_engineering(self, fe_plan: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply feature engineering based on the plan
        
        Parameters:
        -----------
        fe_plan : dict
            Feature engineering plan
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features
        """
        df_engineered = self.df.copy()
        features_added = []
        
        # Create output directories if they don't exist
        os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)
        
        # [Rest of the function remains the same...]
        
        # Update results
        self.results["feature_engineering"] = {
            "features_added": features_added,
            "total_features": len(df_engineered.columns),
            "plan_applied": fe_plan
        }
        
        # Save engineered dataframe
        output_path = os.path.join(self.output_dir, "data", "engineered_features.csv")
        logger.info(f"Saving engineered features to {output_path}")
        df_engineered.to_csv(output_path, index=False)
        
        return df_engineered    

    def get_preprocessing_plan(self, df_engineered: pd.DataFrame) -> Dict[str, Any]:
        """
        Get recommendations for preprocessing
        
        Parameters:
        -----------
        df_engineered : pandas.DataFrame
            DataFrame with engineered features
            
        Returns:
        --------
        dict
            Preprocessing plan
        """
        # Get column information
        column_types = identify_column_types(df_engineered)
        missing_values = df_engineered.isnull().sum().to_dict()
        columns_with_missing = [col for col, count in missing_values.items() if count > 0]
        
        # Check data types and issues
        numeric_columns = column_types.get("numeric", [])
        categorical_columns = column_types.get("categorical", [])
        
        # Query Gemini for preprocessing recommendations
        prompt = f"""
        I need a preprocessing plan for my dataset with engineered features.
        
        Dataset information:
        - Target column: {self.target_column}
        - Total features: {len(df_engineered.columns)}
        - Numeric columns: {numeric_columns}
        - Categorical columns: {categorical_columns}
        - Columns with missing values: {columns_with_missing}
        
        Based on this information, please recommend a preprocessing strategy that includes:
        1. How to handle missing values in each column type
        2. How to encode categorical variables
        3. How to scale numerical features
        4. How to handle outliers
        5. Any other preprocessing steps that might be necessary
        
        Format your response as a JSON object with the following keys:
        {{
            "missing_values": {{"strategy": "auto", "numeric_strategy": "mean", "categorical_strategy": "most_frequent"}},
            "categorical_encoding": {{"method": "onehot", "drop_first": true}},
            "numerical_scaling": {{"method": "standard"}},
            "outlier_handling": {{"method": "clip", "threshold": 3}},
            "other_steps": ["step1", "step2", ...],
            "reasoning": "Your explanation here"
        }}
        """
        
        response = self.query_gemini(prompt)
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            preprocessing_plan = json.loads(json_str)
            
            logger.info(f"Got preprocessing plan: {preprocessing_plan}")
            return preprocessing_plan
        
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            logger.error(f"Response: {response}")
            # Return default plan
            return {
                "missing_values": {"strategy": "auto", "numeric_strategy": "mean", "categorical_strategy": "most_frequent"},
                "categorical_encoding": {"method": "auto", "drop_first": True},
                "numerical_scaling": {"method": "standard"},
                "outlier_handling": {"method": "clip", "threshold": 3},
                "other_steps": [],
                "reasoning": "Failed to parse Gemini response, using defaults."
            }
    
    def apply_preprocessing(self, df_engineered: pd.DataFrame, preprocessing_plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply preprocessing based on the plan
        
        Parameters:
        -----------
        df_engineered : pandas.DataFrame
            DataFrame with engineered features
        preprocessing_plan : dict
            Preprocessing plan
            
        Returns:
        --------
        tuple
            (Preprocessed X_train, X_test, y_train, y_test, preprocessing details)
        """
        # Handle missing values
        missing_values_config = preprocessing_plan.get("missing_values", {})
        strategy = missing_values_config.get("strategy", "auto")
        numeric_strategy = missing_values_config.get("numeric_strategy", "mean")
        categorical_strategy = missing_values_config.get("categorical_strategy", "most_frequent")
        
        logger.info(f"Handling missing values with strategy: {strategy}")
        df_clean = handle_missing_values(df_engineered, strategy=strategy, 
                                        numeric_strategy=numeric_strategy,
                                        categorical_strategy=categorical_strategy)
        
        # Handle outliers
        outlier_config = preprocessing_plan.get("outlier_handling", {})
        outlier_method = outlier_config.get("method", "clip")
        outlier_threshold = outlier_config.get("threshold", 3)
        
        if outlier_method != "none":
            logger.info(f"Handling outliers with method: {outlier_method}")
            column_types = identify_column_types(df_clean)
            numeric_cols = column_types.get("numeric", [])
            
            if self.target_column in numeric_cols:
                numeric_cols.remove(self.target_column)
                
            if numeric_cols:
                from data_preprocessing import handle_outliers
                df_clean = handle_outliers(df_clean, columns=numeric_cols, 
                                          method=outlier_method, threshold=outlier_threshold)
        
        # Split data
        logger.info(f"Splitting data with target column: {self.target_column}")
        X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_set(
            df_clean, self.target_column, test_size=0.2, random_state=42
        )
        
        # Apply preprocessing
        from data_preprocessing import apply_preprocessing
        X_train_processed, X_test_processed, feature_names = apply_preprocessing(
            X_train, X_test, preprocessor
        )
        
        # Update results
        self.results["preprocessing"] = {
            "preprocessing_plan": preprocessing_plan,
            "train_shape": X_train_processed.shape,
            "test_shape": X_test_processed.shape,
            "feature_names": feature_names
        }
        
        # Save preprocessing results
        preprocessed_data = {
            "X_train_processed": X_train_processed,
            "X_test_processed": X_test_processed,
            "y_train": y_train.values,
            "y_test": y_test.values,
            "feature_names": feature_names,
            "preprocessor": preprocessor
        }
        
        return preprocessed_data
    
    def run_pipeline(self, filepath: str) -> Dict[str, Any]:
        """
        Run the complete data science pipeline
        
        Parameters:
        -----------
        filepath : str
            Path to the dataset
            
        Returns:
        --------
        dict
            Results from the pipeline
        """
        logger.info(f"Starting AI-orchestrated data science pipeline for {filepath}")
        
        # Step 1: Get dataset recommendations
        logger.info("Step 1: Getting dataset recommendations")
        dataset_recommendations = self.get_dataset_recommendation(filepath)

        # print(dataset_recommendations)

        # sys.exit()

        
        # Step 2: Run exploratory data analysis
        logger.info("Step 2: Running exploratory data analysis")
        eda_results = self.run_eda(dataset_recommendations)
        
        # Step 3: Get feature engineering plan
        logger.info("Step 3: Getting feature engineering plan")
        feature_engineering_plan = self.get_feature_engineering_plan(eda_results)
        
        # Step 4: Apply feature engineering
        logger.info("Step 4: Applying feature engineering")
        df_engineered = self.apply_feature_engineering(feature_engineering_plan)
        
        # Step 5: Get preprocessing plan
        logger.info("Step 5: Getting preprocessing plan")
        preprocessing_plan = self.get_preprocessing_plan(df_engineered)
        
        # Step 6: Apply preprocessing
        logger.info("Step 6: Applying preprocessing")
        preprocessed_data = self.apply_preprocessing(df_engineered, preprocessing_plan)
        
        logger.info("Pipeline completed successfully")
        
        # Custom JSON encoder for NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, (set, tuple)):
                    return list(obj)
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save final results
        with open(os.path.join(self.output_dir, "pipeline_results.json"), "w") as f:
            # Convert numpy arrays and other non-serializable objects to lists
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {k: v for k, v in value.items() 
                                            if not isinstance(v, (np.ndarray, pd.DataFrame))}
                else:
                    serializable_results[key] = value
            
            # Use the custom encoder when dumping to JSON
            json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
        
        return self.results

# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-orchestrated Data Science Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input data file")
    parser.add_argument("--api_key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--output", type=str, default="orchestrator_output", help="Output directory")
    
    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = DataScienceOrchestrator(api_key=args.api_key)
    orchestrator.output_dir = args.output
    
    # Run pipeline
    results = orchestrator.run_pipeline(args.input)
    
    print(f"Pipeline completed. Results saved to {args.output}/pipeline_results.json")
    print(f"Visualizations saved to {args.output}/plots/")
    print(f"Processed data saved to {args.output}/data/")