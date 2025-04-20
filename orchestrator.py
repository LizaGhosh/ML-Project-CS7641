"""
AI Orchestrator for Data Science Pipeline
"""

import os
import re
import sys
import json
import time
import pprint 
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import google.generativeai as genai
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def setup_gemini(self, api_key: str, model: str = "gemini-1.5-pro-latest"):
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
            
            # Try alternative model names from the available list
            model_alternatives = [
                "models/gemini-1.5-pro-latest",  # Most recent stable version
                "models/gemini-1.5-pro",         # Alternative name
                "models/gemini-1.5-flash-latest", # Faster model
                "models/gemini-1.0-pro-vision-latest",  # If you need vision capabilities
                "models/gemini-2.5-pro-preview-03-25",  # Preview of next version
                "models/gemini-1.5-pro-001",     # Specific version
                "models/gemini-1.5-pro-002"      # Another specific version
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
                raise ValueError(
                    f"Could not connect to any Gemini model. Please check your API key and available models.\n"
                    f"Available models: {genai.list_models()}\n"
                    f"Last error: {str(e)}"
                )
        
        self.chat = self.model.start_chat(history=[])
        logger.info(f"Initialized Gemini model: {self.model._model_name}")


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

    def plotty(self, df, plot_type, x=None, y=None, hue=None, title=None, figsize=(10, 6), **kwargs):
        """
        Generic plotting function that can create various types of plots
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        plot_type : str
            Type of plot to create (histogram, scatter, box, bar, countplot, heatmap, etc.)
        x : str, default=None
            Column for x-axis
        y : str, default=None
            Column for y-axis
        hue : str, default=None
            Column for color encoding
        title : str, default=None
            Plot title
        figsize : tuple, default=(10, 6)
            Figure size
        **kwargs : dict
            Additional parameters for specific plot types
            
        Returns:
        --------
        str
            Path to saved plot file
        """
        
        # Create a safe filename from the plot details
        def create_safe_filename(x=None, y=None, plot_type=None):
            components = []
            if x:
                components.append(str(x))
            if y:
                components.append(str(y))
            if plot_type:
                components.append(str(plot_type))
            
            # Create filename with underscores and remove special characters
            filename = "_".join(components)
            filename = re.sub(r'[^\w\-_]', '_', filename)
            return f"{filename}.png"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        try:
            # Create appropriate plot based on type
            if plot_type.lower() == 'histogram':
                if x:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(df[x]):
                        sns.histplot(data=df, x=x, kde=kwargs.get('kde', True))
                    else:
                        logger.warning(f"Column '{x}' is not numeric. Using countplot instead of histogram.")
                        sns.countplot(data=df, x=x)
                else:
                    raise ValueError("x parameter is required for histogram")
                    
            elif plot_type.lower() == 'scatter':
                if x and y:
                    # Check if both columns are numeric
                    if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                        sns.scatterplot(data=df, x=x, y=y, hue=hue)
                    else:
                        non_numeric = []
                        if not pd.api.types.is_numeric_dtype(df[x]):
                            non_numeric.append(x)
                        if not pd.api.types.is_numeric_dtype(df[y]):
                            non_numeric.append(y)
                        raise ValueError(f"Scatter plot requires numeric columns. Non-numeric columns: {non_numeric}")
                else:
                    raise ValueError("Both x and y parameters are required for scatter plot")
                    
            elif plot_type.lower() == 'box' or plot_type.lower() == 'boxplot':
                if x and y:
                    # For box plots, y should be numeric, x can be categorical
                    if pd.api.types.is_numeric_dtype(df[y]):
                        sns.boxplot(data=df, x=x, y=y, hue=hue)
                    else:
                        raise ValueError(f"Column '{y}' must be numeric for box plot")
                elif y:
                    if pd.api.types.is_numeric_dtype(df[y]):
                        sns.boxplot(data=df, y=y)
                    else:
                        raise ValueError(f"Column '{y}' must be numeric for box plot")
                else:
                    raise ValueError("At least y parameter is required for box plot")
                    
            elif plot_type.lower() == 'bar' or plot_type.lower() == 'barplot':
                if x and y:
                    # For bar plots, y should generally be numeric
                    if pd.api.types.is_numeric_dtype(df[y]):
                        sns.barplot(data=df, x=x, y=y, hue=hue)
                    else:
                        logger.warning(f"Column '{y}' is not numeric for bar plot. Results may be unexpected.")
                        sns.barplot(data=df, x=x, y=y, hue=hue)
                else:
                    raise ValueError("Both x and y parameters are required for bar plot")
                    
            elif plot_type.lower() == 'count' or plot_type.lower() == 'countplot':
                if x:
                    sns.countplot(data=df, x=x, hue=hue)
                elif y:
                    sns.countplot(data=df, y=y, hue=hue)
                else:
                    raise ValueError("Either x or y parameter is required for count plot")
                    
            elif plot_type.lower() == 'heatmap' or plot_type.lower() == 'correlation':
                # Select only numeric columns for correlation/heatmap
                numeric_df = df.select_dtypes(include=['number'])
                
                if numeric_df.empty:
                    raise ValueError("No numeric columns available for correlation/heatmap")
                    
                if kwargs.get('correlation', True):
                    if len(numeric_df.columns) < 2:
                        raise ValueError("Need at least 2 numeric columns for correlation heatmap")
                        
                    # Create correlation matrix with numeric data only
                    corr = numeric_df.corr()
                    sns.heatmap(corr, annot=kwargs.get('annot', True), cmap=kwargs.get('cmap', 'coolwarm'))
                    
                    # Update title to indicate numeric columns only
                    if title:
                        title = f"{title} (numeric columns only)"
                    else:
                        title = "Correlation Matrix (numeric columns only)"
                else:
                    # If a specific heatmap data is provided
                    if x:
                        if isinstance(x, list):
                            # Filter for numeric columns only
                            valid_columns = [col for col in x if col in numeric_df.columns]
                            if not valid_columns:
                                raise ValueError(f"None of the specified columns {x} are numeric")
                            sns.heatmap(numeric_df[valid_columns], annot=kwargs.get('annot', True), 
                                    cmap=kwargs.get('cmap', 'coolwarm'))
                        elif x in numeric_df.columns:
                            sns.heatmap(numeric_df[[x]], annot=kwargs.get('annot', True), 
                                    cmap=kwargs.get('cmap', 'coolwarm'))
                        else:
                            raise ValueError(f"Column '{x}' is not numeric and cannot be used in a heatmap")
                    else:
                        raise ValueError("Either correlation=True or x parameter with columns is required for heatmap")
                        
            elif plot_type.lower() == 'distribution':
                if x:
                    if pd.api.types.is_numeric_dtype(df[x]):
                        if kwargs.get('log_transform', False) and (df[x] > 0).all():
                            sns.histplot(np.log(df[x]), kde=kwargs.get('kde', True))
                            plt.xlabel(f'Log({x})')
                        else:
                            sns.histplot(df[x], kde=kwargs.get('kde', True))
                    else:
                        logger.warning(f"Column '{x}' is not numeric. Using countplot instead of distribution.")
                        sns.countplot(data=df, x=x)
                else:
                    raise ValueError("x parameter is required for distribution plot")
                    
            elif plot_type.lower() == 'violin' or plot_type.lower() == 'violinplot':
                if x and y:
                    # For violin plots, y should be numeric
                    if pd.api.types.is_numeric_dtype(df[y]):
                        sns.violinplot(data=df, x=x, y=y, hue=hue)
                    else:
                        raise ValueError(f"Column '{y}' must be numeric for violin plot")
                else:
                    raise ValueError("Both x and y parameters are required for violin plot")
                    
            elif plot_type.lower() == 'pair' or plot_type.lower() == 'pairplot':
                # Close the current figure for pairplot
                plt.close()
                
                # Filter for numeric columns
                if x:  # x is used as a list of columns for pairplot
                    if isinstance(x, str):
                        cols = [x]
                    else:
                        cols = x
                        
                    # Check which columns are numeric
                    valid_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
                    if not valid_cols:
                        raise ValueError(f"No numeric columns found in {cols} for pairplot")
                    
                    g = sns.pairplot(df[valid_cols], hue=hue)
                    
                    # If some columns were excluded, add to title
                    if len(valid_cols) < len(cols):
                        excluded = set(cols) - set(valid_cols)
                        if title:
                            title = f"{title} (excluded non-numeric: {', '.join(excluded)})"
                        else:
                            title = f"Pairplot (excluded non-numeric: {', '.join(excluded)})"
                        g.fig.suptitle(title)
                else:
                    # Use all numeric columns
                    numeric_df = df.select_dtypes(include=['number'])
                    if numeric_df.empty:
                        raise ValueError("No numeric columns available for pairplot")
                    
                    g = sns.pairplot(numeric_df, hue=hue)
                    
                    if title:
                        g.fig.suptitle(title)
                
                # Create filename
                filename = kwargs.get('filename', create_safe_filename(x, y, plot_type))
                filepath = os.path.join(self.output_dir, "plots", filename)
                
                # Save pairplot
                g.savefig(filepath)
                return filename
                    
            elif plot_type.lower() == 'line' or plot_type.lower() == 'lineplot':
                if x and y:
                    # For line plots, y should be numeric
                    if pd.api.types.is_numeric_dtype(df[y]):
                        sns.lineplot(data=df, x=x, y=y, hue=hue)
                    else:
                        raise ValueError(f"Column '{y}' must be numeric for line plot")
                else:
                    raise ValueError("Both x and y parameters are required for line plot")
            
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
                
            # Set title if provided
            if title:
                plt.title(title)
                
            # Set additional plot properties
            plt.tight_layout()
            
            # Create filename
            filename = kwargs.get('filename', create_safe_filename(x, y, plot_type))
            filepath = os.path.join(self.output_dir, "plots", filename)
            
            # Save plot
            plt.savefig(filepath)
            plt.close()
            
            return filename
            
        except Exception as e:
            plt.close()
            logger.error(f"Error creating {plot_type} plot: {str(e)}")
            
            # Create a simple error plot to indicate the problem
            plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, f"Could not create {plot_type} plot:\n{str(e)}", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
            
            # Save error plot
            filename = create_safe_filename(x, y, f"error_{plot_type}")
            filepath = os.path.join(self.output_dir, "plots", filename)
            plt.savefig(filepath)
            plt.close()
            
            return filename

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
            error_str = str(e)
            logger.error(f"Error querying Gemini: {error_str}")
            
            # Check if this is a rate limit error
            if "429" in error_str and "exceeded your current quota" in error_str:
                # Try to extract the retry delay from the error message
                import re
                import time
                
                retry_seconds = 30  # Default retry delay if we can't parse it
                
                # Try to find the retry delay in the error message
                match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                if match:
                    retry_seconds = int(match.group(1))
                    logger.info(f"Extracted retry delay of {retry_seconds} seconds")
                
                logger.info(f"Rate limit exceeded. Waiting for {retry_seconds} seconds before retrying...")
                time.sleep(retry_seconds)
                
                # Retry the request after waiting
                try:
                    # Try again with chat interface
                    response = self.chat.send_message(prompt)
                    response_text = response.text
                    
                    # Add successful retry to conversation history
                    self.conversation_history.append({"role": "model", "parts": [response_text]})
                    
                    return response_text
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {retry_error}")
                    return f"Error: {str(retry_error)} (after waiting {retry_seconds} seconds)"
            
            return f"Error: {str(e)}"

    # def generate_visualization_from_recommendation(self, vis_rec, df_columns):
    #     """
    #     Use Gemini to interpret a visualization recommendation and generate plotting code
        
    #     Parameters:
    #     -----------
    #     vis_rec : str
    #         Visualization recommendation text
    #     df_columns : list
    #         List of available dataframe columns
            
    #     Returns:
    #     --------
    #     dict
    #         Parameters for plotty function
    #     """
    #     prompt = f"""
    #     I need to create a visualization based on this recommendation: "{vis_rec}"
        
    #     The available columns in my dataframe are: {df_columns}
        
    #     I have a function called plotty with this signature:
        
    #     ```python
    #     plotty(df, plot_type, x=None, y=None, hue=None, title=None, figsize=(10, 6), **kwargs)
    #     ```
        
    #     Given the recommendation and available columns, what parameters should I pass to plotty?
    #     Return your answer as a valid JSON dictionary (use JSON format, not Python - so use [] for arrays instead of tuples).
        
    #     Example output format:
    #     {{
    #         "plot_type": "scatter",
    #         "x": "column_name", 
    #         "y": "another_column",
    #         "title": "Appropriate Title",
    #         "figsize": [10, 6],
    #         "other_param1": value1,
    #         "other_param2": value2
    #     }}
        
    #     No need for code block formatting, just return the JSON dictionary directly.
    #     Please match column names exactly as they appear in the list provided, and ensure the plot_type is one of: 
    #     histogram, scatter, box, bar, countplot, heatmap, correlation, distribution, violin, pair, line
    #     """
        
    #     # This call now has rate limit handling built-in from the updated query_gemini method
    #     response = self.query_gemini(prompt)
        
    #     try:
    #         # Extract the JSON dictionary from the response
    #         import json
    #         import re
            
    #         # Strip code block formatting if present
    #         cleaned_response = response.replace('```python', '').replace('```json', '').replace('```', '')
            
    #         # Find dictionary-like content
    #         dict_match = re.search(r'\{[\s\S]*\}', cleaned_response)
    #         if dict_match:
    #             dict_str = dict_match.group(0)
                
    #             # Convert Python tuples to JSON arrays
    #             dict_str = re.sub(r'\((\s*\d+\s*,\s*\d+\s*)\)', r'[\1]', dict_str)
                
    #             # Fix any missing quotes around keys
    #             dict_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', dict_str)
                
    #             # Fix trailing commas
    #             dict_str = re.sub(r',(\s*[\]}])', r'\1', dict_str)
                
    #             # Special handling for string values that might not be quoted
    #             dict_str = re.sub(r':\s*(\w+)(?=\s*[,}])', r': "\1"', dict_str)
                
    #             # Try to parse the JSON
    #             try:
    #                 plot_params = json.loads(dict_str)
    #                 logger.info(f"Successfully parsed Gemini response into params: {plot_params}")
    #                 return plot_params
    #             except json.JSONDecodeError as e:
    #                 logger.error(f"JSON decode error: {e} in string: {dict_str}")
    #                 # Fall back to manual extraction below
            
    #         # If JSON parsing failed, extract key information manually
    #         logger.warning(f"Could not parse JSON from Gemini response. Extracting parameters manually.")
            
    #         plot_params = {"title": vis_rec}
            
    #         # Manually identify plot type
    #         if "scatter" in vis_rec.lower():
    #             plot_params["plot_type"] = "scatter"
    #         elif "box" in vis_rec.lower():
    #             plot_params["plot_type"] = "box"
    #         elif "histogram" in vis_rec.lower() or "distribution" in vis_rec.lower():
    #             plot_params["plot_type"] = "histogram"
    #         elif "count" in vis_rec.lower():
    #             plot_params["plot_type"] = "countplot"
    #         elif "correlation" in vis_rec.lower() or "heatmap" in vis_rec.lower():
    #             plot_params["plot_type"] = "heatmap"
    #             plot_params["correlation"] = True
    #         else:
    #             plot_params["plot_type"] = "scatter"  # Default
            
    #         # Extract column references "x vs y" pattern
    #         vs_match = re.search(r'(\w+)\s+vs\s+(\w+)', vis_rec.lower())
    #         if vs_match:
    #             x_guess, y_guess = vs_match.groups()
    #             # Match with actual column names (case insensitive)
    #             for col in df_columns:
    #                 if col.lower() == x_guess:
    #                     plot_params["x"] = col
    #                 if col.lower() == y_guess:
    #                     plot_params["y"] = col
            
    #         # Extract "by" pattern (for boxplots, etc.)
    #         by_match = re.search(r'(\w+)\s+by\s+(\w+)', vis_rec.lower())
    #         if by_match:
    #             y_guess, x_guess = by_match.groups()
    #             # Match with actual column names (case insensitive)
    #             for col in df_columns:
    #                 if col.lower() == x_guess:
    #                     plot_params["x"] = col
    #                 if col.lower() == y_guess:
    #                     plot_params["y"] = col
            
    #         # If we still don't have x/y, look for column names
    #         if "x" not in plot_params or "y" not in plot_params:
    #             for col in df_columns:
    #                 if col.lower() in vis_rec.lower():
    #                     if "x" not in plot_params:
    #                         plot_params["x"] = col
    #                     elif "y" not in plot_params:
    #                         plot_params["y"] = col
            
    #         logger.info(f"Manually extracted plot parameters: {plot_params}")
    #         return plot_params
        
    #     except Exception as e:
    #         logger.error(f"Error processing visualization recommendation: {str(e)}")
    #         # Return minimal params that won't cause errors
    #         return {
    #             "plot_type": "histogram",
    #             "x": df_columns[0] if df_columns else None,
    #             "title": vis_rec
    #         }

    def generate_visualization_from_recommendation(self, vis_rec, df_columns):
        """
        Use Gemini to interpret a visualization recommendation and generate plotting code
        
        Parameters:
        -----------
        vis_rec : str
            Visualization recommendation text
        df_columns : list
            List of available dataframe columns
            
        Returns:
        --------
        dict
            Parameters for plotty function
        """
        prompt = f"""
        I need to create a visualization based on this recommendation: "{vis_rec}"
        
        The available columns in my dataframe are: {df_columns}
        
        I have a function called plotty with this signature:
        
        ```python
        plotty(df, plot_type, x=None, y=None, hue=None, title=None, figsize=(10, 6), **kwargs)
        ```
        
        Given the recommendation and available columns, what parameters should I pass to plotty?
        
        IMPORTANT: Return your answer as a SINGLE valid JSON dictionary (not multiple dictionaries). 
        If you need to suggest multiple plots, include them as an array within a single JSON object.
        
        Example for a single plot:
        {{
            "plot_type": "scatter",
            "x": "column_name", 
            "y": "another_column",
            "title": "Appropriate Title",
            "figsize": [10, 6]
        }}
        
        Example for multiple plots:
        {{
            "plots": [
                {{
                    "plot_type": "scatter",
                    "x": "column_name",
                    "y": "another_column",
                    "title": "First Plot Title",
                    "figsize": [10, 6]
                }},
                {{
                    "plot_type": "histogram",
                    "x": "column_name",
                    "title": "Second Plot Title",
                    "figsize": [10, 6]
                }}
            ]
        }}
        
        No need for code block formatting, just return the JSON dictionary directly.
        Please match column names exactly as they appear in the list provided, and ensure the plot_type is one of: 
        histogram, scatter, box, bar, countplot, heatmap, correlation, distribution, violin, pair, line
        """
        
        # This call now has rate limit handling built-in from the updated query_gemini method
        response = self.query_gemini(prompt)
        
        try:
            # Extract and parse JSON from the response
            import json
            import re
            
            # Clean up the response
            cleaned_response = response.replace('```python', '').replace('```json', '').replace('```', '')
            
            # Find all JSON-like patterns in the response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, cleaned_response)
            
            if json_matches:
                # Try to parse each match
                for json_str in json_matches:
                    try:
                        # Fix any missing quotes around keys
                        json_str = re.sub(r'(\w+)(?=\s*:)', r'"\1"', json_str)
                        
                        # Fix trailing commas
                        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
                        
                        # Format tuples to arrays
                        json_str = re.sub(r'\((\s*\d+\s*,\s*\d+\s*)\)', r'[\1]', json_str)
                        
                        # Fix single quotes to double quotes
                        json_str = json_str.replace("'", '"')
                        
                        # Try to parse the JSON
                        plot_params = json.loads(json_str)
                        
                        # Check if we have a "plots" array for multiple plots
                        if "plots" in plot_params and isinstance(plot_params["plots"], list):
                            # Return the first plot configuration
                            logger.info(f"Found multiple plot configurations, using the first one")
                            return plot_params["plots"][0]
                        else:
                            # Return the single plot configuration
                            logger.info(f"Successfully parsed plot parameters: {plot_params}")
                            return plot_params
                            
                    except json.JSONDecodeError:
                        continue
                        
                # If we get here, none of the matches could be parsed directly
                # Try a more aggressive approach with the first match
                json_str = json_matches[0]
                logger.warning(f"Could not parse JSON directly, attempting cleanup")
                
                # More aggressive cleaning
                json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r',\s*}', '}', json_str)
                
                try:
                    plot_params = json.loads(json_str)
                    logger.info(f"Successfully parsed plot parameters after cleanup: {plot_params}")
                    return plot_params
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON after cleanup")
            
            # Manual extraction as fallback
            logger.warning(f"Could not parse JSON from Gemini response. Extracting parameters manually.")
            
            plot_params = {"title": vis_rec}
            
            # Determine plot type based on recommendation text
            if "scatter" in vis_rec.lower():
                plot_params["plot_type"] = "scatter"
            elif "box" in vis_rec.lower():
                plot_params["plot_type"] = "box"
            elif "histogram" in vis_rec.lower() or "distribution" in vis_rec.lower():
                plot_params["plot_type"] = "histogram"
            elif "count" in vis_rec.lower():
                plot_params["plot_type"] = "countplot"
            elif "correlation" in vis_rec.lower() or "heatmap" in vis_rec.lower():
                plot_params["plot_type"] = "heatmap"
                plot_params["correlation"] = True
            elif "bar" in vis_rec.lower():
                plot_params["plot_type"] = "bar"
            else:
                plot_params["plot_type"] = "scatter"  # Default
            
            # Extract column references from recommendation text
            # Extract "X vs Y" pattern
            vs_match = re.search(r'(\w+)\s+vs\.?\s+(\w+)', vis_rec.lower())
            if vs_match:
                x_guess, y_guess = vs_match.groups()
                # Match with actual column names (case insensitive)
                for col in df_columns:
                    if col.lower() == x_guess.lower():
                        plot_params["x"] = col
                    if col.lower() == y_guess.lower():
                        plot_params["y"] = col
            
            # Extract "Y by X" pattern (for boxplots, etc.)
            by_match = re.search(r'(\w+)\s+by\s+(\w+)', vis_rec.lower())
            if by_match:
                y_guess, x_guess = by_match.groups()
                # Match with actual column names (case insensitive)
                for col in df_columns:
                    if col.lower() == x_guess.lower():
                        plot_params["x"] = col
                    if col.lower() == y_guess.lower():
                        plot_params["y"] = col
            
            # If we still don't have x/y, look for column names in the text
            if "x" not in plot_params or "y" not in plot_params:
                for col in df_columns:
                    if col.lower() in vis_rec.lower():
                        if "x" not in plot_params:
                            plot_params["x"] = col
                        elif "y" not in plot_params:
                            plot_params["y"] = col
            
            logger.info(f"Manually extracted plot parameters: {plot_params}")
            return plot_params
        
        except Exception as e:
            logger.error(f"Error processing visualization recommendation: {str(e)}")
            # Return minimal params that won't cause errors
            return {
                "plot_type": "histogram",
                "x": df_columns[0] if df_columns else None,
                "title": vis_rec
            }
            
    def run_eda(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run exploratory data analysis based on AI recommendations using Gemini
        to dynamically interpret visualization requests
        
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
        
        # Get target column with case-insensitive matching
        target_column = recommendations.get("target_column")
        if not target_column:
            logger.warning("No target column specified in recommendations")
            return {}
        
        # Handle case sensitivity and column existence
        if target_column not in self.df.columns:
            # Try case-insensitive matching
            column_mapping = {col.lower(): col for col in self.df.columns}
            if target_column.lower() in column_mapping:
                target_column = column_mapping[target_column.lower()]
                logger.info(f"Found target column with different case: {target_column}")
            else:
                # Try to find a column with 'price' in the name
                potential_targets = [col for col in self.df.columns if 'price' in col.lower()]
                if potential_targets:
                    target_column = potential_targets[0]
                    logger.info(f"Using potential target column: {target_column}")
                else:
                    logger.warning("Target column could not be found in dataset")
                    return {}
        
        self.target_column = target_column
        logger.info(f"Target column: {target_column}")
        
        # Get column types
        column_types = identify_column_types(self.df)
        eda_results["column_types"] = {k: v for k, v in column_types.items()}
        
        # Detect missing values
        missing_values = self.df.isnull().sum()
        missing_values_percent = (missing_values / len(self.df) * 100).round(2)
        missing_data = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage (%)': missing_values_percent
        }).sort_values('Missing Values', ascending=False)
        
        # Only keep columns with missing values
        missing_data = missing_data[missing_data['Missing Values'] > 0]
        
        # Add missing value information to results
        eda_results["missing_values"] = {
            "total_missing": missing_values.sum(),
            "missing_by_column": missing_data.to_dict() if not missing_data.empty else {},
            "columns_with_missing": list(missing_data.index) if not missing_data.empty else []
        }
        
        logger.info(f"Missing values detected: {missing_values.sum()} total")
        if not missing_data.empty:
            logger.info(f"Columns with missing values: {list(missing_data.index)}")
        
        # Visualize missing values if they exist
        if missing_values.sum() > 0:
            try:
                plot_params = {
                    "plot_type": "missing_values",
                    "title": "Missing Values by Column",
                    "figsize": (10, 6)
                }
                filename = self.plotty(self.df, **plot_params)
                if filename:
                    plots_created = ["missing_values.png"]
                    logger.info(f"Created missing values plot: {filename}")
            except Exception as e:
                logger.error(f"Error creating missing values plot: {str(e)}")
                plots_created = []
        else:
            plots_created = []
        
        # Process recommended visualizations using Gemini
        recommended_vis = recommendations.get("recommended_visualizations", [])
        
        # Get available columns for Gemini
        available_columns = list(self.df.columns)
        
        # Process each visualization recommendation with Gemini
        for vis_rec in recommended_vis:
            logger.info(f"Processing visualization recommendation: {vis_rec}")
            
            # Use Gemini to interpret the visualization recommendation
            plot_params = self.generate_visualization_from_recommendation(vis_rec, available_columns)
            
            if plot_params:
                logger.info(f"Generated plot parameters: {plot_params}")
                
                try:
                    # Create the plot using plotty
                    filename = self.plotty(self.df, **plot_params)
                    
                    if filename:
                        plots_created.append(filename)
                        logger.info(f"Created plot: {filename}")
                except Exception as e:
                    logger.error(f"Error creating plot for '{vis_rec}': {str(e)}")
        
        # If no plots were created, generate some default plots
        if not plots_created or len(plots_created) == 1 and "missing_values.png" in plots_created:
            logger.info("No visualization plots were successfully created. Generating default plots.")
            
            # Default plot 1: Target variable distribution
            try:
                plot_params = {
                    "plot_type": "histogram",
                    "x": target_column,
                    "title": f"Distribution of {target_column}",
                    "kde": True
                }
                filename = self.plotty(self.df, **plot_params)
                if filename:
                    plots_created.append(filename)
            except Exception as e:
                logger.error(f"Error creating default target distribution plot: {str(e)}")
            
            # Default plot 2: Correlation matrix
            try:
                plot_params = {
                    "plot_type": "heatmap",
                    "title": "Correlation Matrix",
                    "correlation": True,
                    "annot": True
                }
                filename = self.plotty(self.df, **plot_params)
                if filename:
                    plots_created.append(filename)
            except Exception as e:
                logger.error(f"Error creating default correlation matrix: {str(e)}")
        
        # Identify outliers
        try:
            outliers = identify_outliers(self.df, column_types.get("numeric", []) + [target_column])
            eda_results["outliers"] = outliers
        except Exception as e:
            logger.error(f"Error identifying outliers: {str(e)}")
        
        eda_results["plots_created"] = plots_created
        eda_results["target_column"] = target_column
        
        # Update overall results
        self.results["eda_results"] = eda_results
        
        # Return non-empty results to indicate success
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
        Apply feature engineering based on the plan, focusing on implementing
        recommendations from other_transformations using Gemini
        
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
        
        # Process transformations from other_transformations
        if "other_transformations" in fe_plan and fe_plan["other_transformations"]:
            for transformation in fe_plan["other_transformations"]:
                logger.info(f"Processing transformation: {transformation}")
                
                # Generate Python code for this transformation using Gemini
                prompt = f"""
                I need to implement the following feature engineering transformation on my dataframe:
                "{transformation}"
                
                My dataframe has columns: {list(df_engineered.columns)}
                
                Please write a short Python code snippet that implements this transformation.
                Use pandas and numpy operations. Assume the dataframe is called 'df'.
                Only output the code without any explanations or markdown formatting.
                """
                
                try:
                    # Query Gemini for implementation code
                    code_response = self.query_gemini(prompt)
                    
                    # Clean up response to extract just the code
                    code = code_response
                    # Remove potential markdown code blocks
                    if "```python" in code:
                        code = code.split("```python")[1]
                        if "```" in code:
                            code = code.split("```")[0]
                    elif "```" in code:
                        code = code.split("```")[1]
                        if "```" in code:
                            code = code.split("```")[0]
                    
                    # Log the code that will be executed
                    logger.info(f"Generated code for transformation:\n{code}")
                    
                    # Execute the code in a context with the dataframe
                    local_vars = {"df": df_engineered, "np": np, "pd": pd}
                    exec(code, globals(), local_vars)
                    
                    # Get the updated dataframe
                    df_engineered = local_vars["df"]
                    
                    # Identify new columns by comparing with original columns
                    original_cols = set(self.df.columns)
                    current_cols = set(df_engineered.columns)
                    new_cols = current_cols - original_cols
                    features_added.extend(list(new_cols))
                    
                    logger.info(f"Transformation applied successfully, added features: {new_cols}")
                    
                except Exception as e:
                    logger.error(f"Error applying transformation '{transformation}': {str(e)}")
        
        # Handle basic feature engineering operations that have consistent structure
        
        # 1. Create interaction features if specified and not empty
        if "interaction_features" in fe_plan and fe_plan["interaction_features"]:
            try:
                logger.info("Processing interaction features")
                # Generate code for interactions using Gemini
                cols_str = str(list(df_engineered.columns))
                interactions_str = str(fe_plan["interaction_features"])
                
                prompt = f"""
                I need to create interaction features between column pairs:
                {interactions_str}
                
                My dataframe has columns: {cols_str}
                
                Please write Python code that:
                1. Creates interaction features for valid column pairs
                2. Handles cases where columns might not exist
                3. Returns the updated dataframe
                
                Only provide the code without explanations or markdown formatting.
                Assume the dataframe is called 'df'.
                """
                
                code_response = self.query_gemini(prompt)
                code = code_response
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                
                logger.info(f"Generated code for interaction features:\n{code}")
                
                # Execute the code
                local_vars = {"df": df_engineered, "np": np, "pd": pd}
                exec(code, globals(), local_vars)
                df_engineered = local_vars["df"]
                
                # Identify new columns
                new_cols = set(df_engineered.columns) - set(self.df.columns) - set(features_added)
                features_added.extend(list(new_cols))
                logger.info(f"Added interaction features: {new_cols}")
                
            except Exception as e:
                logger.error(f"Error creating interaction features: {str(e)}")
        
        # 2. Create polynomial features if specified
        if "polynomial_features" in fe_plan and fe_plan["polynomial_features"]:
            try:
                logger.info("Processing polynomial features")
                poly_config = fe_plan["polynomial_features"]
                poly_config_str = str(poly_config)
                cols_str = str(list(df_engineered.columns))
                
                prompt = f"""
                I need to create polynomial features with this configuration:
                {poly_config_str}
                
                My dataframe has columns: {cols_str}
                
                Please write Python code that:
                1. Creates polynomial features for specified columns
                2. Handles cases where columns might not exist
                3. Uses the specified degree
                4. Returns the updated dataframe
                
                Only provide the code without explanations or markdown formatting.
                Assume the dataframe is called 'df'.
                """
                
                code_response = self.query_gemini(prompt)
                code = code_response
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                
                logger.info(f"Generated code for polynomial features:\n{code}")
                
                # Execute the code
                local_vars = {"df": df_engineered, "np": np, "pd": pd}
                exec(code, globals(), local_vars)
                df_engineered = local_vars["df"]
                
                # Identify new columns
                new_cols = set(df_engineered.columns) - set(self.df.columns) - set(features_added)
                features_added.extend(list(new_cols))
                logger.info(f"Added polynomial features: {new_cols}")
                
            except Exception as e:
                logger.error(f"Error creating polynomial features: {str(e)}")
        
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
        # Convert boolean columns to float before handling missing values
        bool_cols = df_engineered.select_dtypes(include='bool').columns
        if not bool_cols.empty:
            logger.info(f"Converting boolean columns to float: {list(bool_cols)}")
            df_engineered[bool_cols] = df_engineered[bool_cols].astype(float)
        
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

        if X_train_processed.shape[1] == 0:
            logger.error("No features remain after preprocessing. Check for overly aggressive filtering.")
            raise ValueError("Preprocessing removed all features. Cannot train models.")
        
        return preprocessed_data

    def get_modeling_recommendation(self, preprocessed_data: Dict[str, Any], target_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommendations for modeling based on the preprocessed data and target characteristics
        """
        X_train = preprocessed_data["X_train_processed"]
        
        
        # Query Gemini for modeling recommendations
        prompt = f"""I need to build predictive models for a dataset with these characteristics:
        
        - Target variable type: {target_info["type"]}
        - Number of unique target values: {target_info["unique_values"]}
        - Is classification problem: {target_info["is_classification"]}
        
        The preprocessed data has:
        - Training samples: {X_train.shape[0]}
        - Features: {X_train.shape[1]}
        
        Please provide recommendations in this exact JSON format (remove all markdown formatting):
        {{
            "model_types": ["type1", "type2"],
            "evaluation_metrics": ["metric1", "metric2"],
            "special_considerations": ["consideration1", "consideration2"],
            "hyperparameter_ranges": {{
                "model1": {{"param1": ["value1", "value2"]}},
                "model2": {{"param1": ["value1", "value2"]}}
            }},
            "reasoning": "Your explanation here"
        }}
        Return ONLY the JSON object with no additional text or formatting."""
        
        response = self.query_gemini(prompt)
        
        try:
            # First attempt to parse directly
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parse failed, attempting cleanup: {e}")
                
                if response.strip().startswith("```"):
                    response = re.sub(r"^```(?:json)?", "", response.strip())
                    response = re.sub(r"```$", "", response.strip())

                # Remove extra newlines or indentation
                json_str = response.strip()

                # Clean up markdown and formatting
                json_str = re.sub(r'```json|```', '', json_str)
                json_str = re.sub(r'^[^{]*', '', json_str)  # Remove anything before first {
                json_str = re.sub(r'[^}]*$', '', json_str)  # Remove anything after last }

                # Replace Python-like syntax with valid JSON
                json_str = json_str.replace("None", "null")
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', json_str)
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                
                # Parse the cleaned JSON
                modeling_plan = json.loads(json_str)
                
                # Standardize model and metric names
                model_mapping = {
                    "Linear Regression": "LinearRegression",
                    "Ridge Regression": "Ridge",
                    "Lasso Regression": "Lasso",
                    "Decision Tree Regression": "DecisionTreeRegressor",
                    "Random Forest Regression": "RandomForestRegressor",
                    "Support Vector Regression": "SVR"
                }
                
                metric_mapping = {
                    "Mean Absolute Error (MAE)": "mae",
                    "Mean Squared Error (MSE)": "mse",
                    "Root Mean Squared Error (RMSE)": "rmse",
                    "R-squared": "r2"
                }
                
                if "model_types" in modeling_plan:
                    modeling_plan["model_types"] = [
                        model_mapping.get(model, model) 
                        for model in modeling_plan["model_types"]
                    ]
                
                if "evaluation_metrics" in modeling_plan:
                    modeling_plan["evaluation_metrics"] = [
                        metric_mapping.get(metric, metric.lower())
                        for metric in modeling_plan["evaluation_metrics"]
                    ]
                
                logger.info(f"Successfully parsed modeling recommendations after cleanup")
                return modeling_plan
                
        except Exception as e:
            logger.error(f"Failed to parse modeling recommendations: {e}")
            logger.error(f"Original response: {response}")
            
            # Return comprehensive default regression plan
            return {
                "model_types": ["LinearRegression", "Ridge", "Lasso", "RandomForestRegressor"],
                "evaluation_metrics": ["rmse", "mae", "r2"],
                "special_considerations": [
                    "Small sample size may lead to high variance in results",
                    "Consider using cross-validation for more reliable performance estimates",
                    "Regularization recommended to prevent overfitting"
                ],
                "hyperparameter_ranges": {
                    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
                    "Lasso": {"alpha": [0.01, 0.1, 1, 10, 100]},
                    "RandomForestRegressor": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10]
                    }
                },
                "reasoning": "Default regression configuration used due to parsing error"
            }

    def train_and_evaluate_models(self, preprocessed_data: Dict[str, Any], modeling_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train and evaluate models based on the modeling plan
        """
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from xgboost import XGBRegressor, XGBClassifier
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import pickle
        
        X_train = preprocessed_data["X_train_processed"]
        X_test = preprocessed_data["X_test_processed"]
        y_train = preprocessed_data["y_train"]
        y_test = preprocessed_data["y_test"]

        
        
        # Determine problem type based on target variable
        unique_targets = len(np.unique(y_train))
        is_classification = unique_targets < 10 and not pd.api.types.is_numeric_dtype(y_train)

        # Get metrics from modeling plan or use defaults
        metrics = modeling_plan.get("evaluation_metrics", ["rmse", "mae", "r2"] if not is_classification else ["accuracy", "f1"])
        feature_names = preprocessed_data.get("feature_names", [])
        
        # Initialize models based on recommendations
        models = []
        model_types = modeling_plan.get("model_types", [])
        
        for model_type in model_types:
            try:
                model_type_lower = model_type.lower()
                
                if is_classification:
                    if "randomforest" in model_type_lower:
                        models.append(("RandomForest", RandomForestClassifier()))
                    elif "xgboost" in model_type_lower or "gradient" in model_type_lower:
                        models.append(("XGBoost", XGBClassifier()))
                    elif "logistic" in model_type_lower:
                        models.append(("LogisticRegression", LogisticRegression()))
                else:  # Regression
                    if "randomforest" in model_type_lower:
                        models.append(("RandomForest", RandomForestRegressor()))
                    elif "xgboost" in model_type_lower or "gradient" in model_type_lower:
                        models.append(("XGBoost", XGBRegressor()))
                    elif "linear" in model_type_lower:
                        models.append(("LinearRegression", LinearRegression()))
                    elif "ridge" in model_type_lower:
                        models.append(("Ridge", Ridge()))
                    elif "lasso" in model_type_lower:
                        models.append(("Lasso", Lasso()))
                        
            except Exception as e:
                logger.error(f"Error initializing {model_type}: {str(e)}")
        
        
        if not models:
            raise ValueError("No valid models were initialized")
        
        # Train and evaluate models
        model_results = {}
        best_model = None
        best_score = -np.inf if metrics[0] in ["accuracy", "r2", "roc_auc"] else np.inf
        

        for name, model in models:
            try:
                logger.info(f"Training {name} model...")
                
                # Train model
                print("X_train, ", X_train)
                print("y_train, ", y_train)
                print("model, ", model)
                
                model.fit(X_train, y_train)
                
                # Evaluate on train and test sets
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                if is_classification and hasattr(model, "predict_proba"):
                    train_proba = model.predict_proba(X_train)
                    test_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                scores = {}
                for metric in metrics:
                    try:
                        if metric == "rmse":
                            scores["train_rmse"] = np.sqrt(mean_squared_error(y_train, train_pred))
                            scores["test_rmse"] = np.sqrt(mean_squared_error(y_test, test_pred))
                        elif metric == "mae":
                            scores["train_mae"] = mean_absolute_error(y_train, train_pred)
                            scores["test_mae"] = mean_absolute_error(y_test, test_pred)
                        elif metric == "r2":
                            scores["train_r2"] = r2_score(y_train, train_pred)
                            scores["test_r2"] = r2_score(y_test, test_pred)
                        elif metric == "accuracy":
                            scores["train_accuracy"] = accuracy_score(y_train, train_pred)
                            scores["test_accuracy"] = accuracy_score(y_test, test_pred)
                        elif metric == "f1":
                            scores["train_f1"] = f1_score(y_train, train_pred, average='weighted')
                            scores["test_f1"] = f1_score(y_test, test_pred, average='weighted')
                        elif metric == "roc_auc":
                            if unique_targets == 2:
                                scores["train_roc_auc"] = roc_auc_score(y_train, train_proba[:, 1])
                                scores["test_roc_auc"] = roc_auc_score(y_test, test_proba[:, 1])
                            else:
                                scores["train_roc_auc"] = roc_auc_score(y_train, train_proba, multi_class='ovr')
                                scores["test_roc_auc"] = roc_auc_score(y_test, test_proba, multi_class='ovr')
                    except Exception as e:
                        logger.error(f"Error calculating {metric} for {name}: {str(e)}")
                        scores[metric] = None
                
                # Store results
                model_results[name] = {
                    "scores": scores,
                    "model": model,
                    "feature_importances": None
                }
                
                # Try to get feature importances if available
                try:
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                        model_results[name]["feature_importances"] = dict(zip(feature_names, importances))
                    elif hasattr(model, "coef_"):
                        coef = model.coef_
                        if len(coef.shape) == 1:
                            model_results[name]["feature_importances"] = dict(zip(feature_names, coef))
                        else:
                            # For multi-class classification
                            model_results[name]["feature_importances"] = {
                                f"class_{i}": dict(zip(feature_names, coef[i])) 
                                for i in range(coef.shape[0])
                            }
                except Exception as e:
                    logger.error(f"Error getting feature importances for {name}: {str(e)}")
                
                # Check if this is the best model so far
                primary_metric = metrics[0]
                test_score_key = f"test_{primary_metric}"
                
                if test_score_key in scores and scores[test_score_key] is not None:
                    if (primary_metric in ["accuracy", "r2", "roc_auc"] and scores[test_score_key] > best_score) or \
                    (primary_metric in ["rmse", "mae"] and scores[test_score_key] < best_score):
                        best_score = scores[test_score_key]
                        best_model = name
                
                logger.info(f"{name} trained. Scores: {scores}")
            
            except Exception as e:
                logger.error(f"Error training {name} model: {str(e)}")
                model_results[name] = {"error": str(e)}
        
        # Save model artifacts
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        
        for name, result in model_results.items():
            if "model" in result:
                model_path = os.path.join(self.output_dir, "models", f"{name.lower()}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(result["model"], f)
                model_results[name]["model_path"] = model_path
        
        # Plot feature importances for the best model if available
        if best_model and model_results[best_model].get("feature_importances"):
            try:
                importances = model_results[best_model]["feature_importances"]
                
                if isinstance(importances, dict):
                    # For single output models
                    importance_df = pd.DataFrame({
                        "feature": list(importances.keys()),
                        "importance": list(importances.values())
                    }).sort_values("importance", ascending=False)
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x="importance", y="feature", data=importance_df.head(20))
                    plt.title(f"Feature Importances - {best_model}")
                    plt.tight_layout()
                    
                    plot_path = os.path.join(self.output_dir, "plots", f"{best_model.lower()}_feature_importances.png")
                    plt.savefig(plot_path)
                    plt.close()
                    
                    model_results[best_model]["feature_importance_plot"] = plot_path
                
                elif isinstance(importances, list):
                    # For multi-output models
                    for i, imp_dict in enumerate(importances):
                        importance_df = pd.DataFrame({
                            "feature": list(imp_dict.keys()),
                            "importance": list(imp_dict.values())
                        }).sort_values("importance", ascending=False)
                        
                        plt.figure(figsize=(10, 6))
                        sns.barplot(x="importance", y="feature", data=importance_df.head(20))
                        plt.title(f"Feature Importances - {best_model} (Class {i})")
                        plt.tight_layout()
                        
                        plot_path = os.path.join(self.output_dir, "plots", f"{best_model.lower()}_feature_importances_class_{i}.png")
                        plt.savefig(plot_path)
                        plt.close()
                        
                        model_results[best_model][f"feature_importance_plot_class_{i}"] = plot_path
            
            except Exception as e:
                logger.error(f"Error plotting feature importances: {str(e)}")
        
        # Update results
        self.results["modeling"] = {
            "model_results": model_results,
            "best_model": best_model,
            "metrics_used": metrics,
            "modeling_plan": modeling_plan
        }
        
        return model_results

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
        
        # Step 1: Get dataset recommendations (includes target column identification)
        logger.info("Step 1: Getting dataset recommendations")
        dataset_recommendations = self.get_dataset_recommendation(filepath)

        # Step 2: Run exploratory data analysis (includes target validation)
        logger.info("Step 2: Running exploratory data analysis")
        eda_results = self.run_eda(dataset_recommendations)

        if eda_results == {}:
            logger.error("Stopping since no valid target column was found")
            sys.exit()
        
        
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
        
        # Step 7: Get modeling recommendations (using target info from EDA results)
        logger.info("Step 7: Getting modeling recommendations")

        
        # Determine problem type from existing EDA results
        target_type = "numeric" if pd.api.types.is_numeric_dtype(self.df[self.target_column]) else "categorical"
        unique_values = len(self.df[self.target_column].unique())
        is_classification = False  # Hard-code
        
        target_info = {
            "type": target_type,
            "unique_values": unique_values,
            "is_classification": is_classification,
            "class_balance": self.df[self.target_column].value_counts(normalize=True).to_dict() if is_classification else None
        }
        
        modeling_plan = self.get_modeling_recommendation(preprocessed_data, target_info)

        # Step 8: Train and evaluate models
        logger.info("Step 8: Training and evaluating models")
        model_results = self.train_and_evaluate_models(preprocessed_data, modeling_plan)
        
        # Step 9: Generate final report and save results
        logger.info("Step 9: Generating final report")
        
        # Custom JSON encoder for NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (float, np.float64, np.float32, np.float16)):
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
        def make_json_safe(obj):
            """Recursively remove or replace non-serializable values like models, functions, etc."""
            if isinstance(obj, dict):
                return {
                    k: make_json_safe(v)
                    for k, v in obj.items()
                    if not callable(v) and k != "model"  # remove models and callables
                }
            elif isinstance(obj, (list, tuple, set)):
                return [make_json_safe(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (float, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray, pd.Series)):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return f"<DataFrame with shape {obj.shape}>"
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                try:
                    json.dumps(obj)  # test if it's serializable
                    return obj
                except:
                    return str(obj)  # fallback to string

        # Convert full results to JSON-safe structure
        safe_results = make_json_safe(self.results)

        # Save to file
        output_json_path = os.path.join(self.output_dir, "pipeline_results.json")
        with open(output_json_path, "w") as f:
            json.dump(safe_results, f, indent=2)

        # Also pretty-print to console for inspection
        pprint.pprint(safe_results, indent=2, width=100)
        logger.info(f"Final results saved to {output_json_path}")

        
        logger.info("Pipeline completed successfully")
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