# AI-Powered Data Science Pipeline

A modular machine learning system for price prediction with both traditional ML functions and an AI orchestrator that makes intelligent data science decisions using Google's Gemini API.

## Features

This project offers two ways to build machine learning models:

### Core Modules (Use Individually)
✅ **Data Loading & Preprocessing**: Robust data handling with automatic type detection and cleaning  
✅ **Feature Engineering**: Automated creation of interaction, polynomial, and ratio features  
✅ **Exploratory Data Analysis**: Comprehensive visualization and analysis utilities  
✅ **Machine Learning Models**: Multiple supervised regression models implementation  
✅ **Unsupervised Learning**: K-Means clustering to discover price segments  
✅ **Hyperparameter Tuning**: Grid search, randomized search, and Bayesian optimization  
✅ **Model Evaluation**: Detailed performance metrics and diagnostic visualizations  

### AI Orchestrator (Autonomous Pipeline)
✅ **Intelligent Decision Making**: Uses Google's Gemini API to analyze data and recommend next steps  
✅ **Automated Workflow**: Sequences the appropriate functions based on data characteristics  
✅ **Dynamic Feature Engineering**: Identifies and creates the most relevant features for your specific dataset  
✅ **Adaptive Preprocessing**: Chooses optimal preprocessing strategies based on data analysis  
✅ **Pipeline Visualization**: Generates visualizations at each step to document the process  
✅ **Complete Documentation**: Records all decisions and results in structured output  

## Project Structure

### Core Modules
- `data_loader.py`: Utilities for loading and basic inspection of data
- `eda.py`: Exploratory data analysis functions
- `feature_engineering.py`: Feature transformation and generation
- `data_preprocessing.py`: Data cleaning, transformation, and preparation
- `model_training.py`: Training various regression models
- `model_evaluation.py`: Evaluating model performance with metrics and visualizations
- `hyperparameter_tuning.py`: Tuning model hyperparameters
- `main.py`: Main script to run the complete pipeline manually

### AI Orchestrator
- `orchestrator.py`: AI-powered pipeline that makes intelligent decisions using Gemini API

## Requirements

- Python 3.7+
- Core libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- ML libraries: xgboost, lightgbm, scikit-optimize (optional)
- Google Gemini API key (for orchestrator only)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-data-science-pipeline.git
cd ai-data-science-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Manual Pipeline

Use individual modules directly or run the complete manual pipeline:

```bash
python main.py --input laptop_data.csv --output results --target price
```

For hyperparameter tuning:

```bash
python main.py --input laptop_data.csv --output results --target price --tune
```

### AI Orchestrator

Run the AI-powered pipeline with Gemini making key decisions:

```bash
python orchestrator.py --input data/laptop_data.csv --api_key YOUR_GEMINI_API_KEY --output ai_results
```

### Making Predictions

Use your trained models to make predictions on new data:

```bash
python predict.py --input new_laptops.csv --model results/models/random_forest.joblib --preprocessor results/models/preprocessor.joblib --output predictions.csv
```

## Example: Using the AI Orchestrator

The orchestrator turns your modular code into an intelligent data science agent:

```python
from orchestrator import DataScienceOrchestrator

# Initialize the orchestrator with your Gemini API key
orchestrator = DataScienceOrchestrator(api_key="YOUR_GEMINI_API_KEY")

# Run the complete AI-powered pipeline
results = orchestrator.run_pipeline("laptop_data.csv")

# Access various insights and results
dataset_info = results["dataset_info"]
eda_results = results["eda_results"]
feature_engineering = results["feature_engineering"]
preprocessing = results["preprocessing"]
```

The orchestrator:
1. Analyzes your dataset and identifies the appropriate target variable
2. Determines the most informative EDA visualizations to generate
3. Creates an optimal feature engineering plan based on data characteristics
4. Designs a customized preprocessing strategy
5. Documents every decision and generates comprehensive visualization

## Example: Manual Feature Engineering

You can also use the modules directly for more control:

```python
import pandas as pd
from feature_engineering import create_polynomial_features, create_interaction_features

# Load data
df = pd.read_csv("laptop_data.csv")

# Create polynomial features for RAM and storage
df = create_polynomial_features(df, ["ram_gb", "ssd_storage"], degree=2)

# Create interaction features between RAM and CPU speed
df = create_interaction_features(df, [("ram_gb", "cpu_speed")])
```

## Example: Model Training and Evaluation

Train and evaluate multiple regression models:

```python
from data_preprocessing import prepare_train_test_set, apply_preprocessing
from model_training import train_multiple_models
from model_evaluation import compare_models, plot_model_comparison

# Prepare data
X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_set(df, "price")
X_train_processed, X_test_processed, feature_names = apply_preprocessing(X_train, X_test, preprocessor)

# Train multiple models
models, results_df = train_multiple_models(X_train_processed, y_train, X_test_processed, y_test)

# Compare model performance
plot_model_comparison(results_df)
```

## How the AI Orchestrator Works

The orchestrator combines:
1. Your carefully crafted modular functions for specific data science tasks
2. Google's Gemini API for intelligent decision making at key workflow stages
3. Automated documentation and visualization generation
4. Consistent, reproducible pipeline execution

At each stage, the orchestrator:
- Analyzes the current state of the data and previous results
- Formulates targeted questions for the Gemini API
- Parses AI recommendations and formats them as actionable plans
- Executes the appropriate functions with recommended parameters
- Documents decisions and saves outputs

## Contributing

Contributions are welcome! Here are some ways you can contribute:

- Add new feature engineering techniques
- Implement additional machine learning models
- Improve data visualization options
- Enhance the AI orchestrator with new capabilities
- Add support for additional AI backends beyond Gemini

## License

This project is available under the MIT License.