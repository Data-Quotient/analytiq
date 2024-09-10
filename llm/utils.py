import polars as pl
from typing import List, Dict, Any
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import streamlit as st
import h2o
from openai import OpenAI
from machine_learning.model_mapping import MODEL_MAPPING
from llm.ollama_utils import get_ollama_response

# Load the OpenAI API key from Streamlit secrets
def get_llm_response(prompt: str) -> str:    
    
    system_prompt = {'BASE':"You are an AI assistant that provides insightful analysis of machine learning models and results, focusing on actionable insights for business decision-makers."}
    
    try:
        use_ollama = st.secrets["OLLAMA_USAGE"] == "True"
        
        if use_ollama:
            res= get_ollama_response(system_prompt['BASE']+prompt)
            return res
        else:
            client = OpenAI(api_key=st.secrets.get("openai_api_key"))

            # Create the chat completion
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt['BASE']},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # Return the content of the response
            return completion.choices[0].message.content
    
    except Exception as e:
        return f"Error getting LLM explanation: {str(e)}"

def suggest_models(use_case: str, problem_type: str, data_head: pl.DataFrame, summary: dict, detailed_stats: pl.DataFrame) -> List[str]:
    if problem_type not in MODEL_MAPPING:
        return []

    available_models = MODEL_MAPPING[problem_type]

    # Convert data head and detailed statistics to string for inclusion in the prompt
    data_head_str = data_head.head().to_pandas().to_string(index=False)
    detailed_stats_str = detailed_stats.describe().to_pandas().to_string(index=False)

    prompt = f"""
    You are an AI assistant that suggests the most appropriate machine learning models based on the use case, data summary, and statistics provided.

    Use case: {use_case}

    Data Head:
    {data_head_str}

    Data Summary:
    Number of Rows: {summary['Number of Rows']}
    Number of Columns: {summary['Number of Columns']}
    Missing Values: {summary['Missing Values']}
    Duplicate Rows: {summary['Duplicate Rows']}
    Memory Usage: {summary['Memory Usage (MB)']} MB

    Detailed Statistics:
    {detailed_stats_str}

    Based on the above information, suggest the top 3 most appropriate machine learning models from the following list:

    {', '.join([f'{k} ({v})' for k, v in available_models.items()])}

    Please provide your suggestions as a comma-separated list of model names, without numbering or explanations. For example: " GBM, DeepLearning"
    """

    suggestions = get_llm_response(prompt)

    # Split the suggestions into a list and strip whitespace
    suggested_models = [model.strip() for model in suggestions.split(',')]

    # Filter suggestions to only include valid models from the mapping
    valid_models = [model for model in suggested_models if model in available_models]

    return valid_models

def explain_predictions(predictions_df: pl.DataFrame, problem_type: str, feature_importance: Dict[str, Any] = None, actual_values: pl.Series = None) -> str:
    pred_summary = predictions_df.select(pl.col('predict').describe()).to_pandas().to_dict()
    total_predictions = len(predictions_df)
    
    performance_metrics = ""
    if actual_values is not None:
        actual_values_pd = actual_values.to_pandas()
        if problem_type == "regression":
            mse = mean_squared_error(actual_values_pd, predictions_df['predict'].to_pandas())
            r2 = r2_score(actual_values_pd, predictions_df['predict'].to_pandas())
            performance_metrics = f"Mean Squared Error: {mse:.4f}\nR-squared: {r2:.4f}"
        elif problem_type == "classification":
            accuracy = accuracy_score(actual_values_pd, predictions_df['predict'].to_pandas())
            conf_matrix = confusion_matrix(actual_values_pd, predictions_df['predict'].to_pandas())
            class_report = classification_report(actual_values_pd, predictions_df['predict'].to_pandas())
            performance_metrics = f"Accuracy: {accuracy:.4f}\nConfusion Matrix:\n{conf_matrix}\nClassification Report:\n{class_report}"

    prompt = f"""
    As a senior data scientist, provide a comprehensive analysis of the following {problem_type} model predictions:

    Prediction Summary:
    {pred_summary}
    Total Predictions: {total_predictions}

    Sample Predictions:
    {predictions_df.head(10).to_pandas().to_string()}

    {'Feature Importance:' + str(feature_importance) if feature_importance else 'Feature importance not available.'}

    Performance Metrics:
    {performance_metrics}

    Analysis Instructions:
    1. Model Performance:
       - Evaluate overall performance using provided metrics.
       - Identify strengths and weaknesses.
       - Suggest specific improvements.

    2. Prediction Patterns:
       - Analyze prediction distribution and anomalies.
       - Connect patterns to real-world implications.
       - Highlight unexpected findings.

    3. Feature Impact:
       - Interpret feature importance in context.
       - Explain how top features influence predictions.
       - Recommend feature-based strategies.

    4. Business Implications:
       - Translate results into concrete business impacts.
       - Provide actionable recommendations for stakeholders.
       - Address potential risks in applying predictions.

    5. Future Steps:
       - Suggest areas for further investigation.
       - Recommend additional use cases.
       - Outline steps to maximize business value.

    Deliver your analysis in clear, strategic language suitable for both technical and non-technical stakeholders. Focus on actionable insights and decision support.
    """
    
    explanation = get_llm_response(prompt)
    return explanation

def suggest_target_column(task: str, available_columns: pl.Series, use_case: str, data_head: pl.DataFrame, summary: dict, detailed_stats: pl.DataFrame) -> str:
    """Suggest the most appropriate target column based on the task, list of available columns, use case, and dataset details."""
    
    # Convert columns list to a string
    columns_str = ", ".join(available_columns.to_list())
    
    # Convert data head and detailed statistics to string for inclusion in the prompt
    data_head_str = data_head.head().to_pandas().to_string(index=False)
    detailed_stats_str = detailed_stats.describe().to_pandas().to_string(index=False)
    
    # Build the prompt
    prompt = f"""
    You are an AI assistant specialized in machine learning. Your task is to suggest the most appropriate target column for a {task} task.
    
    The use case is: {use_case}.
    
    Here is the head of the dataset:
    {data_head_str}
    
    Here is a summary of the dataset:
    Number of Rows: {summary['Number of Rows']}
    Number of Columns: {summary['Number of Columns']}
    Missing Values: {summary['Missing Values']}
    Duplicate Rows: {summary['Duplicate Rows']}
    Memory Usage: {summary['Memory Usage (MB)']} MB
    
    Here are the detailed statistics for the dataset:
    {detailed_stats_str}
    
    You can ONLY choose one column from the following list of columns: {columns_str}.
    Do NOT suggest any columns that are not in this list.
    
    Please respond with exactly one column name that is most suitable as the target column for the {task} task based on the use case and the provided list.
    """
    
    suggested_column = get_llm_response(prompt).strip()
   
    return suggested_column

def generate_leaderboard_commentary(use_case: str, data_head: pl.DataFrame, selected_models: List[str], leaderboard: pl.DataFrame) -> str:
    """Generate commentary on the leaderboard from the LLM."""
    
    # Convert data head and leaderboard to string for inclusion in the prompt
    data_head_str = data_head.head().to_pandas().to_string(index=False)
    leaderboard_str = leaderboard.to_pandas().to_string(index=False)

    prompt = f"""
    You are an AI assistant specialized in machine learning. Your task is to provide a commentary on the model leaderboard.

    Use case: {use_case}

    Data Head:
    {data_head_str}

    Selected Models: {', '.join(selected_models)}

    Leaderboard:
    {leaderboard_str}

    Please provide a comprehensive analysis of the model performances, including the strengths and weaknesses of the top models, and suggest the most appropriate model for the use case based on the leaderboard results.
    """
    
    leaderboard_commentary = get_llm_response(prompt).strip()
   
    return leaderboard_commentary

def explain_predictions_commentary(predictions_df: pl.DataFrame, actual_values: pl.Series = None) -> str:
    pred_summary = predictions_df.select(pl.col('predict').describe()).to_pandas().to_dict()
    total_predictions = len(predictions_df)

    performance_metrics = ""
    if actual_values is not None:
        actual_values_pd = actual_values.to_pandas()
        mse = mean_squared_error(actual_values_pd, predictions_df['predict'].to_pandas())
        r2 = r2_score(actual_values_pd, predictions_df['predict'].to_pandas())
        performance_metrics = f"Mean Squared Error: {mse:.4f}\nR-squared: {r2:.4f}"

    prompt = f"""
    Provide an analysis of the following regression predictions:

    Prediction Summary:
    {pred_summary}
    Total Predictions: {total_predictions}

    Performance Metrics:
    {performance_metrics}

    Please provide insights into the model's predictions, identify patterns, and suggest improvements or business implications.
    """

    explanation = get_llm_response(prompt)
    return explanation

def explain_feature_importance_commentary(feature_importance_df: pl.DataFrame) -> str:
    feature_importance_summary = feature_importance_df.describe().to_pandas().to_dict()

    prompt = f"""
    Analyze the following feature importance summary and provide insights:

    Feature Importance Summary:
    {feature_importance_summary}

    Provide a comprehensive analysis of how these features likely influence the model's decisions and suggest potential business actions.
    """

    explanation = get_llm_response(prompt)
    return explanation

def explain_insights_commentary(predictions_df: pl.DataFrame, feature_importance_df: pl.DataFrame) -> str:
    pred_summary = predictions_df.select(pl.col('predict').describe()).to_pandas().to_dict()
    feature_importance_summary = feature_importance_df.describe().to_pandas().to_dict()

    prompt = f"""
    Provide a business-oriented analysis of the following machine learning model predictions and feature importance:

    Prediction Summary:
    {pred_summary}

    Feature Importance Summary:
    {feature_importance_summary}

    Provide actionable business insights, potential risks, and recommendations for decision-makers.
    """

    explanation = get_llm_response(prompt)
    return explanation

def generate_industry_report(use_case: str, task: str, data: pl.DataFrame, target_column: str, aml, predictions: pl.DataFrame, feature_importance: pl.DataFrame) -> str:
    """Generate a comprehensive, jargon-free industry report based on the ML analysis."""
    
    # Safely convert to pandas DataFrame if needed
    def safe_to_pandas(df):
        if isinstance(df, h2o.H2OFrame):
            return df.as_data_frame()
        elif isinstance(df, pl.DataFrame):
            return df.to_pandas()
        else:
            return pd.DataFrame(df)  # Attempt to convert unknown types

    leaderboard_df = safe_to_pandas(aml.leaderboard)
    predictions_df = safe_to_pandas(predictions)
    feature_importance_df = safe_to_pandas(feature_importance) if feature_importance is not None else None

    prompt = f"""
    As a seasoned data scientist and business consultant, create a concise, jargon-free industry report based on the following machine learning analysis:

    Use Case: {use_case}
    Task: {task}
    Target Variable: {target_column}
    Dataset Overview:
    - Rows: {len(data)}
    - Columns: {len(data.columns)}
    - Features: {', '.join(data.columns)}

    Model Performance:
    {leaderboard_df.head().to_string()}

    Top 5 Important Features:
    {feature_importance_df.head().to_string() if feature_importance_df is not None else "Not available"}

    Prediction Sample:
    {predictions_df.head().to_string()}

    Instructions:
    1. Summarize the business problem and its importance.
    2. Explain the chosen approach and why it's suitable for this use case.
    3. Present key findings and insights derived from the model.
    4. Discuss the practical implications of these findings for the industry.
    5. Provide 3-5 actionable recommendations based on the analysis.
    6. Suggest next steps or areas for further investigation.

    Your report should be clear, concise, and free of technical jargon. Focus on the business value and practical applications of the insights gained from this analysis.
    """

    industry_report = get_llm_response(prompt)
    return industry_report
