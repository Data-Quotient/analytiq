try:
    from openai import OpenAI
    import os
    from typing import List, Dict, Any
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
    import streamlit as st
except ModuleNotFoundError as e:
    import subprocess
    import sys
    missing_package = str(e).split("'")[1]  # Get the missing package name

    # Install the missing package
    subprocess.check_call([sys.executable, "-m", "pip", "install", missing_package])

    # Re-import the module after installation
    if missing_package == "openai":
        from openai import OpenAI
    elif missing_package == "sklearn":
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
    elif missing_package == "streamlit":
        import streamlit as st

    from openai import OpenAI
    import os
    from typing import List, Dict, Any
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
    import streamlit as st

import pandas as pd
from machine_learning.model_mapping import MODEL_MAPPING


# Load the OpenAI API key from Streamlit secrets

def get_llm_response(prompt: str) -> str:
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key = st.secrets["openai_api_key"])

        # Create the chat completion
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides insightful analysis of machine learning models and results, focusing on actionable insights for business decision-makers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # Return the content of the response
        return completion.choices[0].message.content
    
    except Exception as e:
        return f"Error getting LLM explanation: {str(e)}"

    
def suggest_models(use_case: str, problem_type: str, data_head: pd.DataFrame, summary: dict, detailed_stats: pd.DataFrame) -> List[str]:
    if problem_type not in MODEL_MAPPING:
        return []

    available_models = MODEL_MAPPING[problem_type]

    # Convert data head and detailed statistics to string for inclusion in the prompt
    data_head_str = data_head.to_string(index=False)
    detailed_stats_str = detailed_stats.to_string(index=False)

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

    Please provide your suggestions as a comma-separated list of model names, without numbering or explanations. For example: "XGBoost, GBM, DeepLearning"
    """

    suggestions = get_llm_response(prompt)

    # Split the suggestions into a list and strip whitespace
    suggested_models = [model.strip() for model in suggestions.split(',')]

    # Filter suggestions to only include valid models from the mapping
    valid_models = [model for model in suggested_models if model in available_models]

    return valid_models


def explain_predictions(predictions_df, problem_type: str, feature_importance: Dict[str, Any] = None, actual_values: pd.Series = None) -> str:
    pred_summary = predictions_df['predict'].describe().to_dict()
    total_predictions = len(predictions_df)
    
    performance_metrics = ""
    if actual_values is not None:
        if problem_type == "regression":
            mse = mean_squared_error(actual_values, predictions_df['predict'])
            r2 = r2_score(actual_values, predictions_df['predict'])
            performance_metrics = f"Mean Squared Error: {mse:.4f}\nR-squared: {r2:.4f}"
        elif problem_type == "classification":
            accuracy = accuracy_score(actual_values, predictions_df['predict'])
            conf_matrix = confusion_matrix(actual_values, predictions_df['predict'])
            class_report = classification_report(actual_values, predictions_df['predict'])
            performance_metrics = f"Accuracy: {accuracy:.4f}\nConfusion Matrix:\n{conf_matrix}\nClassification Report:\n{class_report}"

    prompt = f"""
    Analyze the following {problem_type} predictions and provide a comprehensive, insightful explanation:

    Prediction Summary:
    {pred_summary}
    Total Predictions: {total_predictions}

    Sample of predictions:
    {predictions_df.head(10).to_string()}

    {'Feature Importance:' + str(feature_importance) if feature_importance else ''}

    Performance Metrics:
    {performance_metrics}

    Please provide an in-depth analysis of these predictions, including:

    1. Model Performance:
       - Evaluate the overall performance of the model based on the provided metrics.
       - Identify areas where the model excels and where it might be falling short.
       - Suggest potential improvements or next steps for model enhancement.

    2. Prediction Patterns and Insights:
       - Analyze the distribution of predictions and identify any significant patterns or anomalies.
       - Relate these patterns to potential real-world implications or business scenarios.
       - Highlight any surprising or counterintuitive findings in the predictions.

    3. Feature Impact Analysis:
       - Interpret the feature importance in the context of the predictions.
       - Explain how the most important features are likely influencing the model's decisions.
       - Suggest potential actions or strategies based on the feature importance.

    4. Business Implications and Actionable Insights:
       - Translate the model's predictions and performance into concrete business implications.
       - Provide specific, actionable recommendations for decision-makers based on these insights.
       - Identify any potential risks or limitations in applying these predictions to real-world scenarios.

    5. Future Outlook and Recommendations:
       - Based on the current model performance and predictions, suggest areas for further investigation or data collection.
       - Recommend potential use cases or applications for this model in the business context.
       - Outline next steps for leveraging these predictions to drive business value.

    Provide your explanation in clear, strategic language suitable for business stakeholders, focusing on actionable insights and decision-making support.
    """
    
    explanation = get_llm_response(prompt)
    return explanation
