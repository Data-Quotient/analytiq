#before running application please run pip install h2o

import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
import pickle
import os
from openai import OpenAI
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from typing import Dict, Any, List

# Initialize session state variables
if 'aml' not in st.session_state:
    st.session_state.aml = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None
if 'selected_model_id' not in st.session_state:
    st.session_state.selected_model_id = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'actual_values' not in st.session_state:
    st.session_state.actual_values = None

def get_llm_response(prompt: str) -> str:
    try:
        client = OpenAI(api_key='#api goes here')
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides insightful analysis of machine learning models and results, focusing on actionable insights for business decision-makers."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting LLM explanation: {str(e)}"

def suggest_models(use_case: str) -> List[str]:
    prompt = f"""
    Based on the following use case, suggest the most appropriate machine learning models from the list: GBM, DRF, XGBoost, GLM, DeepLearning.

    Use case: {use_case}

    Please provide your suggestions as a comma-separated list of model names, without numbering or explanations. For example: "XGBoost, GBM, DeepLearning"

    Limit your suggestions to the top 3 most appropriate models.
    """
    
    suggestions = get_llm_response(prompt)
    
    # Split the suggestions into a list and strip whitespace
    suggested_models = [model.strip() for model in suggestions.split(',')]
    
    # Map suggested models to valid H2O algorithm names
    valid_models = []
    model_mapping = {
        "GBM": "GBM",
        "DRF": "DRF",
        "XGBoost": "XGBoost",
        "GLM": "GLM",
        "DeepLearning": "DeepLearning"
    }
    
    for model in suggested_models:
        if model in model_mapping:
            valid_models.append(model_mapping[model])
    
    return valid_models

def explain_predictions(predictions_df: pd.DataFrame, problem_type: str, feature_importance: Dict[str, Any] = None, actual_values: pd.Series = None) -> str:
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

def run_h2o_automl(data, target_column, problem_type, selected_models, max_models=20):
    h2o.init()
    
    h2o_data = h2o.H2OFrame(data)
    
    x = h2o_data.columns
    y = target_column
    x.remove(y)
    
    train, valid, test = h2o_data.split_frame(ratios=[.7, .15])
    
    aml = H2OAutoML(max_models=max_models, seed=1, include_algos=selected_models)
    aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
    
    return aml, test

def run_selected_model():
    selected_model = h2o.get_model(st.session_state.selected_model_id)
    st.session_state.predictions = selected_model.predict(st.session_state.test_data)
    predictions_df = st.session_state.predictions.as_data_frame()
    
    st.session_state.feature_importance = None
    if hasattr(selected_model, 'varimp'):
        st.session_state.feature_importance = selected_model.varimp(use_pandas=True)

def get_explanation():
    predictions_df = st.session_state.predictions.as_data_frame()
    feature_importance_dict = st.session_state.feature_importance.to_dict() if st.session_state.feature_importance is not None else None
    st.session_state.explanation = explain_predictions(
        predictions_df, 
        st.session_state.problem_type,
        feature_importance_dict,
        st.session_state.actual_values
    )

def main():
    st.title("Advanced H2O AutoML Streamlit App with LLM-based Model Suggestions")
    
    # Initialize session state for selected models if it doesn't exist
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())
        
        target_column = st.selectbox("Select the target column", data.columns)
        
        st.session_state.problem_type = st.selectbox("Select the problem type", ["classification", "regression"])
        
        model_selection_method = st.radio("Choose model selection method", ["Automatic (LLM-based)", "Manual"])
        
        available_models = ["GBM", "DRF", "XGBoost", "GLM", "DeepLearning"]
        
        if model_selection_method == "Automatic (LLM-based)":
            example_use_cases = [
                "Predict customer churn for a telecom company",
                "Forecast sales for an e-commerce platform",
                "Classify sentiment of product reviews",
                "Predict housing prices in a metropolitan area",
                "Identify fraudulent transactions for a financial institution"
            ]
            selected_use_case = st.selectbox("Select an example use case or write your own:", 
                                             ["Write your own"] + example_use_cases)
            
            if selected_use_case == "Write your own":
                use_case = st.text_area("Describe your use case for model suggestion:")
            else:
                use_case = selected_use_case
                st.text_area("Selected use case:", use_case, disabled=True)
            
            if st.button("Get Model Suggestions"):
                with st.spinner("Generating model suggestions..."):
                    suggested_models = suggest_models(use_case)
                st.write("Suggested models based on your use case:")
                st.write(", ".join(suggested_models))
                st.session_state.selected_models = st.multiselect("Select models to run", suggested_models, default=suggested_models)
        else:
            st.session_state.selected_models = st.multiselect("Select models to run", available_models, default=st.session_state.selected_models)

        if st.session_state.selected_models:
            st.write("Selected models:", ", ".join(st.session_state.selected_models))
            if st.button("Run AutoML"):
                st.session_state.aml, st.session_state.test_data = run_h2o_automl(data, target_column, st.session_state.problem_type, st.session_state.selected_models)
                st.write("AutoML completed. Model leaderboard:")
                leaderboard = st.session_state.aml.leaderboard
                st.dataframe(leaderboard.as_data_frame().style.highlight_max(axis=0))
                
                best_model = st.session_state.aml.leader
                st.markdown(f"**Best model:** {best_model.model_id}")
        else:
            st.warning("Please select at least one model before running AutoML.")
    
    
    if st.session_state.aml is not None:
        leaderboard = st.session_state.aml.leaderboard
        model_ids = leaderboard['model_id'].as_data_frame().values.flatten().tolist()
        st.session_state.selected_model_id = st.selectbox("Select a model to test", model_ids)
        
        test_data_option = st.radio("Choose test data", ["Use same data", "Upload new test data"])
        
        if test_data_option == "Upload new test data":
            test_file = st.file_uploader("Choose a CSV file for testing", type="csv")
            if test_file is not None:
                test_data = pd.read_csv(test_file)
                st.session_state.test_data = h2o.H2OFrame(test_data)
                st.session_state.actual_values = test_data[st.session_state.aml.target]
        
        if st.button("Run selected model"):
            run_selected_model()
        
        if st.session_state.predictions is not None:
            st.write("Model predictions:")
            st.dataframe(st.session_state.predictions.as_data_frame())
            
            if st.session_state.feature_importance is not None:
                st.write("Feature Importance:")
                st.dataframe(st.session_state.feature_importance)
            
            if st.button("Generate Insightful Explanation"):
                with st.spinner("Generating comprehensive analysis..."):
                    get_explanation()
            
            if st.session_state.explanation:
                st.write("Prediction Analysis and Business Insights:")
                st.markdown(st.session_state.explanation)
        
            if st.button("Download model"):
                selected_model = h2o.get_model(st.session_state.selected_model_id)
                model_path = h2o.download_model(selected_model, path=".")
                with open(model_path, "rb") as f:
                    bytes_data = f.read()
                st.download_button(
                    label="Download PKL file",
                    data=bytes_data,
                    file_name=f"{st.session_state.selected_model_id}.pkl",
                    mime="application/octet-stream"
                )
                
                
if __name__ == "__main__":
    main()