import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from typing import Dict, Any, List
from llm.utils import explain_predictions
import streamlit as st
import pickle

def run_h2o_automl(data: pd.DataFrame, target_column: str, problem_type: str, selected_models: List[str], max_models: int = 20):
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

def save_model(model, file_name: str):
    """Save the model using the built-in pickle module."""
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name: str):
    """Load the model using the built-in pickle module."""
    with open(file_name, 'rb') as file:
        return pickle.load(file)
