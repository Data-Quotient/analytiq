import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from typing import Dict, Any, List
from llm.utils import explain_predictions
import streamlit as st
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
import io

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

def generate_pdf_report(report_content):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    title = 'Industry Report: ML Analysis Insights'
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))

    for line in report_content.split('\n'):
        if line.startswith('**') and line.endswith('**'):
            # It's a header
            story.append(Paragraph(line.strip('*'), styles['Heading2']))
        else:
            # It's a regular paragraph
            story.append(Paragraph(line, styles['Justify']))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer
def save_model(model, file_name: str):
    """Save the model using the built-in pickle module."""
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name: str):
    """Load the model using the built-in pickle module."""
    with open(file_name, 'rb') as file:
        return pickle.load(file)
