import pandas as pd
import streamlit as st

import plotly.express as px

from data_analysis import (
    display_univariate_analysis,
    display_bivariate_analysis,
    display_multivariate_analysis,
    display_correlation_analysis
)

def handle_data_analysis_tab(filtered_data):
    """Handles all content and logic within the Data Analysis Tab."""
    st.header("Data Analysis")
    st.write("This tab will host various analysis tools, such as univariate, bivariate, and multivariate analysis, along with other advanced data analysis features.")
    
    # Dropdown for analysis options
    analysis_option = st.selectbox(
        "Select an Analysis Type",
        options=[
            "Univariate Analysis",
            "Bivariate Analysis",
            "Multivariate Analysis",
            "Correlation Analysis",
            "Cross Tabulation"
        ]
    )
    
    # Show the description based on the selected analysis
    description = {
        "Univariate Analysis": "Analyze the distribution and summary statistics of individual variables.",
        "Bivariate Analysis": "Analyze the relationship between two variables.",
        "Multivariate Analysis": "Analyze relationships involving more than two variables.",
        "Correlation Analysis": "Analyze correlations between numerical variables.",
        "Cross Tabulation": "Analyze relationships between categorical variables."
    }[analysis_option]
    
    st.subheader(analysis_option)
    st.write(f"Description: {description}")
    st.markdown("---")
    
    # Univariate Analysis implementation
    if analysis_option == "Univariate Analysis":
        selected_column = st.selectbox("Select Column for Univariate Analysis", filtered_data.columns)
        if selected_column:
            display_univariate_analysis(filtered_data, selected_column)
    
    # Bivariate Analysis implementation
    elif analysis_option == "Bivariate Analysis":
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("Select X-axis Column", filtered_data.columns)
        with col2:
            y_column = st.selectbox("Select Y-axis Column", filtered_data.columns)
        if x_column and y_column:
            display_bivariate_analysis(filtered_data, x_column, y_column)
    # Multivariate Analysis implementation
    elif analysis_option == "Multivariate Analysis":
        selected_columns = st.multiselect("Select Columns for Multivariate Analysis", filtered_data.columns)
        if selected_columns:
            display_multivariate_analysis(filtered_data, selected_columns)
    elif analysis_option == "Correlation Analysis":
        display_correlation_analysis(filtered_data)