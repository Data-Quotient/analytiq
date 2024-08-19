import pandas as pd
import streamlit as st

import plotly.express as px

from data_utils import (
    column_summary, 
    generate_summary,
    detailed_statistics
)

# Function to display the summary as tiles
def display_summary_tiles(summary):
    """Displays the summary statistics in a tile format."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Rows", summary['Number of Rows'])
    col1.metric("Number of Columns", summary['Number of Columns'])
    col2.metric("Missing Values", summary['Missing Values'])
    col2.metric("Duplicate Rows", summary['Duplicate Rows'])
    col3.metric("Memory Usage (MB)", summary['Memory Usage (MB)'])

# Function to display the column-level summary and distribution side by side
def display_column_summary(df, column):
    """Displays the summary of the selected column with distribution plots."""
    summary = column_summary(df, column)
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Data Type:", summary['Data Type'])
        st.write("Unique Values:", summary['Unique Values'])
        st.write("Missing Values:", summary['Missing Values'])
        st.write("Mean:", summary['Mean'])
        st.write("Median:", summary['Median'])
        st.write("Mode:", summary['Mode'])
        st.write("Standard Deviation:", summary['Standard Deviation'])
        st.write("Min:", summary['Min'])
        st.write("Max:", summary['Max'])
    
    with col2:
        st.subheader(f"Distribution of {column}")
        if pd.api.types.is_numeric_dtype(df[column]):
            fig = px.histogram(df, x=column, marginal="box", nbins=30, title=f'Distribution of {column}')
        else:
            fig = px.histogram(df, x=column, color=column, title=f'Distribution of {column}')
        st.plotly_chart(fig, use_container_width=True)

# Function to handle the first tab (Data Summary)
def handle_data_summary_tab(filtered_data):
    """Handles all content and logic within the Data Summary tab."""
    st.header("Data Summary")
    
    # Display summary statistics in tiles
    summary = generate_summary(filtered_data)
    display_summary_tiles(summary)
    
    # Combined accordion with two tabs for detailed statistics and column-level summary
    st.header("Detailed Analysis")
    with st.expander("View Detailed Analysis", expanded=False):
        sub_tabs = st.tabs(["Detailed Statistics", "Column-Level Summary"])
        
        with sub_tabs[0]:
            st.subheader("Detailed Statistics")
            st.dataframe(detailed_statistics(filtered_data), use_container_width=True)
        
        with sub_tabs[1]:
            st.subheader("Column-Level Summary")
            selected_column = st.selectbox("Select Column", filtered_data.columns)
            if selected_column:
                display_column_summary(filtered_data, selected_column)