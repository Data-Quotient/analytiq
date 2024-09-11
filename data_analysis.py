import streamlit as st
import polars as pl
import plotly.express as px
from data_utils import *
from polars_datatypes import NUMERIC_TYPES

def display_bivariate_analysis(df: pl.DataFrame, x_column: str, y_column: str):
    """Displays bivariate analysis including scatter plots, bar charts, or correlation coefficients."""
    st.write(f"Analyzing the relationship between {x_column} and {y_column}")
    
    x_dtype = df[x_column].dtype
    y_dtype = df[y_column].dtype

    st.write(f"X-axis ({x_column}) Data Type: {x_dtype}")
    st.write(f"Y-axis ({y_column}) Data Type: {y_dtype}")

    # Check for missing values
    if df[x_column].is_null().any() or df[y_column].is_null().any():
        st.warning(f"Missing values detected in {x_column} or {y_column}. This might affect the analysis.")
    
    # Visualizations
    st.subheader(f"Visualizations for {x_column} vs {y_column}")
    
    try:
        if pl.datatypes.Float32 in [df[x_column].dtype, df[y_column].dtype] or pl.datatypes.Float64 in [df[x_column].dtype, df[y_column].dtype]:
            st.write("Scatter Plot")
            scatter_fig = px.scatter(df.to_pandas(), x=x_column, y=y_column, title=f'Scatter Plot of {x_column} vs {y_column}')
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            st.write("Correlation Coefficient")
            correlation = df.select(pl.corr(x_column, y_column)).to_pandas().iloc[0, 0]
            st.metric(label=f"Correlation between {x_column} and {y_column}", value=f"{correlation:.2f}")
        
        elif pl.datatypes.Utf8 in [df[x_column].dtype, df[y_column].dtype]:
            st.write("Bar Chart")
            bar_fig = px.bar(df.to_pandas(), x=x_column, y=y_column, title=f'Bar Chart of {x_column} vs {y_column}', color=x_column)
            st.plotly_chart(bar_fig, use_container_width=True)
        
        else:
            st.warning(f"Cannot generate visualization for the selected columns: {x_column} (type: {x_dtype}) and {y_column} (type: {y_dtype}). The data types may not be compatible for visualization.")
    
    except Exception as e:
        st.error(f"An error occurred while generating the visualization: {e}")

def display_multivariate_analysis(df: pl.DataFrame, selected_columns: list):
    """Displays multivariate analysis including pair plots, heatmaps, or 3D scatter plots."""
    st.write(f"Analyzing relationships between: {', '.join(selected_columns)}")
    
    # Convert to pandas for plotting
    pandas_df = df.select(selected_columns).to_pandas()

    # Visualizations
    st.subheader("Visualizations")
    
    try:
        # Pair Plot
        if len(selected_columns) > 1:
            st.write("Pair Plot")
            pair_plot_fig = px.scatter_matrix(pandas_df, dimensions=selected_columns, title="Pair Plot")
            st.plotly_chart(pair_plot_fig, use_container_width=True)
        
        # Heatmap
        if len(selected_columns) > 1:
            st.write("Heatmap")
            corr_matrix = pandas_df.corr()
            heatmap_fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # 3D Scatter Plot (only if 3 columns are selected)
        if len(selected_columns) == 3 and all(pandas_df[col].dtype in [float, int] for col in selected_columns):
            st.write("3D Scatter Plot")
            scatter_3d_fig = px.scatter_3d(pandas_df, x=selected_columns[0], y=selected_columns[1], z=selected_columns[2], title="3D Scatter Plot")
            st.plotly_chart(scatter_3d_fig, use_container_width=True)
        
        if len(selected_columns) > 3:
            st.warning("3D Scatter Plot only supports three variables. Please select exactly three columns to view the plot.")
        
    except Exception as e:
        st.error(f"An error occurred while generating the visualization: {e}")

def display_univariate_analysis(df: pl.DataFrame, column: str):
    """Displays univariate analysis including distribution plots and summary statistics."""
    st.write(f"Analyzing {column}")
    
    # Display summary statistics in a more compact format
    summary = column_summary(df, column)
    
    st.write("Summary Statistics:")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Convert values to strings where necessary for display
    col1.metric("Data Type", str(summary['Data Type']))
    col2.metric("Unique Values", summary['Unique Values'])
    col3.metric("Missing Values", summary['Missing Values'])
    
    if pl.datatypes.Float32 in [df[column].dtype, pl.datatypes.Float64] or pl.datatypes.Int8 in [df[column].dtype, pl.datatypes.Int16, pl.datatypes.Int32, pl.datatypes.Int64]:
        col4.metric("Mean", summary['Mean'] if summary['Mean'] is not None else "N/A")
        col5.metric("Median", summary['Median'] if summary['Median'] is not None else "N/A")
        col6.metric("Mode", summary['Mode'] if summary['Mode'] is not None else "N/A")
        
        col1.metric("Standard Deviation", summary['Standard Deviation'] if summary['Standard Deviation'] is not None else "N/A")
        col2.metric("Min", summary['Min'] if summary['Min'] is not None else "N/A")
        col3.metric("Max", summary['Max'] if summary['Max'] is not None else "N/A")
    
    else:
        col4.metric("Mode", str(summary['Mode']))

    # Display visualizations
    st.subheader(f"Visualizations for {column}")
    
    if pl.datatypes.Float32 in [df[column].dtype, pl.datatypes.Float64] or pl.datatypes.Int8 in [df[column].dtype, pl.datatypes.Int16, pl.datatypes.Int32, pl.datatypes.Int64]:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Histogram")
            hist_fig = px.histogram(df, x=column, nbins=30, title=f'Histogram of {column}')
            st.plotly_chart(hist_fig, use_container_width=True)

        with col2:
            st.write("Box Plot")
            box_fig = px.box(df, y=column, title=f'Box Plot of {column}')
            st.plotly_chart(box_fig, use_container_width=True)
        
        st.write("Density Plot")
        density_fig = px.density_contour(df, x=column, title=f'Density Plot of {column}')
        st.plotly_chart(density_fig, use_container_width=True)
    
    else:
        st.write("Bar Plot")
        value_counts_df = df.group_by(column).agg(pl.count())
        bar_fig = px.bar(value_counts_df, x=column, y='count', title=f'Bar Plot of {column}')
        st.plotly_chart(bar_fig, use_container_width=True)

def display_correlation_analysis(df: pl.DataFrame):
    """Displays correlation analysis including a correlation matrix and heatmap."""
    st.write("Analyzing correlations between numerical variables")

    # Select only numeric columns for correlation analysis
    numeric_df = df.select([pl.col(col) for col in df.columns if df.schema[col] in NUMERIC_TYPES])
    
    if numeric_df.is_empty():
        st.warning("No numeric columns available for correlation analysis.")
        return

    # Convert to pandas for plotting
    pandas_numeric_df = numeric_df.to_pandas()

    # Display the correlation matrix in an accordion
    with st.expander("Correlation Matrix", expanded=True):
        st.subheader("Correlation Matrix")
        corr_matrix = pandas_numeric_df.corr()
        st.dataframe(corr_matrix)

        # Debugging: Display the correlation matrix to ensure it has data
        st.write("Debug: Correlation Matrix Data")
        st.write(corr_matrix)

    # Display heatmap of the correlation matrix in another accordion
    with st.expander("Correlation Heatmap", expanded=False):
        st.subheader("Correlation Heatmap")
        try:
            heatmap_fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(heatmap_fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred while generating the heatmap: {e}")
