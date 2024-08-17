import streamlit as st
from data_utils import *
from data_analysis import *

import json

from models import get_db, DQRule, DatasetAction, DatasetVersion  # Import the Dataset model and database session
from sqlalchemy.orm import Session


# Try to import necessary packages and install if not available
# Try to import necessary packages and install if not available
try:
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
    from scipy.stats import zscore
except ModuleNotFoundError as e:
    import subprocess
    import sys
    missing_package = str(e).split("'")[1]  # Get the missing package name
    
    # Correctly handle the sklearn package by installing scikit-learn
    if missing_package == 'sklearn':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    elif missing_package == 'plotly':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    elif missing_package == 'scipy':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    
    # Reimport after installation
    if missing_package == "plotly":
        import plotly.express as px
    elif missing_package == "sklearn":
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
    elif missing_package == "scipy":
        from scipy.stats import zscore

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

# Function to handle the Data Quality tab
def handle_data_quality_tab(filtered_data, dataset_id):
    """Handles all content and logic within the Data Quality tab."""
    st.header("Data Quality Check")
    
    # Fetch DQ rules for the selected dataset from the database
    db: Session = next(get_db())
    rules = db.query(DQRule).filter(DQRule.dataset_id == dataset_id).all()

    # Apply DQ rules and show loader while processing
    with st.spinner("Applying Data Quality Rules..."):
        violations = apply_dq_rules(filtered_data, rules)
    
    if violations:
        st.warning("Data Quality Issues Found:")
        for violation in violations:
            st.write(f"{violation['severity']}: {violation['message']} in column {violation['column']}")
    else:
        st.success("No data quality issues found!")

# Function to handle the Data Manipulation Tab
def handle_data_manipulation_tab(filtered_data, selected_version):
    """Handles all content and logic within the Data Manipulation Tab."""
    st.header("Data Manipulation")
    
    # Dropdown to select a manipulation action
    action = st.selectbox(
        "Select an Action",
        [
            "Rename Column",
            "Change Data Type",
            "Delete Column",
            "Filter Rows",
            "Add Calculated Column",
            "Fill Missing Values",
            "Duplicate Column",
            "Reorder Columns",
            "Replace Values"
        ]
    )

    db: Session = next(get_db())

    def log_action(version_id, action_type, parameters):
        """Logs the action to the database."""
        new_action = DatasetAction(
            version_id=version_id,
            action_type=action_type,
            parameters=json.dumps(parameters)  # Convert parameters to a JSON string
        )
        db.add(new_action)
        db.commit()
        # After logging the action, update the session state and the history
        if "actions" in st.session_state:
            st.session_state.actions.append(new_action)
        else:
            st.session_state.actions = [new_action]
        st.rerun()
        

    # Handling each action
    if action == "Rename Column":
        selected_column = st.selectbox("Select Column to Rename", filtered_data.columns)
        new_column_name = st.text_input("Enter New Column Name")
        if st.button("Rename Column"):
            filtered_data.rename(columns={selected_column: new_column_name}, inplace=True)
            st.write(f"Renamed column {selected_column} to {new_column_name}")
            log_action(selected_version.id, "Rename Column", {"old_name": selected_column, "new_name": new_column_name})

    elif action == "Change Data Type":
        selected_column = st.selectbox("Select Column to Change Data Type", filtered_data.columns)
        new_data_type = st.selectbox("Select New Data Type", ["int", "float", "str", "bool"])
        if st.button("Change Data Type"):
            try:
                if new_data_type == "int":
                    filtered_data[selected_column] = filtered_data[selected_column].astype(int)
                elif new_data_type == "float":
                    filtered_data[selected_column] = filtered_data[selected_column].astype(float)
                elif new_data_type == "str":
                    filtered_data[selected_column] = filtered_data[selected_column].astype(str)
                elif new_data_type == "bool":
                    filtered_data[selected_column] = filtered_data[selected_column].astype(bool)
                st.write(f"Changed data type of column {selected_column} to {new_data_type}")
                log_action(selected_version.id, "Change Data Type", {"column": selected_column, "new_type": new_data_type})
            except ValueError as e:
                st.error(f"Error changing data type: {e}")
            

    elif action == "Delete Column":
        selected_columns = st.multiselect("Select Columns to Delete", filtered_data.columns)
        if st.button("Delete Columns"):
            filtered_data.drop(columns=selected_columns, inplace=True)
            st.write(f"Deleted columns: {', '.join(selected_columns)}")
            log_action(selected_version.id, "Delete Column", {"columns": selected_columns})

    elif action == "Filter Rows":
        filter_condition = st.text_input("Enter Filter Condition (e.g., `age >= 18`)")
        if st.button("Apply Filter"):
            try:
                filtered_data.query(filter_condition, inplace=True)
                st.write(f"Applied filter: {filter_condition}")
                log_action(selected_version.id, "Filter Rows", {"condition": filter_condition})
            except Exception as e:
                st.error(f"Error applying filter: {e}")
    elif action == "Add Calculated Column":
        new_column_name = st.text_input("Enter New Column Name")
        formula = st.text_input("Enter Formula (e.g., `quantity * price`)")
        if st.button("Add Calculated Column"):
            try:
                filtered_data[new_column_name] = eval(formula, {'__builtins__': None}, filtered_data)
                st.write(f"Added calculated column {new_column_name} with formula: {formula}")
                log_action(selected_version.id, "Add Calculated Column", {"new_column": new_column_name, "formula": formula})
            except Exception as e:
                st.error(f"Error adding calculated column: {e}")
    elif action == "Fill Missing Values":
        selected_column = st.selectbox("Select Column to Fill Missing Values", filtered_data.columns)
        fill_method = st.selectbox("Select Fill Method", ["Specific Value", "Mean", "Median", "Mode"])
        fill_value = st.text_input("Enter Value (if 'Specific Value' selected)")
        if st.button("Fill Missing Values"):
            if fill_method == "Specific Value":
                filtered_data[selected_column].fillna(fill_value, inplace=True)
                log_action(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method, "value": fill_value})
            elif fill_method == "Mean":
                filtered_data[selected_column].fillna(filtered_data[selected_column].mean(), inplace=True)
                log_action(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method})
            elif fill_method == "Median":
                filtered_data[selected_column].fillna(filtered_data[selected_column].median(), inplace=True)
                log_action(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method})
            elif fill_method == "Mode":
                filtered_data[selected_column].fillna(filtered_data[selected_column].mode()[0], inplace=True)
                log_action(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method})
            st.write(f"Filled missing values in column {selected_column} using method: {fill_method}")

    elif action == "Duplicate Column":
        selected_column = st.selectbox("Select Column to Duplicate", filtered_data.columns)
        if st.button("Duplicate Column"):
            filtered_data[f"{selected_column}_duplicate"] = filtered_data[selected_column]
            st.write(f"Duplicated column: {selected_column}")
            log_action(selected_version.id, "Duplicate Column", {"column": selected_column})

    elif action == "Reorder Columns":
        new_order = st.multiselect("Select Columns in New Order", filtered_data.columns, default=list(filtered_data.columns))
        if st.button("Reorder Columns"):
            filtered_data = filtered_data[new_order]
            st.write(f"Reordered columns to: {', '.join(new_order)}")
            log_action(selected_version.id, "Reorder Columns", {"new_order": new_order})
    elif action == "Replace Values":
        selected_column = st.selectbox("Select Column to Replace Values", filtered_data.columns)
        to_replace = st.text_input("Value to Replace")
        replace_with = st.text_input("Replace With")
        if st.button("Replace Values"):
            filtered_data[selected_column].replace(to_replace, replace_with, inplace=True)
            st.write(f"Replaced {to_replace} with {replace_with} in column {selected_column}")
            log_action(selected_version.id, "Replace Values", {"column": selected_column, "to_replace": to_replace, "replace_with": replace_with})

    st.session_state.original_data = filtered_data
            



# Function to handle the Preprocessing Tab
def handle_preprocessing_tab(filtered_data, selected_version):
    """Handles all content and logic within the Preprocessing Tab."""
    st.header("Data Preprocessing")

    # Dropdown to select a preprocessing action
    action = st.selectbox(
        "Select a Preprocessing Action",
        [
            "Scale Data",
            "Encode Categorical Variables",
            "Impute Missing Values",
            "Remove Outliers"
        ]
    )

    db: Session = next(get_db())

    def log_action(version_id, action_type, parameters):
        """Logs the action to the database."""
        new_action = DatasetAction(
            version_id=version_id,
            action_type=action_type,
            parameters=json.dumps(parameters)  # Convert parameters to a JSON string
        )
        db.add(new_action)
        db.commit()
        # After logging the action, update the session state and the history
        if "actions" in st.session_state:
            st.session_state.actions.append(new_action)
        else:
            st.session_state.actions = [new_action]
        st.rerun()

    # Handling each preprocessing action
    if action == "Scale Data":
        selected_columns = st.multiselect("Select Columns to Scale", filtered_data.columns)
        scaling_method = st.selectbox("Select Scaling Method", ["StandardScaler", "MinMaxScaler"])
        if st.button("Scale Data"):
            scaler = StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
            filtered_data[selected_columns] = scaler.fit_transform(filtered_data[selected_columns])
            st.write(f"Scaled columns: {', '.join(selected_columns)} using {scaling_method}")
            log_action(selected_version.id, "Scale Data", {"columns": selected_columns, "method": scaling_method})

    elif action == "Encode Categorical Variables":
        selected_columns = st.multiselect("Select Columns to Encode", filtered_data.select_dtypes(include=['object']).columns)
        encoding_type = st.selectbox("Select Encoding Type", ["OneHotEncoding", "LabelEncoding"])
        if st.button("Encode Data"):
            if encoding_type == "OneHotEncoding":
                encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated to use sparse_output
                encoded_data = encoder.fit_transform(filtered_data[selected_columns])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(selected_columns))
                filtered_data.drop(columns=selected_columns, inplace=True)
                filtered_data = pd.concat([filtered_data, encoded_df], axis=1)
            else:
                encoder = LabelEncoder()
                for col in selected_columns:
                    filtered_data[col] = encoder.fit_transform(filtered_data[col])
            st.write(f"Encoded columns: {', '.join(selected_columns)} using {encoding_type}")
            log_action(selected_version.id, "Encode Data", {"columns": selected_columns, "type": encoding_type})

    elif action == "Impute Missing Values":
        selected_columns = st.multiselect("Select Columns to Impute", filtered_data.columns)
        impute_method = st.selectbox("Select Imputation Method", ["Mean", "Median", "Mode"])
        if st.button("Impute Missing Values"):
            for col in selected_columns:
                if impute_method == "Mean":
                    filtered_data[col].fillna(filtered_data[col].mean(), inplace=True)
                elif impute_method == "Median":
                    filtered_data[col].fillna(filtered_data[col].median(), inplace=True)
                elif impute_method == "Mode":
                    filtered_data[col].fillna(filtered_data[col].mode()[0], inplace=True)
            st.write(f"Imputed missing values in columns: {', '.join(selected_columns)} using {impute_method}")
            log_action(selected_version.id, "Impute Missing Values", {"columns": selected_columns, "method": impute_method})

    elif action == "Remove Outliers":
        selected_column = st.selectbox("Select Column to Remove Outliers", filtered_data.columns)
        method = st.selectbox("Select Outlier Removal Method", ["IQR Method", "Z-Score Method"])
        if st.button("Remove Outliers"):
            if method == "IQR Method":
                Q1 = filtered_data[selected_column].quantile(0.25)
                Q3 = filtered_data[selected_column].quantile(0.75)
                IQR = Q3 - Q1
                filtered_data = filtered_data[~((filtered_data[selected_column] < (Q1 - 1.5 * IQR)) | (filtered_data[selected_column] > (Q3 + 1.5 * IQR)))]
            elif method == "Z-Score Method":
                filtered_data = filtered_data[(zscore(filtered_data[selected_column]).abs() < 3)]
            st.write(f"Removed outliers from column {selected_column} using {method}")
            log_action(selected_version.id, "Remove Outliers", {"column": selected_column, "method": method})

    st.session_state.original_data = filtered_data
