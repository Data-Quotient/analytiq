import streamlit as st
from data_utils import *
from data_analysis import *
from data_hub_tabs.tab_funcs import *

from models import get_db, Dataset, DatasetVersion, DatasetAction  # Import the models
from sqlalchemy.orm import Session
import time
from datetime import datetime
import json

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

# Set page config for dark mode
st.set_page_config(
    page_title="AnalytiQ",
    layout="wide",
    initial_sidebar_state="expanded"
)


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

# Main function
def main():
    st.title("AnalytiQ")

    # Fetch datasets from the database
    db: Session = next(get_db())
    datasets = db.query(Dataset).all()

    if not datasets:
        st.write("No datasets available. Please upload a dataset first.")
        return

    dataset_names = [dataset.name for dataset in datasets]

    # Sidebar for dataset selection, limit input, and filters
    st.sidebar.header("Select Dataset")
    dataset_name = st.sidebar.selectbox("Select Dataset", dataset_names)
    data_limit = st.sidebar.number_input("Number of Rows to Fetch", min_value=1, value=1000, step=1000)

    # Load the selected dataset with a loading spinner
    if dataset_name:

        selected_dataset = db.query(Dataset).filter(Dataset.name == dataset_name).first()

        # Select version for the selected dataset
        versions = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == selected_dataset.id).all()
        version_names = [version.version_number for version in versions]

        selected_version = st.sidebar.selectbox("Select Version", version_names)
        st.sidebar.write(f"Selected Version: {selected_version}")

        # Fetch the selected version object
        selected_version_obj = db.query(DatasetVersion).filter(
            DatasetVersion.dataset_id == selected_dataset.id,
            DatasetVersion.version_number == selected_version
        ).first()

        # Accordion for creating a new version
        with st.expander("Create New Version", expanded=False):
            st.write("Use the form below to create a new version of the dataset.")

            # Fetch existing versions for the selected dataset
            existing_versions = [v.version_number for v in selected_dataset.versions]

            # Input fields for the new version
            new_version_name = st.text_input("Version Name")
            new_version_description = st.text_area("Version Description")
            parent_version = st.selectbox("Base Version", existing_versions)

            if st.button("Create Version"):
                try:
                    # Create a new dataset version
                    new_version = DatasetVersion(
                        dataset_id=selected_dataset.id,
                        version_number=new_version_name,
                        description=new_version_description
                    )
                    db.add(new_version)
                    db.commit()

                    # Display success message as a flash message
                    st.success(f"Version '{new_version_name}' created successfully.")
                    time.sleep(3)  # Display the message for 3 seconds

                except Exception as e:
                    db.rollback()
                    st.error(f"Failed to create version: {e}")

        # Logic to delete action before fetching data
        with st.sidebar.expander("Action History", expanded=True):
            actions = db.query(DatasetAction).filter(DatasetAction.version_id == selected_version_obj.id).all()
            if actions:
                for action in actions:
                    with st.container():
                        st.write(f"**Action Type:** {action.action_type}")
                        st.write(f"**Parameters:** {json.dumps(json.loads(action.parameters), indent=2)}")
                        if st.button(f"Remove Action {action.id}", key=f"remove_{action.id}"):
                            try:
                                db.delete(action)
                                db.commit()
                                # Update the actions list after deletion
                                actions = [a for a in actions if a.id != action.id]
                                st.success("Action removed successfully.")
                                st.rerun()
                            except Exception as e:
                                db.rollback()
                                st.error(f"Failed to delete action: {e}")
            else:
                st.write("No actions applied to this version.")

        data_path = selected_dataset.filepath

        with st.spinner(f"Loading {dataset_name}..."):
            selected_data = load_data(data_path, data_limit)
            
        # Apply actions to the original data if any
        if actions:
            selected_data = apply_actions_to_dataset(selected_data, actions)
        
        st.session_state.original_data = selected_data
        st.session_state.unfiltered_data = selected_data.copy()  # Save a copy for filter options

        # Sidebar for filter options based on unfiltered data
        with st.sidebar.expander("Filters", expanded=False):
            filters = {}
            for column in st.session_state.unfiltered_data.columns:
                unique_vals = st.session_state.unfiltered_data[column].unique()
                if len(unique_vals) < 100:  # Only show filter options if there are less than 100 unique values
                    filters[column] = st.selectbox(f"Filter by {column}", options=[None] + list(unique_vals))

            # Apply filters to the original data
            if filters:
                st.session_state.filtered_data = apply_filters(st.session_state.original_data.copy(), filters)
            else:
                st.session_state.filtered_data = st.session_state.original_data.copy()

        # Tabs for different views (e.g., Data View, Analysis, etc.)
        tabs = st.tabs(["Summary", "Data Quality", "Analysis", "Data Manipulation", "Preprocessing"])

        with tabs[0]:
            handle_data_summary_tab(st.session_state.filtered_data)

        with tabs[1]:
            handle_data_quality_tab(st.session_state.filtered_data, selected_dataset.id)

        with tabs[2]:
            handle_data_analysis_tab(st.session_state.filtered_data)

        with tabs[3]:
            handle_data_manipulation_tab(st.session_state.filtered_data, selected_version_obj)
        with tabs[4]:
            handle_preprocessing_tab(st.session_state.filtered_data, selected_version_obj)


        st.write(f"Displaying first {data_limit} rows of {dataset_name}")
        st.dataframe(st.session_state.filtered_data, use_container_width=True)

if __name__ == "__main__":
    main()
