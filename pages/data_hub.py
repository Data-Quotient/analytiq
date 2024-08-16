import streamlit as st
from data_utils import *
from data_analysis import *
from data_hub_tabs.tab_funcs import *

from models import get_db, Dataset, DatasetVersion, DatasetAction  # Import the models
from sqlalchemy.orm import Session
import time
from datetime import datetime
import json

# Try to import Plotly and install if not available
try:
    import plotly.express as px
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.express as px

# Set page config for dark mode
st.set_page_config(
    page_title="AnalytiQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_actions_to_dataset(dataset, actions):
    """
    Apply a list of actions to a dataset.
    
    Args:
        dataset (DataFrame): The dataset to which actions will be applied.
        actions (list): List of DatasetAction objects.
        
    Returns:
        DataFrame: The dataset after all actions have been applied.
    """
    for action in actions:
        action_type = action.action_type
        parameters = json.loads(action.parameters)  # Decode JSON string to dictionary
        
        if action_type == "Rename Column":
            dataset.rename(columns={parameters["old_name"]: parameters["new_name"]}, inplace=True)
        
        elif action_type == "Change Data Type":
            dataset[parameters["column"]] = dataset[parameters["column"]].astype(parameters["new_type"])
        
        elif action_type == "Delete Column":
            dataset.drop(columns=parameters["columns"], inplace=True)
        
        elif action_type == "Filter Rows":
            dataset = dataset.query(parameters["condition"])
        
        elif action_type == "Add Calculated Column":
            dataset[parameters["new_column"]] = eval(parameters["formula"], {'__builtins__': None}, dataset)
        
        elif action_type == "Fill Missing Values":
            if parameters["method"] == "Specific Value":
                dataset[parameters["column"]].fillna(parameters["value"], inplace=True)
            elif parameters["method"] == "Mean":
                dataset[parameters["column"]].fillna(dataset[parameters["column"]].mean(), inplace=True)
            elif parameters["method"] == "Median":
                dataset[parameters["column"]].fillna(dataset[parameters["column"]].median(), inplace=True)
            elif parameters["method"] == "Mode":
                dataset[parameters["column"]].fillna(dataset[parameters["column"]].mode()[0], inplace=True)
        
        elif action_type == "Duplicate Column":
            dataset[f"{parameters['column']}_duplicate"] = dataset[parameters["column"]]
        
        elif action_type == "Reorder Columns":
            dataset = dataset[parameters["new_order"]]
        
        elif action_type == "Replace Values":
            dataset[parameters["column"]].replace(parameters["to_replace"], parameters["replace_with"], inplace=True)

    return dataset


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
        tabs = st.tabs(["Summary", "Data Quality", "Analysis", "Data Manipulation"])

        with tabs[0]:
            handle_data_summary_tab(st.session_state.filtered_data)

        with tabs[1]:
            handle_data_quality_tab(st.session_state.filtered_data, selected_dataset.id)

        with tabs[2]:
            handle_data_analysis_tab(st.session_state.filtered_data)

        with tabs[3]:
            handle_data_manipulation_tab(st.session_state.filtered_data, selected_version_obj)

        st.write(f"Displaying first {data_limit} rows of {dataset_name}")
        st.dataframe(st.session_state.filtered_data, use_container_width=True)

if __name__ == "__main__":
    main()
