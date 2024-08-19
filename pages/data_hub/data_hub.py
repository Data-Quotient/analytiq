import time
import json

import streamlit as st

from .tabs.ml_tab import handle_ml_tab
from .tabs.summary_tab import handle_data_summary_tab
from .tabs.data_analysis_tab import handle_data_analysis_tab
from .tabs.data_quality_tab import handle_data_quality_tab
from .tabs.manipulation_tabs import (
    handle_data_manipulation_tab, 
    handle_preprocessing_tab,
    handle_merge_datasets_tab,
)

from models import get_db, Dataset, DatasetVersion, DatasetAction  # Import the models
from sqlalchemy.orm import Session

from data_utils import load_data, apply_actions_to_dataset, apply_filters

# Set page config for dark mode
st.set_page_config(
    page_title="AnalytiQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        tabs = st.tabs(
            [
                "Summary", 
                "Data Quality", 
                "Analysis", 
                "Merge Datasets",
                "Data Manipulation", 
                "Preprocessing", 
                "Machine Learning", 
            ]
        )
        with tabs[0]:
            handle_data_summary_tab(st.session_state.filtered_data)

        with tabs[1]:
            handle_data_quality_tab(st.session_state.filtered_data, selected_dataset.id)

        with tabs[2]:
            handle_data_analysis_tab(st.session_state.filtered_data)

        with tabs[3]:
            handle_merge_datasets_tab(st.session_state.filtered_data, selected_version_obj)
        with tabs[4]:
            handle_data_manipulation_tab(st.session_state.filtered_data, selected_version_obj)
        with tabs[5]:
            handle_preprocessing_tab(st.session_state.filtered_data, selected_version_obj)
        with tabs[6]:
            handle_ml_tab(st.session_state.filtered_data)

        st.write(f"Displaying first {data_limit} rows of {dataset_name}")
        st.dataframe(st.session_state.filtered_data, use_container_width=True)

if __name__ == "__main__":
    main()
