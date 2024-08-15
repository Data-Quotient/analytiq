import streamlit as st
from data_utils import *
from data_analysis import *
from data_hub_tabs.tab_funcs import *

from models import get_db, Dataset, DatasetVersion  # Import the Dataset and DatasetVersion models
from sqlalchemy.orm import Session
import time

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

        data_path = selected_dataset.filepath

        with st.spinner(f"Loading {dataset_name}..."):
            selected_data = load_data(data_path, data_limit)

        # Store the loaded data in session state to persist changes
        if "filtered_data" not in st.session_state:
            st.session_state.filtered_data = selected_data

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

        # Tabs for different views (e.g., Data View, Analysis, etc.)
        tabs = st.tabs(["Summary", "Data Quality", "Analysis", "Data Manipulation"])

        with tabs[0]:
            # Collapsible filters section for Summary tab
            with st.sidebar.expander("Filters", expanded=False):
                filters = {}
                for column in st.session_state.filtered_data.columns:
                    unique_vals = st.session_state.filtered_data[column].unique()
                    if len(unique_vals) < 100:  # Only show filter options if there are less than 100 unique values
                        filters[column] = st.selectbox(f"Filter by {column}", options=[None] + list(unique_vals))
                st.session_state.filtered_data = apply_filters(st.session_state.filtered_data, filters)

            handle_data_summary_tab(st.session_state.filtered_data)

        with tabs[1]:
            handle_data_quality_tab(st.session_state.filtered_data, selected_dataset.id)

        with tabs[2]:
            handle_data_analysis_tab(st.session_state.filtered_data)

        with tabs[3]:
            handle_data_manipulation_tab(st.session_state.filtered_data, selected_version)

        st.write(f"Displaying first {data_limit} rows of {dataset_name}")
        st.dataframe(st.session_state.filtered_data, use_container_width=True)

if __name__ == "__main__":
    main()
