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

from llm.utils import suggest_models, explain_predictions
from machine_learning.model_mapping import MODEL_MAPPING

# Set page config for dark mode
st.set_page_config(
    page_title="AnalytiQ",
    layout="wide",
    initial_sidebar_state="expanded"
)


def handle_ml_tab(filtered_data):
    """Handles all content and logic within the Machine Learning Tab."""
    st.header("Machine Learning Assistant")

    with st.container():
        st.subheader("1. Explain Your Use Case")
        use_case = st.text_area("Describe your use case", placeholder="E.g., I want to predict house prices based on various features.")

        st.subheader("2. Select Your Task")
        task = st.selectbox("What do you want to do?", ["Classification", "Regression", "Clustering", "Anomaly Detection", "Dimensionality Reduction", "Time Series"])

        # Initialize an empty list for suggestions and available models
        suggested_algorithms = []
        available_algorithms = MODEL_MAPPING.get(task.lower(), {})

        # Generate data summary and detailed statistics
        summary = generate_summary(filtered_data)
        detailed_stats = detailed_statistics(filtered_data)
        data_head = filtered_data.head()

        st.subheader("3. Get Algorithm Suggestions")
        if st.button("Get Suggestions"):
            if use_case:
                st.info("Sending your data and use case to the LLM for algorithm suggestions...")
                suggested_algorithms = suggest_models(use_case, task.lower(), data_head, summary, detailed_stats)
                if suggested_algorithms:
                    st.success(f"Suggested Algorithms: {', '.join(suggested_algorithms)}")
                else:
                    st.warning("No suggestions received. Please check your use case description.")
            else:
                st.error("Please describe your use case before getting suggestions.")

        st.subheader("4. Select Algorithms to Use")
        selected_algorithms = st.multiselect(
            "Select the algorithms you want to run:",
            options=list(available_algorithms.keys()),
            default=suggested_algorithms
        )

        st.subheader("5. Model Comparison and Training")
        st.write("After you get the algorithm suggestions, you can compare models and train the best one.")

        if st.button("Run Selected Models"):
            if selected_algorithms:
                st.info(f"Running the following models: {', '.join(selected_algorithms)}")
                # Here you would implement the logic to run the selected models
            else:
                st.error("Please select at least one algorithm to run.")

        st.warning("Model comparison and training will be implemented in the next steps.")

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
        tabs = st.tabs(["Summary", "Data Quality", "Analysis", "Data Manipulation", "Preprocessing", "Machine Learning"])

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
        with tabs[5]:
            handle_ml_tab(st.session_state.filtered_data)


        st.write(f"Displaying first {data_limit} rows of {dataset_name}")
        st.dataframe(st.session_state.filtered_data, use_container_width=True)

if __name__ == "__main__":
    main()
