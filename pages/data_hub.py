import streamlit as st
from data_utils import *
from data_analysis import *
from data_hub_tabs.tab_funcs import *

from machine_learning.utils import *

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

from llm.utils import suggest_models, explain_predictions, suggest_target_column, explain_feature_importance_commentary, explain_insights_commentary, explain_predictions_commentary, generate_leaderboard_commentary
from machine_learning.model_mapping import MODEL_MAPPING

from llm.utils import get_llm_response
# Set page config for dark mode
st.set_page_config(
    page_title="AnalytiQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

def handle_ml_tab(filtered_data):
    """Handles all content and logic within the Machine Learning Tab."""
    st.header("Machine Learning Assistant")

    # Initialize session state variables if they don't exist
    if 'aml' not in st.session_state:
        st.session_state.aml = None
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    if 'selected_model_id' not in st.session_state:
        st.session_state.selected_model_id = None
    if 'explanation' not in st.session_state:
        st.session_state.explanation = None
    if 'actual_values' not in st.session_state:
        st.session_state.actual_values = None

    with st.container():
        st.subheader("1. Explain Your Use Case")
        use_case = st.text_area("Describe your use case", placeholder="E.g., I want to predict house prices based on various features.")

        st.subheader("2. Select Your Task")
        task = st.selectbox("What do you want to do?", ["Classification", "Regression", "Clustering", "Anomaly Detection", "Dimensionality Reduction", "Time Series"])

        st.subheader("3. Get Algorithm Suggestions")
        if st.button("Get Suggestions"):
            if use_case:
                st.info("Sending your data and use case to the LLM for algorithm suggestions...")
                suggested_algorithms = suggest_models(use_case, task.lower(), filtered_data.head(), generate_summary(filtered_data), detailed_statistics(filtered_data))
                if suggested_algorithms:
                    st.session_state.suggested_algorithms = suggested_algorithms
                    st.success(f"Suggested Algorithms: {', '.join(suggested_algorithms)}")
                else:
                    st.warning("No suggestions received. Please check your use case description.")
            else:
                st.error("Please describe your use case before getting suggestions.")

        st.subheader("4. Select Algorithms to Use")
        selected_algorithms = st.multiselect(
            "Select the algorithms you want to run:",
            options=list(MODEL_MAPPING.get(task.lower(), {}).keys()),
            default=st.session_state.get('suggested_algorithms', [])
        )

        if task in ["Classification", "Regression", "Time Series"]:
            st.subheader("5. Get Target Column Suggestion")
            llm_commentary = None
            if st.button("Get Suggested Target Column"):
                try:
                    llm_commentary = suggest_target_column(
                        task,
                        filtered_data.columns,
                        use_case,
                        filtered_data.head(),
                        generate_summary(filtered_data),
                        detailed_statistics(filtered_data)
                    )
                    st.session_state.target_column = llm_commentary
                    st.success(f"Suggested Target Column: {llm_commentary}")
                except ValueError as e:
                    st.error(str(e))
            
            st.session_state.target_column = st.selectbox(
                "Select Target Column",
                filtered_data.columns
            )

        st.subheader("6. Model Comparison and Training")
        if st.button("Run Selected Models"):
            if selected_algorithms:
                st.info(f"Running the following models: {', '.join(selected_algorithms)}")
                if 'target_column' in st.session_state:
                    st.session_state.aml, st.session_state.test_data = run_h2o_automl(filtered_data, st.session_state.target_column, task.lower(), selected_algorithms)
                else:
                    st.error("Target column must be selected for this task.")
                    return
                
                st.write("AutoML completed. Model leaderboard:")
                leaderboard = st.session_state.aml.leaderboard
                st.dataframe(leaderboard.as_data_frame().style.highlight_max(axis=0))
                
                # Generate and display leaderboard commentary
                leaderboard_commentary = generate_leaderboard_commentary(use_case, filtered_data.head(), selected_algorithms, leaderboard.as_data_frame())
                st.markdown("### Leaderboard Commentary")
                st.markdown(leaderboard_commentary)

            else:
                st.error("Please select at least one algorithm to run.")

        # Display tabs for Post-Leaderboard Analysis after running a selected model
        if st.session_state.aml is not None:
            leaderboard = st.session_state.aml.leaderboard
            model_ids = leaderboard['model_id'].as_data_frame().values.flatten().tolist()
            st.session_state.selected_model_id = st.selectbox("Select a model to test", model_ids)
            
            test_data_option = st.radio("Choose test data", ["Use same data", "Upload new test data"])
            if test_data_option == "Upload new test data":
                test_file = st.file_uploader("Choose a CSV file for testing", type="csv")
                if test_file is not None:
                    test_data = pd.read_csv(test_file)
                    st.session_state.test_data = h2o.H2OFrame(test_data)
                    st.session_state.actual_values = test_data[st.session_state.target_column]

            if st.button("Run selected model"):
                run_selected_model()

            if st.session_state.predictions is not None:
                predictions_with_actuals = st.session_state.test_data.cbind(st.session_state.predictions)
                # Display the tabs for further analysis
                tabs = st.tabs(["Model Evaluation", "Feature Importance", "Business Insights"])

                with tabs[0]:
                    st.write("### Model Evaluation")
                    # Merge the predicted outcomes with the actual data
                    st.write("Actual vs Predicted:")
                    st.dataframe(predictions_with_actuals.as_data_frame())
                    
                    # Generate and display predictions commentary
                    predictions_commentary = explain_predictions_commentary(predictions_with_actuals.as_data_frame(), st.session_state.actual_values)
                    st.markdown("### Predictions Commentary")
                    st.markdown(predictions_commentary)

                with tabs[1]:
                    if st.session_state.feature_importance is not None:
                        st.write("### Feature Importance:")
                        st.dataframe(st.session_state.feature_importance)
                        
                        # Generate and display feature importance commentary
                        feature_importance_commentary = explain_feature_importance_commentary(st.session_state.feature_importance)
                        st.markdown("### Feature Importance Commentary")
                        st.markdown(feature_importance_commentary)

                with tabs[2]:
                    st.write("### Business Insights")
                    # Generate and display overall insights commentary
                    insights_commentary = explain_insights_commentary(predictions_with_actuals.as_data_frame(), st.session_state.feature_importance)
                    st.markdown(insights_commentary)

                if st.button("Download model"):
                    selected_model = h2o.get_model(st.session_state.selected_model_id)
                    model_path = h2o.download_model(selected_model, path=".")
                    with open(model_path, "rb") as f:
                        bytes_data = f.read()
                    st.download_button(
                        label="Download PKL file",
                        data=bytes_data,
                        file_name=f"{st.session_state.selected_model_id}.pkl",
                        mime="application/octet-stream"
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
