import json
import streamlit as st

import pandas as pd

from models import get_db, DatasetAction, Dataset, DatasetVersion  # Import the Dataset model and database session
from sqlalchemy.orm import Session

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from scipy.stats import zscore

from data_utils import load_data, apply_actions_to_dataset

def log_action(version_id, action_type, parameters):
    """Logs the action to the database."""
    new_action = DatasetAction(
        version_id=version_id,
        action_type=action_type,
        parameters=json.dumps(parameters)  # Convert parameters to a JSON string
    )
    db: Session = next(get_db())

    db.add(new_action)
    db.commit()
    # After logging the action, update the session state and the history
    if "actions" in st.session_state:
        st.session_state.actions.append(new_action)
    else:
        st.session_state.actions = [new_action]
    st.rerun()

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

def handle_merge_datasets_tab(current_dataset, orignal_version):
    st.header("Merge Datasets")

    # Fetch datasets from the database
    db: Session = next(get_db())
    datasets = db.query(Dataset).all()

    if not datasets:
        st.write("No datasets available. Please upload a dataset first.")
        return

    dataset_names = [dataset.name for dataset in datasets]

    # Dropdown to select one additional dataset for merging with the active dataset
    dataset_selection = st.selectbox("Select Dataset to Merge With", dataset_names)

    if not dataset_selection:
        st.warning("Please select a dataset to merge.")
        return

    selected_dataset = db.query(Dataset).filter(Dataset.name == dataset_selection).first()
    versions = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == selected_dataset.id).all()
    version_names = [version.version_number for version in versions]
    selected_version_name = st.selectbox(f"Select Version for {dataset_selection}", version_names)

    # Load the active dataset columns
    active_columns = st.session_state.original_data.columns

    # Load the dataset for the selected version
    selected_version = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == selected_dataset.id,
        DatasetVersion.version_number == selected_version_name
    ).first()
    selected_data = load_data(selected_version.dataset.filepath)

    # Apply actions recorded for the selected version
    actions = db.query(DatasetAction).filter(DatasetAction.version_id == selected_version.id).all()
    if actions:
        selected_data = apply_actions_to_dataset(selected_data, actions)

    selected_columns = selected_data.columns

    # Find common columns for merging
    common_columns = list(set(active_columns).intersection(set(selected_columns)))

    if not common_columns:
        st.warning("No common columns available for merging.")
        return

    # Select join type
    join_type = st.selectbox("Select Join Type", ["inner", "left", "right", "outer"])

    # Dropdown to select the column to merge on
    merge_column = st.selectbox("Select the column to merge on", options=common_columns)

    # Preview button
    if merge_column and st.button("Preview Merged Dataset"):
        st.session_state.merged_data = merge_datasets(
            current_dataset,
            selected_dataset.id,
            selected_version_name,
            merge_column,
            join_type
        )
        if st.session_state.merged_data is not None:
            st.write("Merged Dataset Preview")
            st.dataframe(st.session_state.merged_data)

    # Merge button
    if "merged_data" in st.session_state and st.session_state.merged_data is not None:
        if st.button("Merge"):
            # Log the merge action
            log_action(orignal_version.id, "Merge Datasets", {
                "merge_with": selected_dataset.id,
                "merge_version": selected_version.id,
                "join_column": merge_column,
                "join_type": join_type
            })

            st.success(f"Merged dataset updated in the current version '{selected_version.version_number}'.")

def merge_datasets(active_data, dataset_id, version_name, merge_column, join_type):
    db: Session = next(get_db())

    # Fetch the correct version of the dataset
    selected_version = db.query(DatasetVersion).filter(
        DatasetVersion.version_number == version_name,
        DatasetVersion.dataset_id == dataset_id  # Ensure the correct dataset is selected
    ).first()

    # Load the dataset for the selected version
    data_path = selected_version.dataset.filepath
    data = load_data(data_path)  # Load the raw data

    # Apply actions recorded for the selected version
    actions = db.query(DatasetAction).filter(DatasetAction.version_id == selected_version.id).all()

    if actions:
        data = apply_actions_to_dataset(data, actions)  # Apply all recorded actions to get the manipulated data

    try:
        # Perform the merge between the active dataset and the selected dataset version
        merged_data = pd.merge(active_data, data, on=merge_column, how=join_type)
    except KeyError as e:
        error_msg = (
            f"Merge failed due to missing column: {e.args[0]}.\n"
            f"Ensure that both datasets have the column '{merge_column}' available.\n"
            f"Available columns in active dataset: {list(active_data.columns)}\n"
            f"Available columns in selected dataset: {list(data.columns)}"
        )
        st.error(error_msg)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during the merge: {str(e)}")
        return None

    return merged_data
