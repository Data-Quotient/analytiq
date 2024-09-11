import json
import streamlit as st
import polars as pl

import pandas as pd

from models import get_db, DatasetOperation, Dataset, DatasetVersion  # Import the Dataset model and database session
from sqlalchemy.orm import Session

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from scipy.stats import zscore
from polars_datatypes import DATA_TYPE_OPTIONS
from data_utils import load_data, apply_operations_to_dataset

def log_operation(version_id, operation_type, parameters):
    """Logs the operation to the database."""
    new_operation = DatasetOperation(
        version_id=version_id,
        operation_type=operation_type,
        parameters=json.dumps(parameters)  # Convert parameters to a JSON string
    )
    db: Session = next(get_db())

    db.add(new_operation)
    db.commit()
    # After logging the operation, update the session state and the history
    if "operations" in st.session_state:
        st.session_state.operations.append(new_operation)
    else:
        st.session_state.operations = [new_operation]
    st.rerun()

# Function to handle the Data Manipulation Tab
def handle_data_manipulation_tab(filtered_data: pl.DataFrame, selected_version):
    """Handles all content and logic within the Data Manipulation Tab."""
    st.header("Data Manipulation")
    
    # Dropdown to select a manipulation operation
    operation = st.selectbox(
        "Select an Operation",
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
    # Handling each operation
    if operation == "Rename Column":
        selected_column = st.selectbox("Select Column to Rename", filtered_data.columns)
        new_column_name = st.text_input("Enter New Column Name")
        if st.button("Rename Column"):
            filtered_data = filtered_data.rename({selected_column: new_column_name})
            st.write(f"Renamed column {selected_column} to {new_column_name}")
            log_operation(selected_version.id, "Rename Column", {"old_name": selected_column, "new_name": new_column_name})

    elif operation == "Change Data Type":
        selected_column = st.selectbox("Select Column to Change Data Type", filtered_data.columns)
        col_dtype = filtered_data.schema[selected_column]
        index = list(DATA_TYPE_OPTIONS.keys()).index(str(col_dtype)) if str(col_dtype) in DATA_TYPE_OPTIONS.keys() else 0
        new_data_type = st.selectbox("Select New Data Type", DATA_TYPE_OPTIONS.keys(), index=index)
        if st.button("Change Data Type"):
            try:
                filtered_data = filtered_data.with_columns([
                    pl.col(selected_column).cast(DATA_TYPE_OPTIONS[new_data_type])
                ])
                st.write(f"Changed data type of column {selected_column} to {new_data_type}")
                log_operation(selected_version.id, "Change Data Type", {"column": selected_column, "new_type": new_data_type})
            except Exception as e:
                st.error(f"Error changing data type: {e}")
            

    elif operation == "Delete Column":
        selected_columns = st.multiselect("Select Columns to Delete", filtered_data.columns)
        if st.button("Delete Columns"):
            filtered_data = filtered_data.drop(selected_columns)
            st.write(f"Deleted columns: {', '.join(selected_columns)}")
            log_operation(selected_version.id, "Delete Column", {"columns": selected_columns})

    elif operation == "Filter Rows": # TODO: Fix this
        filter_condition = st.text_input("Enter Filter Condition (e.g., `age >= 18`)")
        if st.button("Apply Filter"):
            try:
                filtered_data = filtered_data.filter(pl.col(filter_condition))
                st.write(f"Applied filter: {filter_condition}")
                log_operation(selected_version.id, "Filter Rows", {"condition": filter_condition})
            except Exception as e:
                st.error(f"Error applying filter: {e}")
    elif operation == "Add Calculated Column": # TODO: Fix this
        new_column_name = st.text_input("Enter New Column Name")
        formula = st.text_input("Enter Formula (e.g., `quantity * price`)")
        if st.button("Add Calculated Column"):
            try:
                filtered_data = filtered_data.with_columns([
                    pl.eval(pl.col(formula)).alias(new_column_name)
                ])
                st.write(f"Added calculated column {new_column_name} with formula: {formula}")
                log_operation(selected_version.id, "Add Calculated Column", {"new_column": new_column_name, "formula": formula})
            except Exception as e:
                st.error(f"Error adding calculated column: {e}")
    elif operation == "Fill Missing Values":
        selected_column = st.selectbox("Select Column to Fill Missing Values", filtered_data.columns)
        fill_method = st.selectbox("Select Fill Method", ["Specific Value", "Mean", "Median", "Mode"])
        fill_value = st.text_input("Enter Value (if 'Specific Value' selected)")
        if st.button("Fill Missing Values"):
            if fill_method == "Specific Value":
                filtered_data = filtered_data.with_columns(
                    pl.col(selected_column).fill_null(fill_value)
                )
                log_operation(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method, "value": fill_value})
            elif fill_method == "Mean":
                mean_value = filtered_data[pl.col(selected_column)].mean()
                filtered_data = filtered_data.fill_none(mean_value)
                log_action(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method})
                filtered_data[selected_column].fillna(filtered_data[selected_column].mean(), inplace=True)
                log_operation(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method})
            elif fill_method == "Median":
                median_value = filtered_data[pl.col(selected_column)].median()
                filtered_data = filtered_data.fill_none(median_value)
                log_action(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method})
                filtered_data[selected_column].fillna(filtered_data[selected_column].median(), inplace=True)
                log_operation(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method})
            elif fill_method == "Mode":
                filtered_data[selected_column].fillna(filtered_data[selected_column].mode()[0], inplace=True)
                log_operation(selected_version.id, "Fill Missing Values", {"column": selected_column, "method": fill_method})
            st.write(f"Filled missing values in column {selected_column} using method: {fill_method}")

    elif operation == "Duplicate Column":
        selected_column = st.selectbox("Select Column to Duplicate", filtered_data.columns)
        if st.button("Duplicate Column"):
            filtered_data = filtered_data.with_columns([
                pl.col(selected_column).alias(f"{selected_column}_duplicate")
            ])
            st.write(f"Duplicated column: {selected_column}")
            log_operation(selected_version.id, "Duplicate Column", {"column": selected_column})

    elif operation == "Reorder Columns":
        new_order = st.multiselect("Select Columns in New Order", filtered_data.columns, default=list(filtered_data.columns))
        if st.button("Reorder Columns"):
            filtered_data = filtered_data.select(new_order)
            st.write(f"Reordered columns to: {', '.join(new_order)}")
            log_operation(selected_version.id, "Reorder Columns", {"new_order": new_order})
    elif operation == "Replace Values":
        selected_column = st.selectbox("Select Column to Replace Values", filtered_data.columns)
        to_replace = st.text_input("Value to Replace")
        replace_with = st.text_input("Replace With")
        if st.button("Replace Values"):
            filtered_data = filtered_data.with_columns([
                pl.col(selected_column).replace(to_replace, replace_with)
            ])
            st.write(f"Replaced {to_replace} with {replace_with} in column {selected_column}")
            log_operation(selected_version.id, "Replace Values", {"column": selected_column, "to_replace": to_replace, "replace_with": replace_with})

    st.session_state.original_data = filtered_data

# Function to handle the Preprocessing Tab
def handle_preprocessing_tab(filtered_data: pl.DataFrame, selected_version):
    """Handles all content and logic within the Preprocessing Tab."""
    st.header("Data Preprocessing")

    # Dropdown to select a preprocessing operation
    operation = st.selectbox(
        "Select a Preprocessing Operation",
        [
            "Scale Data",
            "Encode Categorical Variables",
            "Impute Missing Values",
            "Remove Outliers"
        ]
    )
    # Handling each preprocessing operation
    if operation == "Scale Data":
        selected_columns = st.multiselect("Select Columns to Scale", filtered_data.columns)
        scaling_method = st.selectbox("Select Scaling Method", ["StandardScaler", "MinMaxScaler"])
        if st.button("Scale Data"):
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            # Convert to pandas DataFrame for scaling
            filtered_data_pd = filtered_data.to_pandas()
            filtered_data_pd[selected_columns] = scaler.fit_transform(filtered_data_pd[selected_columns])
            filtered_data = pl.from_pandas(filtered_data_pd)
            st.write(f"Scaled columns: {', '.join(selected_columns)} using {scaling_method}")
            log_operation(selected_version.id, "Scale Data", {"columns": selected_columns, "method": scaling_method})

    elif operation == "Encode Categorical Variables":
        selected_columns = st.multiselect("Select Columns to Encode", filtered_data.select_dtypes(include=['object']).columns)
        encoding_type = st.selectbox("Select Encoding Type", ["OneHotEncoding", "LabelEncoding"])
        if st.button("Encode Data"):
            if encoding_type == "OneHotEncoding":
                filtered_data = filtered_data.to_pandas()
                for col in selected_columns:
                    filtered_data = pd.get_dummies(filtered_data, columns=[col], drop_first=True)
                filtered_data = pl.from_pandas(filtered_data)
            else:
                encoder = LabelEncoder()
                filtered_data = filtered_data.with_columns([
                    pl.col(col).apply(lambda x: encoder.fit_transform(x), return_dtype=pl.Int64).alias(col) for col in selected_columns
                ])
            st.write(f"Encoded columns: {', '.join(selected_columns)} using {encoding_type}")
            log_operation(selected_version.id, "Encode Data", {"columns": selected_columns, "type": encoding_type})

    elif operation == "Impute Missing Values":
        selected_columns = st.multiselect("Select Columns to Impute", filtered_data.columns)
        impute_method = st.selectbox("Select Imputation Method", ["Mean", "Median", "Mode"])
        if st.button("Impute Missing Values"):
            for col in selected_columns:
                if impute_method == "Mean":
                    mean_value = filtered_data[col].mean()
                    filtered_data = filtered_data.with_columns([
                        pl.col(col).fill_null(mean_value)
                    ])
                elif impute_method == "Median":
                    median_value = filtered_data[col].median()
                    filtered_data = filtered_data.with_columns([
                        pl.col(col).fill_null(median_value)
                    ])
                elif impute_method == "Mode":
                    mode_value = filtered_data[col].mode()[0]
                    filtered_data = filtered_data.with_columns([
                        pl.col(col).fill_null(mode_value)
                    ])
                
            st.write(f"Imputed missing values in columns: {', '.join(selected_columns)} using {impute_method}")
            log_operation(selected_version.id, "Impute Missing Values", {"columns": selected_columns, "method": impute_method})

    elif operation == "Remove Outliers":
        selected_column = st.selectbox("Select Column to Remove Outliers", filtered_data.columns)
        method = st.selectbox("Select Outlier Removal Method", ["IQR Method", "Z-Score Method"])
        if st.button("Remove Outliers"):
            if method == "IQR Method":
                Q1 = filtered_data[selected_column].quantile(0.25)
                Q3 = filtered_data[selected_column].quantile(0.75)
                IQR = Q3 - Q1
                filtered_data = filtered_data.filter(
                    (pl.col(selected_column) >= (Q1 - 1.5 * IQR)) & (pl.col(selected_column) <= (Q3 + 1.5 * IQR))
                )
            elif method == "Z-Score Method":
                z_scores = zscore(filtered_data[selected_column].to_pandas())
                filtered_data = filtered_data.filter(pl.Series(z_scores).abs() < 3)
            st.write(f"Removed outliers from column {selected_column} using {method}")
            log_operation(selected_version.id, "Remove Outliers", {"column": selected_column, "method": method})

    st.session_state.original_data = filtered_data

def handle_merge_datasets_tab(current_dataset, original_version):
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

    # Apply operations recorded for the selected version
    operations = db.query(DatasetOperation).filter(DatasetOperation.version_id == selected_version.id).all()
    if operations:
        selected_data = apply_operations_to_dataset(selected_data, operations)

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
            st.dataframe(st.session_state.merged_data.to_pandas())

    # Merge button
    if "merged_data" in st.session_state and st.session_state.merged_data is not None:
        if st.button("Merge"):
            # Log the merge operation
            log_operation(original_version.id, "Merge Datasets", {
                "merge_with": selected_dataset.id,
                "merge_version": selected_version.id,
                "join_column": merge_column,
                "join_type": join_type
            })

            st.success(f"Merged dataset updated in the current version '{selected_version.version_number}'.")

def merge_datasets(active_data: pl.DataFrame, dataset_id, version_name, merge_column, join_type):
    db: Session = next(get_db())

    # Fetch the correct version of the dataset
    selected_version = db.query(DatasetVersion).filter(
        DatasetVersion.version_number == version_name,
        DatasetVersion.dataset_id == dataset_id
    ).first()

    # Load the dataset for the selected version
    data_path = selected_version.dataset.filepath
    data = load_data(data_path)  # Load the raw data

    # Apply operations recorded for the selected version
    operations = db.query(DatasetOperation).filter(DatasetOperation.version_id == selected_version.id).all()

    if operations:
        data = apply_operations_to_dataset(data, operations)  # Apply all recorded operations to get the manipulated data

    try:
        # Perform the merge between the active dataset and the selected dataset version
        merged_data = active_data.join(data, on=merge_column, how=join_type)
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
