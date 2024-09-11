import os
import streamlit as st
import polars as pl
from db_utils import add_dataset
from sqlalchemy.orm import Session
from models import Dataset, get_db
from polars_datatypes import DATA_TYPE_OPTIONS

# Constants
DATASETS_DIR = "datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)


def upload_file():
    """Handles file upload and returns the file path."""
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file:
        file_path = os.path.join(DATASETS_DIR, uploaded_file.name)
        return file_path, uploaded_file
    return None, None

def parse_csv(uploaded_file, infer_schema_length, ignore_errors, null_values):
    """Reads the CSV file and returns a Polars DataFrame."""
    try:
        return pl.read_csv(
            uploaded_file,
            infer_schema_length=infer_schema_length,
            ignore_errors=ignore_errors,
            null_values=null_values
        )
    except Exception as e:
        st.error("Please **ReUpload** the csv file and Try using **Advanced Options**")
        st.error(f"Error parsing CSV: {e}")
        return None

def display_column_types(df):
    """Displays inferred column types."""
    column_types = {col: str(dtype) for col, dtype in df.schema.items()}
    st.write("**Inferred Column Types:**")
    st.write(column_types)
    return column_types

def select_columns(column_types):
    """Allows the user to select columns and their data types."""
    columns_to_select = st.multiselect("Select columns to keep", options=list(column_types.keys()), default=list(column_types.keys()))
    new_column_types = {}
    for col in columns_to_select:
        selected_dtype = st.selectbox(
            f"Select data type for '{col}'",
            options=list(DATA_TYPE_OPTIONS.keys()),
            index=list(DATA_TYPE_OPTIONS.keys()).index(column_types[col])
        )
        new_column_types[col] = DATA_TYPE_OPTIONS[selected_dtype]
    return columns_to_select, new_column_types

def display_data_types(original_types, updated_types):
    """Displays original and updated column types."""
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Column Types:**")
        st.write(original_types)
    with col2:
        st.write("**Updated Column Types:**")
        st.write(updated_types)

def apply_changes(df, columns_to_select, new_column_types, file_path):
    """Applies data type changes and saves the DataFrame as a Parquet file."""
    try:
        filtered_df = df.select(columns_to_select)
        updated_df = filtered_df.with_columns([
            pl.col(col).cast(new_column_types[col])
            for col in columns_to_select
        ])
        st.write("Creating the dataframe using the following schema")
        st.write(updated_df.schema)
        new_file_path = file_path.replace('.csv', '.parquet')
        updated_df.write_parquet(new_file_path)
        return new_file_path
    except Exception as e:
        st.error(f"Error casting columns: {e}")
        return None

def add_to_database(name, description, file_path):
    """Adds dataset info to the database."""
    try:
        db: Session = next(get_db())
        dataset_info = add_dataset(name, description, file_path)
        st.success(f"Dataset '{dataset_info['name']}' added successfully with a default version!")
        st.write("File saved at:", file_path)
    except Exception as e:
        st.error(f"Error adding dataset to database: {e}")

def display_existing_datasets():
    """Displays existing datasets from the database."""
    db: Session = next(get_db())
    datasets = db.query(Dataset).all()

    if datasets:
        st.subheader("Existing Datasets")
        df = pl.DataFrame([{
            'Name': dataset.name,
            'Description': dataset.description,
            'File Path': dataset.filepath
        } for dataset in datasets])
        st.dataframe(df)
    else:
        st.write("No datasets available.")

def main():
    st.title("Manage Datasets")

    st.write("Upload a file to create a new dataset.")

    # Form fields
    name = st.text_input("Dataset Name", value="")
    description = st.text_area("Description", value="")
    show_advanced = st.checkbox("Show Advanced Options")
    infer_schema_length = st.number_input("Infer Schema Length", min_value=1, value=10000, step=1000) if show_advanced else 1000
    ignore_errors = st.checkbox("Ignore Errors", value=False) if show_advanced else False
    null_values_input = st.text_input("Null Values (comma-separated)", value="") if show_advanced else ""
    
    file_path, uploaded_file = upload_file()
    
    if uploaded_file:
        null_values = null_values_input.split(',') if null_values_input else []
        df = None
        df = parse_csv(uploaded_file, infer_schema_length, ignore_errors, null_values)

        if df is not None:
            st.dataframe(df)
            column_types = display_column_types(df)
            columns_to_select, new_column_types = select_columns(column_types)
            
            if columns_to_select:
                updated_data_types = {col: str(dtype) for col, dtype in new_column_types.items()}
                display_data_types(column_types, updated_data_types)

                if st.button("Confirm and Apply Changes"):
                    new_file_path = apply_changes(df, columns_to_select, new_column_types, file_path)
                    if new_file_path:
                        add_to_database(name, description, new_file_path)
            else:
                st.write("No columns selected.")
    else:
        st.write("No file uploaded.")
    
    display_existing_datasets()

if __name__ == "__main__":
    main()
