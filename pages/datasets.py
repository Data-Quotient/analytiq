import os
import streamlit as st
from streamlit.components.v1 import html
import polars as pl
from db_utils import add_dataset
from sqlalchemy.orm import Session
from models import Dataset, get_db
from streamlit_javascript import st_javascript
import time

# Ensure the datasets directory exists
DATASETS_DIR = "datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

DATA_TYPE_OPTIONS = {
    "Int8": pl.Int8,
    "Int16": pl.Int16,
    "Int32": pl.Int32,
    "Int64": pl.Int64,
    "UInt8": pl.UInt8,
    "UInt16": pl.UInt16,
    "UInt32": pl.UInt32,
    "UInt64": pl.UInt64,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "Utf8": pl.Utf8,
    "Boolean": pl.Boolean,
    "Date": pl.Date,
    "Datetime": pl.Datetime,
    "Time": pl.Time,
    "Duration": pl.Duration,
    "Categorical": pl.Categorical,
    "List": pl.List,
    "Object": pl.Object,
    "String": pl.String
}

def main():
    st.title("Manage Datasets")

    # Step 1: Upload File and Configure
    st.write("Upload a file to create a new dataset.")
    
    # Form fields
    name = st.text_input("Dataset Name", value="")
    description = st.text_area("Description", value="")
    show_advanced = st.checkbox("Show Advanced Options")
    if show_advanced:
        infer_schema_length = st.number_input("Infer Schema Length", min_value=1, value=10000, step=1000)
        ignore_errors = st.checkbox("Ignore Errors", value=False)
        null_values_input = st.text_input("Null Values (comma-separated)", value="")
    else:
        infer_schema_length = 1000
        ignore_errors = False
        null_values_input = ""
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    

    if uploaded_file:
        # Save the file to the datasets directory
        file_path = os.path.join(DATASETS_DIR, uploaded_file.name)
        # with open(file_path, "wb") as f:
        #     f.write(uploaded_file.getbuffer())

        # Parse the CSV with user-defined parameters or defaults
        null_values = null_values_input.split(',') if null_values_input else []
        df = None
        
        
        try:
            df = pl.read_csv(
                uploaded_file,
                infer_schema_length=infer_schema_length,
                ignore_errors=ignore_errors,
                null_values=null_values
            )

            # Step 2: Display Inferred Column Types
            column_types = {col: str(dtype) for col, dtype in df.schema.items()}

            # Step 3: Select Columns
            columns_to_select = st.multiselect("Select columns to keep", options=list(column_types.keys()), default=list(column_types.keys()))
            
            # Add a confirmation button after column selection
            if columns_to_select:
                # Step 4: Update Data Types
                new_column_types = {}
                for col in columns_to_select:
                    selected_dtype = st.selectbox(
                        f"Select data type for '{col}'",
                        options=list(DATA_TYPE_OPTIONS.keys()),
                        index=list(DATA_TYPE_OPTIONS.keys()).index(column_types[col])
                    )
                    new_column_types[col] = DATA_TYPE_OPTIONS[selected_dtype]

                updated_data_types = {col: str(dtype) for col, dtype in new_column_types.items()}

                # Create two columns for side-by-side layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Column Types:**")
                    st.write(column_types)
                    
                with col2:
                    st.write("**Updated Column Types:**")
                    updated_data_types = {col: str(dtype) for col, dtype in new_column_types.items()}
                    st.write(updated_data_types)

                if st.button("Confirm and Apply Changes"):
                    try:
                        updated_df = df.with_columns([
                            pl.col(col).cast(new_column_types[col])
                            for col in columns_to_select
                        ])
                        st.write("Creating the dataframe using the following schema")
                        st.write(updated_df.schema)
                        # Save the updated DataFrame as a Parquet file
                        new_file_path = os.path.join(DATASETS_DIR, uploaded_file.name.replace('.csv', '.parquet'))

                        updated_df.write_parquet(new_file_path)

                        # Add dataset to the database
                        db: Session = next(get_db())
                        dataset_info = add_dataset(name, description, new_file_path)

                        st.success(f"Dataset '{dataset_info['name']}' added successfully with a default version!")
                        st.write("File saved at:", new_file_path)
                    except Exception as e:
                        st.error(f"Error casting columns: {e}")

                        
            else:
                st.write("No columns selected.")

        except Exception as e:
            st.error("Please **REMOVE and RE-UPLOAD** the file")
            st.error(f"Unable to parse csv, Try using **Advanced Options**")
            st.error(f"Error: {str(e)}")
    # Display existing datasets
    db: Session = next(get_db())
    datasets = db.query(Dataset).all()

    if datasets:
        st.subheader("Existing Datasets")

        # Convert datasets to a DataFrame for better display
        df = pl.DataFrame([{
            'Name': dataset.name,
            'Description': dataset.description,
            'File Path': dataset.filepath
        } for dataset in datasets])

        # Display the DataFrame as a table
        st.dataframe(df)
    else:
        st.write("No datasets available.")

if __name__ == "__main__":
    main()
