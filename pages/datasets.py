import os
import streamlit as st
import pandas as pd
from db_utils import add_dataset
from sqlalchemy.orm import Session
from models import Dataset, get_db

# Ensure the datasets directory exists
DATASETS_DIR = "datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

def main():
    st.title("Manage Datasets")
    st.write("Upload a file to create a new dataset.")

    # Initialize session state for form inputs
    if 'name' not in st.session_state:
        st.session_state['name'] = ''
    if 'description' not in st.session_state:
        st.session_state['description'] = ''
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

    # Add the form inside an accordion
    with st.expander("Create a New Dataset", expanded=False):
        with st.form("upload_dataset"):
            # Form fields
            st.session_state['name'] = st.text_input("Dataset Name", value=st.session_state['name'])
            st.session_state['description'] = st.text_area("Description", value=st.session_state['description'])
            st.session_state['uploaded_file'] = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])

            submitted = st.form_submit_button("Create Dataset")

            if submitted and st.session_state['uploaded_file'] is not None:
                # Save the file to the datasets directory
                file_path = os.path.join(DATASETS_DIR, st.session_state['uploaded_file'].name)
                with open(file_path, "wb") as f:
                    f.write(st.session_state['uploaded_file'].getbuffer())

                # Add the dataset to the database and create default version
                dataset_info = add_dataset(st.session_state['name'], st.session_state['description'], file_path)

                # Use the accessed attributes
                st.success(f"Dataset '{dataset_info['name']}' added successfully with a default version!")
                st.write("File saved at:", dataset_info['filepath'])

                # Clear form fields after successful submission
                st.session_state['name'] = ''
                st.session_state['description'] = ''
                st.session_state['uploaded_file'] = None

    # Display existing datasets
    db: Session = next(get_db())
    datasets = db.query(Dataset).all()

    if datasets:
        st.subheader("Existing Datasets")

        # Convert datasets to a DataFrame for better display
        df = pd.DataFrame([{
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
