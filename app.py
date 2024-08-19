# app.py

import streamlit as st
from pages import datasets, dq_rules
from pages.data_hub import data_hub

# Map page names to their respective functions
PAGES = {
    "Home": data_hub.main,
    "Manage Datasets": datasets.main,
    "Data Quality Rules": dq_rules.main,
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Call the function that corresponds to the selected page
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
