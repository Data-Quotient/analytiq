import streamlit as st

from models import get_db, DQRule
from sqlalchemy.orm import Session

from data_utils import apply_dq_rules

# Function to handle the Data Quality tab
def handle_data_quality_tab(filtered_data, dataset_id):
    """Handles all content and logic within the Data Quality tab."""
    st.header("Data Quality Check")
    
    # Fetch DQ rules for the selected dataset from the database
    db: Session = next(get_db())
    rules = db.query(DQRule).filter(DQRule.dataset_id == dataset_id).all()

    # Apply DQ rules and show loader while processing
    with st.spinner("Applying Data Quality Rules..."):
        violations = apply_dq_rules(filtered_data, rules)
    
    if violations:
        st.warning("Data Quality Issues Found:")
        for violation in violations:
            st.write(f"{violation['severity']}: {violation['message']} in column {violation['column']}")
    else:
        st.success("No data quality issues found!")
