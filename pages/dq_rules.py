import streamlit as st
from sqlalchemy.orm import Session
from models import Dataset, DQRule, get_db
import polars as pl

def main():
    st.title("Data Quality Rules")
    
    db: Session = next(get_db())
    datasets = db.query(Dataset).all()
    
    if not datasets:
        st.write("No datasets available. Please upload a dataset first.")
        return
    
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select Dataset", dataset_names)
    
    if selected_dataset_name:
        selected_dataset = db.query(Dataset).filter(Dataset.name == selected_dataset_name).first()
        st.subheader(f"Define Rules for {selected_dataset_name}")
        
        # Read the dataset using polars
        df = pl.read_parquet(selected_dataset.filepath)
        columns = df.columns

        rule_type = st.selectbox("Rule Type", ["Range Check", "Null Check", "Uniqueness Check", "Custom Lambda"])
        
        with st.form("define_rule"):
            rule_name = st.text_input("Rule Name")
            target_columns = st.multiselect("Target Columns", columns)
            
            condition = None
            if rule_type == "Range Check":
                min_value = st.number_input("Minimum Value", value=0.0)
                max_value = st.number_input("Maximum Value", value=100.0)
                condition = f"lambda x: {min_value} <= x <= {max_value}"
            elif rule_type == "Custom Lambda":
                condition = st.text_input("Condition (Lambda)")

            severity = st.selectbox("Severity", ["Warning", "Error"])
            description = st.text_area("Description (Use ${col_name} for dynamic column name)")

            submitted = st.form_submit_button("Add Rule")
            
            if submitted:
                for target_column in target_columns:
                    dynamic_message = description.replace("${col_name}", target_column)
                    new_rule = DQRule(
                        dataset_id=selected_dataset.id,
                        rule_name=rule_name,
                        rule_type=rule_type,
                        target_column=target_column,
                        condition=condition if condition else "",
                        severity=severity,
                        message=dynamic_message
                    )
                    db.add(new_rule)
                db.commit()
                st.success(f"Rule '{rule_name}' added successfully!")
        
        st.subheader(f"Existing Rules for {selected_dataset_name}")
        rules = db.query(DQRule).filter(DQRule.dataset_id == selected_dataset.id).all()
        
        if rules:
            # Convert rules to a polars DataFrame
            df_rules = pl.DataFrame([{
                "Rule Name": rule.rule_name,
                "Type": rule.rule_type,
                "Column": rule.target_column,
                "Condition": rule.condition,
                "Severity": rule.severity,
                "Message": rule.message
            } for rule in rules])
            st.table(df_rules)
        else:
            st.write("No rules defined for this dataset yet.")

if __name__ == "__main__":
    main()
