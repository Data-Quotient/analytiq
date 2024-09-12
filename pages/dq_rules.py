import streamlit as st
from sqlalchemy.orm import Session
from models import Dataset, DQRule, get_db
import polars as pl
from constants import DQ_RULES
from polars_datatypes import NUMERIC_TYPES

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

        rule_type = st.selectbox("Rule Type", [rule.value for rule in DQ_RULES])
        
        with st.form("define_rule"):
            rule_name = st.text_input("Rule Name")

            # Only show target columns for numeric types if rule_type is RANGE_CHECK
            if rule_type == DQ_RULES.RANGE_CHECK.value:
                numeric_df = df.select([pl.col(col) for col in df.columns if df.schema[col] in NUMERIC_TYPES])
                columns = numeric_df.columns
            target_columns = st.multiselect("Target Columns", columns)
            
            condition = None
            if rule_type == DQ_RULES.RANGE_CHECK.value:
                min_value = st.number_input("Minimum Value", value=None, step=0.1, format="%.2f", key="min_value")
                max_value = st.number_input("Maximum Value", value=None, step=0.1, format="%.2f", key="max_value")

                # Build condition based on which values are provided
                if min_value is not None and max_value is not None:
                    if min_value > max_value:
                        st.error("min value exceeds the max value")
                    else:
                        condition = f"lambda x: {min_value} <= x <= {max_value}"
                elif min_value is not None:
                    condition = f"lambda x: {min_value} <= x"
                elif max_value is not None:
                    condition = f"lambda x: x <= {max_value}"
                else:
                    condition = None 

            elif rule_type == DQ_RULES.CUSTOM_LAMBDA.value:
                condition = st.text_input("Condition (Lambda)")

            severity = st.selectbox("Severity", ["Warning", "Error"])
            description = st.text_area("Description (Use ${col_name} for dynamic column name)")

            submitted = st.form_submit_button("Add Rule")
            
            if submitted:
                # Check for valid rule name
                if rule_name.strip() == "":
                    st.error("Please provide a Rule Name")

                # Check for empty description
                elif description.strip() == "":
                    st.error("Please provide a Rule Description")
                
                # Ensure columns are selected
                elif not target_columns:
                    st.error("Please select at least one target column")

                elif not condition:
                    st.error("Invalid Condition")
                # If all validations pass, proceed with adding the rule
                else:
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
