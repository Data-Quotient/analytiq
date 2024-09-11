import polars as pl
import streamlit as st

from data_utils import generate_summary, detailed_statistics

from llm.utils import (
    suggest_models, 
    suggest_target_column, 
    explain_feature_importance_commentary, 
    explain_insights_commentary, 
    explain_predictions_commentary, 
    generate_leaderboard_commentary, 
    generate_industry_report
)

import h2o
from machine_learning.model_mapping import MODEL_MAPPING
from machine_learning.utils import (
    run_h2o_automl, 
    run_selected_model, 
    generate_pdf_report
)


def handle_ml_tab(filtered_data):
    """Handles all content and logic within the Machine Learning Tab."""
    st.header("Machine Learning Assistant")

    # Initialize session state variables
    for var in ['aml', 'test_data', 'predictions', 'feature_importance', 'selected_model_id', 'explanation', 'actual_values']:
        if var not in st.session_state:
            st.session_state[var] = None

    with st.container():
        st.subheader("1. Explain Your Use Case")
        use_case = st.text_area("Describe your use case", placeholder="E.g., I want to predict house prices based on various features.")

        st.subheader("2. Select Your Task")
        task = st.selectbox("What do you want to do?", ["Classification", "Regression"])

        st.subheader("3. Get Algorithm Suggestions")
        if st.button("Get Suggestions"):
            if use_case:
                with st.spinner("Sending your data and use case to the LLM for algorithm suggestions..."):
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

        st.subheader("5. Get Target Column Suggestion")
        if st.button("Get Suggested Target Column"):
            try:
                with st.spinner("Getting suggestions"):
                    suggested_target = suggest_target_column(
                        task,
                        filtered_data.columns,
                        use_case,
                        filtered_data.head().to_pandas(),
                        generate_summary(filtered_data),
                        detailed_statistics(filtered_data)
                    )
                    st.session_state.target_column = suggested_target
                    st.success(f"Suggested Target Column: {suggested_target}")
            except ValueError as e:
                st.error(str(e))
        
        st.session_state.target_column = st.selectbox(
            "Select Target Column",
            filtered_data.columns
            
        )

        st.subheader("6. Model Comparison and Training")
        if st.button("Run Selected Models"):
            if selected_algorithms and 'target_column' in st.session_state:
                with st.spinner(f"Running the following models: {', '.join(selected_algorithms)}"):
                    st.session_state.aml, st.session_state.test_data = run_h2o_automl(filtered_data, st.session_state.target_column, task.lower(), selected_algorithms)
                
                st.write("AutoML completed. Model leaderboard:")
                leaderboard = st.session_state.aml.leaderboard
                print(type(leaderboard))
                st.dataframe(leaderboard.as_data_frame(use_multi_thread=True).style.highlight_max(axis=0))
                
                leaderboard_commentary = generate_leaderboard_commentary(use_case, filtered_data.head().to_pandas(), selected_algorithms, leaderboard.as_data_frame(use_multi_thread=True))
                st.markdown("### Leaderboard Commentary")
                st.markdown(leaderboard_commentary)
            else:
                st.error("Please select at least one algorithm and a target column.")

        # Post-Leaderboard Analysis
        if st.session_state.aml is not None:
            leaderboard = st.session_state.aml.leaderboard
            model_ids = leaderboard['model_id'].as_data_frame().values.flatten().tolist()
            st.session_state.selected_model_id = st.selectbox("Select a model to test", model_ids)
            
            test_data_option = st.radio("Choose test data", ["Use same data", "Upload new test data"])
            if test_data_option == "Upload new test data":
                test_file = st.file_uploader("Choose a CSV file for testing", type="csv")
                if test_file is not None:
                    test_data = pl.read_parquet(test_file)
                    st.session_state.test_data = h2o.H2OFrame(test_data)
                    st.session_state.actual_values = test_data[st.session_state.target_column].to_pandas()

            if st.button("Run selected model"):
                run_selected_model()

            if st.session_state.predictions is not None:
                predictions_with_actuals = st.session_state.test_data.cbind(st.session_state.predictions)
                tabs = st.tabs(["Model Evaluation", "Feature Importance", "Business Insights", "Industry Report"])

                with tabs[0]:
                    st.write("### Model Evaluation")
                    st.write("Actual vs Predicted:")
                    st.dataframe(predictions_with_actuals.as_data_frame(use_multi_thread=True))
                    predictions_commentary = explain_predictions_commentary(predictions_with_actuals, st.session_state.actual_values)
                    st.markdown("### Predictions Commentary")
                    st.markdown(predictions_commentary)

                with tabs[1]:
                    if st.session_state.feature_importance is not None:
                        st.write("### Feature Importance:")
                        st.dataframe(st.session_state.feature_importance)
                        feature_importance_commentary = explain_feature_importance_commentary(st.session_state.feature_importance)
                        st.markdown("### Feature Importance Commentary")
                        st.markdown(feature_importance_commentary)

                with tabs[2]:
                    st.write("### Business Insights")
                    insights_commentary = explain_insights_commentary(predictions_with_actuals, st.session_state.feature_importance)
                    st.markdown(insights_commentary)

                with tabs[3]:
                    st.write("### Industry Report")
                    industry_report = generate_industry_report(
                        use_case,
                        task,
                        filtered_data,
                        st.session_state.target_column,
                        st.session_state.aml,
                        predictions_with_actuals,
                        st.session_state.feature_importance
                    )
                    st.markdown(industry_report)

                    pdf_buffer = generate_pdf_report(industry_report)
                    st.download_button(
                        label="Download Report as PDF",
                        data=pdf_buffer,
                        file_name="industry_report.pdf",
                        mime="application/pdf"
                    )

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
