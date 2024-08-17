MODEL_MAPPING = {
        "classification": {
            "GBM": "Gradient Boosting Machine (GBM)",
            "DRF": "Distributed Random Forest (DRF)",
            "XGBoost": "Extreme Gradient Boosting (XGBoost)",
            "GLM": "Generalized Linear Model (GLM)",
            "DeepLearning": "Deep Learning",
            "NaiveBayes": "Naive Bayes",
            "StackedEnsemble": "Stacked Ensemble",
            "RuleFit": "RuleFit",
            "SVM": "Support Vector Machine (SVM)",
        },
        "regression": {
            "GBM": "Gradient Boosting Machine (GBM)",
            "DRF": "Distributed Random Forest (DRF)",
            "XGBoost": "Extreme Gradient Boosting (XGBoost)",
            "GLM": "Generalized Linear Model (GLM)",
            "DeepLearning": "Deep Learning",
            "StackedEnsemble": "Stacked Ensemble",
            "RuleFit": "RuleFit",
            "SVM": "Support Vector Machine (SVM)",
            "GAM": "Generalized Additive Models (GAM)",
        },
        "clustering": {
            "KMeans": "K-Means",
            "GLRM": "Generalized Low Rank Model (GLRM)"
        },
        "anomaly_detection": {
            "IsolationForest": "Isolation Forest",
            "AutoEncoder": "AutoEncoder",
            "OneClassSVM": "One-Class SVM"
        },
        "dimensionality_reduction": {
            "PCA": "Principal Component Analysis (PCA)",
            "GLRM": "Generalized Low Rank Model (GLRM)"
        },
        "time_series": {
            "DeepLearning": "Deep Learning",
            "DRF": "Distributed Random Forest (DRF)",
            "GBM": "Gradient Boosting Machine (GBM)",
            "XGBoost": "Extreme Gradient Boosting (XGBoost)",
            "ARIMA": "AutoRegressive Integrated Moving Average (ARIMA)",
            "Prophet": "Prophet"
        },
        # Add more mappings for other problem types if needed
    }