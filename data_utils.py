import pandas as pd
import os
import json

# Try to import necessary packages and install if not available
# Try to import necessary packages and install if not available
try:
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
    from scipy.stats import zscore
    from sklearn import __version__ as sklearn_version

except ModuleNotFoundError as e:
    import subprocess
    import sys
    missing_package = str(e).split("'")[1]  # Get the missing package name
    
    # Correctly handle the sklearn package by installing scikit-learn
    if missing_package == 'sklearn':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    elif missing_package == 'plotly':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    elif missing_package == 'scipy':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    
    # Reimport after installation
    if missing_package == "plotly":
        import plotly.express as px
    elif missing_package == "sklearn":
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
    elif missing_package == "scipy":
        from scipy.stats import zscore

    from sklearn import __version__ as sklearn_version



# Function to load datasets
def load_datasets(folder_path):
    """Loads CSV file names from the specified folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    return files

# Function to load selected dataset
def load_data(file_path, limit=None):
    """Loads data from the selected CSV file and applies a row limit."""
    data = pd.read_csv(file_path)
    if limit:
        data = data.head(limit)
    return data

# Function to apply filters to the dataset
def apply_filters(df, filters):
    """Applies the user-selected filters to the DataFrame."""
    for col, val in filters.items():
        if val:
            df = df[df[col] == val]
    return df

# Function to generate a summary of the dataset
def generate_summary(df):
    """Generates a summary of the DataFrame."""
    summary = {
        'Number of Rows': len(df),
        'Number of Columns': len(df.columns),
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum(),
        'Memory Usage (MB)': round(df.memory_usage(deep=True).sum() / (1024**2), 2)
    }
    return summary

# Function to display detailed statistics for each column
def detailed_statistics(df):
    """Displays detailed statistics for each column."""
    return df.describe(include='all')

# Function to generate a column-level summary
def column_summary(df, col):
    """Generates a detailed summary for a single column."""
    summary = {
        'Data Type': df[col].dtype,
        'Unique Values': df[col].nunique(),
        'Missing Values': df[col].isnull().sum(),
        'Mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
        'Median': df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
        'Mode': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
        'Standard Deviation': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
        'Min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
        'Max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
    }
    return summary


def apply_dq_rules(df, rules):
    violations = []
    
    for rule in rules:
        target_column = rule.target_column
        
        try:
            # Check if the target column exists before applying the rule
            if target_column not in df.columns:
                raise KeyError(f"Column '{target_column}' not found.")

            # Define the lambda function directly in the rule
            if rule.rule_type == "Range Check":
                condition = lambda x: eval(rule.condition)
            elif rule.rule_type == "Null Check":
                condition = lambda x: pd.notnull(x)
            elif rule.rule_type == "Uniqueness Check":
                condition = lambda x: df[target_column].is_unique
            elif rule.rule_type == "Custom Lambda":
                condition = eval(rule.condition)  # Custom Lambda provided by the user

            # Apply the condition and check for violations
            if not df[target_column].apply(condition).all():
                violations.append({
                    'column': target_column,
                    'message': rule.message,
                    'severity': rule.severity
                })

        except KeyError:
            # Handle the case where the column was dropped during data manipulation
            violations.append({
                'column': target_column,
                'message': f"Column '{target_column}' not found. It may have been dropped during data manipulation.",
                'severity': 'High'  # Adjust severity as needed
            })
        except Exception as e:
            # Handle any other unexpected errors
            violations.append({
                'column': target_column,
                'message': f"Error applying rule: {str(e)}",
                'severity': 'High'
            })
    
    return violations



def apply_actions_to_dataset(dataset, actions):
    """
    Apply a list of actions to a dataset.
    
    Args:
        dataset (DataFrame): The dataset to which actions will be applied.
        actions (list): List of DatasetAction objects.
        
    Returns:
        DataFrame: The dataset after all actions have been applied.
    """
    for action in actions:
        action_type = action.action_type
        parameters = json.loads(action.parameters)  # Decode JSON string to dictionary
        
        if action_type == "Rename Column":
            dataset.rename(columns={parameters["old_name"]: parameters["new_name"]}, inplace=True)
        
        elif action_type == "Change Data Type":
            dataset[parameters["column"]] = dataset[parameters["column"]].astype(parameters["new_type"])
        
        elif action_type == "Delete Column":
            dataset.drop(columns=parameters["columns"], inplace=True)
        
        elif action_type == "Filter Rows":
            dataset = dataset.query(parameters["condition"])
        
        elif action_type == "Add Calculated Column":
            dataset[parameters["new_column"]] = eval(parameters["formula"], {'__builtins__': None}, dataset)
        
        elif action_type == "Fill Missing Values":
            if parameters["method"] == "Specific Value":
                dataset[parameters["column"]].fillna(parameters["value"], inplace=True)
            elif parameters["method"] == "Mean":
                dataset[parameters["column"]].fillna(dataset[parameters["column"]].mean(), inplace=True)
            elif parameters["method"] == "Median":
                dataset[parameters["column"]].fillna(dataset[parameters["column"]].median(), inplace=True)
            elif parameters["method"] == "Mode":
                dataset[parameters["column"]].fillna(dataset[parameters["column"]].mode()[0], inplace=True)
        
        elif action_type == "Duplicate Column":
            dataset[f"{parameters['column']}_duplicate"] = dataset[parameters["column"]]
        
        elif action_type == "Reorder Columns":
            dataset = dataset[parameters["new_order"]]
        
        elif action_type == "Replace Values":
            dataset[parameters["column"]].replace(parameters["to_replace"], parameters["replace_with"], inplace=True)

        # Handle Preprocessing Actions
        elif action_type == "Scale Data":
            scaler = StandardScaler() if parameters["method"] == "StandardScaler" else MinMaxScaler()
            dataset[parameters["columns"]] = scaler.fit_transform(dataset[parameters["columns"]])
        
        elif action_type == "Encode Data":
            if parameters["type"] == "OneHotEncoding":
                if sklearn_version >= '1.2':
                    encoder = OneHotEncoder(sparse_output=False, drop='first')
                else:
                    encoder = OneHotEncoder(sparse=False, drop='first')
                encoded_data = encoder.fit_transform(dataset[parameters["columns"]])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(parameters["columns"]))
                dataset.drop(columns=parameters["columns"], inplace=True)
                dataset = pd.concat([dataset, encoded_df], axis=1)
            else:
                encoder = LabelEncoder()
                for col in parameters["columns"]:
                    dataset[col] = encoder.fit_transform(dataset[col])

        elif action_type == "Impute Missing Values":
            for col in parameters["columns"]:
                if parameters["method"] == "Mean":
                    dataset[col].fillna(dataset[col].mean(), inplace=True)
                elif parameters["method"] == "Median":
                    dataset[col].fillna(dataset[col].median(), inplace=True)
                elif parameters["method"] == "Mode":
                    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

        elif action_type == "Remove Outliers":
            if parameters["method"] == "IQR Method":
                Q1 = dataset[parameters["column"]].quantile(0.25)
                Q3 = dataset[parameters["column"]].quantile(0.75)
                IQR = Q3 - Q1
                dataset = dataset[~((dataset[parameters["column"]] < (Q1 - 1.5 * IQR)) | (dataset[parameters["column"]] > (Q3 + 1.5 * IQR)))]
            elif parameters["method"] == "Z-Score Method":
                dataset = dataset[(zscore(dataset[parameters["column"]]).abs() < 3)]

        elif action_type == "Merge Datasets":
            from models import get_db, DatasetAction, Dataset, DatasetVersion  # Import the Dataset model and database session
            from sqlalchemy.orm import Session

            merge_with = parameters["merge_with"]
            merge_column = parameters["join_column"]
            join_type = parameters["join_type"]
            merge_version_num = parameters["merge_version"]
            
            # Load the dataset to merge with
            db: Session = next(get_db())

            selected_dataset = db.query(Dataset).filter(Dataset.id == merge_with).first()

            selected_version = db.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == selected_dataset.id,
                DatasetVersion.id == merge_version_num
            ).first()
            selected_data = load_data(selected_version.dataset.filepath)

            # Apply actions recorded for the selected version
            actions = db.query(DatasetAction).filter(DatasetAction.version_id == selected_version.id).all()
            if actions:
                selected_data = apply_actions_to_dataset(selected_data, actions)

            # Perform the merge between the original dataset and the selected merge dataset version
            dataset = pd.merge(dataset, selected_data, on=merge_column, how=join_type)

    return dataset
