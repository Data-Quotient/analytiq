import polars as pl
import os
import json
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from scipy.stats import zscore

NUMERIC_TYPES = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
    pl.Float32, pl.Float64
]

# Function to load datasets
def load_datasets(folder_path):
    """Loads CSV file names from the specified folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    return files

# Function to load selected dataset
def load_data(file_path, limit=None) -> pl.dataframe:
    """Loads data from the selected CSV file and applies a row limit."""
    data = pl.read_parquet(file_path)
    if limit:
        data = data.head(limit)
    return data

# Function to apply filters to the dataset
def apply_filters(df, filters):
    """Applies the user-selected filters to the DataFrame."""
    for col, val in filters.items():
        if val:
            df = df.filter(pl.col(col) == val)
    return df

# Function to generate a summary of the dataset
def generate_summary(df: pl.DataFrame):
    """Generates a summary of the DataFrame."""
    summary = {
        'Number of Rows': len(df),
        'Number of Columns': len(df.columns),
        'Missing Values': df.null_count().to_pandas().sum().sum(),
        'Duplicate Rows': int(df.is_duplicated().sum()/2),
        'Memory Usage (MB)': round(df.estimated_size()/ (1024**2), 2)
    }
    return summary

# Function to display detailed statistics for each column
def detailed_statistics(df):
    """Displays detailed statistics for each column."""
    return df.describe()

# Function to generate a column-level summary
def column_summary(df, col):
    """Generates a detailed summary for a single column."""
    column = df[col]
    dtype = column.dtype
    summary = {
        'Data Type': dtype,
        'Unique Values': column.n_unique(),
        'Missing Values': column.is_null().sum(),
        'Mean': column.mean() if dtype in NUMERIC_TYPES else 'N/A',
        'Median': column.median() if dtype in NUMERIC_TYPES else 'N/A',
        'Mode': column.mode()[0] if dtype in NUMERIC_TYPES else 'N/A',
        'Standard Deviation': column.std() if dtype in NUMERIC_TYPES else 'N/A',
        'Min': column.min() if dtype in NUMERIC_TYPES else 'N/A',
        'Max': column.max() if dtype in NUMERIC_TYPES else 'N/A',
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
                condition = lambda x: x.is_not_null()
            elif rule.rule_type == "Uniqueness Check":
                condition = lambda x: x.is_unique
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
            dataset = dataset.rename({parameters["old_name"]: parameters["new_name"]})
        
        elif action_type == "Change Data Type":
            dataset = dataset.with_column(pl.col(parameters["column"]).cast(parameters["new_type"]))
        
        elif action_type == "Delete Column":
            dataset = dataset.drop(parameters["columns"])
        
        elif action_type == "Filter Rows":
            dataset = dataset.filter(pl.col(parameters["condition"]))
        
        elif action_type == "Add Calculated Column":
            dataset = dataset.with_column(pl.eval(parameters["formula"]).alias(parameters["new_column"]))
        
        elif action_type == "Fill Missing Values":
            if parameters["method"] == "Specific Value":
                dataset = dataset.fill_null(pl.lit(parameters["value"]))
            elif parameters["method"] == "Mean":
                mean = dataset[parameters["column"]].mean()
                dataset = dataset.fill_null(mean)
            elif parameters["method"] == "Median":
                median = dataset[parameters["column"]].median()
                dataset = dataset.fill_null(median)
            elif parameters["method"] == "Mode":
                mode = dataset[parameters["column"]].mode()[0]
                dataset = dataset.fill_null(mode)
        
        elif action_type == "Duplicate Column":
            dataset = dataset.with_column(dataset[parameters["column"]].alias(f"{parameters['column']}_duplicate"))
        
        elif action_type == "Reorder Columns":
            dataset = dataset.select(parameters["new_order"])
        
        elif action_type == "Replace Values":
            dataset = dataset.with_column(pl.col(parameters["column"]).replace(parameters["to_replace"], parameters["replace_with"]))
        
        # Handle Preprocessing Actions
        elif action_type == "Scale Data":
            scaler = StandardScaler() if parameters["method"] == "StandardScaler" else MinMaxScaler()
            scaled_data = scaler.fit_transform(dataset[parameters["columns"]])
            dataset = dataset.with_columns([pl.Series(col, scaled_data[:, idx]) for idx, col in enumerate(parameters["columns"])])
        
        elif action_type == "Encode Data":
            if parameters["type"] == "OneHotEncoding":
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded_data = encoder.fit_transform(dataset[parameters["columns"]])
                encoded_df = pl.DataFrame(encoded_data, columns=encoder.get_feature_names_out(parameters["columns"]))
                dataset = dataset.drop(parameters["columns"])
                dataset = dataset.hstack(encoded_df)
            else:
                encoder = LabelEncoder()
                for col in parameters["columns"]:
                    dataset = dataset.with_column(pl.col(col).apply(lambda x: encoder.fit_transform(x)))
        
        elif action_type == "Impute Missing Values":
            for col in parameters["columns"]:
                if parameters["method"] == "Mean":
                    mean = dataset[col].mean()
                    dataset = dataset.fill_null(pl.lit(mean))
                elif parameters["method"] == "Median":
                    median = dataset[col].median()
                    dataset = dataset.fill_null(pl.lit(median))
                elif parameters["method"] == "Mode":
                    mode = dataset[col].mode()[0]
                    dataset = dataset.fill_null(pl.lit(mode))

        elif action_type == "Remove Outliers":
            if parameters["method"] == "IQR Method":
                Q1 = dataset[parameters["column"]].quantile(0.25)
                Q3 = dataset[parameters["column"]].quantile(0.75)
                IQR = Q3 - Q1
                dataset = dataset.filter(~((pl.col(parameters["column"]) < (Q1 - 1.5 * IQR)) | (pl.col(parameters["column"]) > (Q3 + 1.5 * IQR))))
            elif parameters["method"] == "Z-Score Method":
                z_scores = zscore(dataset[parameters["column"]])
                dataset = dataset.filter(pl.col(parameters["column"]).apply(lambda x: abs(x) < 3))
        
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
            dataset = dataset.join(selected_data, on=merge_column, how=join_type)

    return dataset
