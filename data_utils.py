import polars as pl
import os
import json

import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from scipy.stats import zscore
from sklearn import __version__ as sklearn_version

# Function to load datasets
def load_datasets(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    return files

# Function to load selected dataset
def load_data(file_path, limit=None):
    data = pd.read_csv(file_path)
    if limit:
        data = data.head(limit)
    return data

# Function to apply filters to the dataset
def apply_filters(df, filters):
    for col, val in filters.items():
        if val:
            df = df.filter(pl.col(col) == val)
    return df

# Function to generate a summary of the dataset
def generate_summary(df):
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
    return df.describe(include='all')

# Function to generate a column-level summary
def column_summary(df, col):
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
            if target_column not in df.columns:
                raise KeyError(f"Column '{target_column}' not found.")

            if rule.rule_type == "Range Check":
                # 'eval' the custom condition (should be a string like "x > 10 and x < 20")
                condition = pl.col(target_column).apply(lambda x: eval(rule.condition))
            elif rule.rule_type == "Null Check":
                condition = pl.col(target_column).is_not_null()
            elif rule.rule_type == "Uniqueness Check":
                # Polars uniqueness check is done through n_unique comparison
                condition = (pl.col(target_column).n_unique() == pl.col(target_column).count())
            elif rule.rule_type == "Custom Lambda":
                condition = eval(rule.condition)

            if not df[target_column].apply(condition).all():
                violations.append({
                    'column': target_column,
                    'message': rule.message,
                    'severity': rule.severity
                })

        except KeyError:
            violations.append({
                'column': target_column,
                'message': f"Column '{target_column}' not found. It may have been dropped during data manipulation.",
                'severity': 'High'
            })
        except Exception as e:
            violations.append({
                'column': target_column,
                'message': f"Error applying rule: {str(e)}",
                'severity': 'High'
            }) 
    
    return violations

def apply_operations_to_dataset(dataset, operations):
    for operation in operations:
        operation_type = operation.operation_type
        parameters = json.loads(operation.parameters)
        
        if operation_type == "Rename Column":
            dataset.rename(columns={parameters["old_name"]: parameters["new_name"]}, inplace=True)
        
        elif operation_type == "Change Data Type":
            dataset[parameters["column"]] = dataset[parameters["column"]].astype(parameters["new_type"])
        
        elif operation_type == "Delete Column":
            dataset.drop(columns=parameters["columns"], inplace=True)
        
        elif operation_type == "Filter Rows":
            dataset = dataset.query(parameters["condition"])
        
        elif operation_type == "Add Calculated Column":
            dataset[parameters["new_column"]] = eval(parameters["formula"], {'__builtins__': None}, dataset)
        
        elif operation_type == "Fill Missing Values":
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
        
        elif operation_type == "Duplicate Column":
            dataset[f"{parameters['column']}_duplicate"] = dataset[parameters["column"]]
        
        elif operation_type == "Reorder Columns":
            dataset = dataset[parameters["new_order"]]
        
        elif operation_type == "Replace Values":
            dataset[parameters["column"]].replace(parameters["to_replace"], parameters["replace_with"], inplace=True)

        elif operation_type == "Scale Data":
            scaler = StandardScaler() if parameters["method"] == "StandardScaler" else MinMaxScaler()
            scaled_data = scaler.fit_transform(dataset[parameters["columns"]])
            dataset = dataset.with_columns([pl.Series(col, scaled_data[:, idx]) for idx, col in enumerate(parameters["columns"])])
        
        elif operation_type == "Encode Data":
            if parameters["type"] == "OneHotEncoding":
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded_data = encoder.fit_transform(dataset[parameters["columns"]])
                encoded_df = pl.DataFrame(encoded_data, columns=encoder.get_feature_names_out(parameters["columns"]))
                dataset = dataset.drop(parameters["columns"])
                dataset = dataset.hstack(encoded_df)
            else:
                encoder = LabelEncoder()
                for col in parameters["columns"]:
                    dataset[col] = encoder.fit_transform(dataset[col])

        elif operation_type == "Impute Missing Values":
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

        elif operation_type == "Remove Outliers":
            if parameters["method"] == "IQR Method":
                Q1 = dataset[parameters["column"]].quantile(0.25)
                Q3 = dataset[parameters["column"]].quantile(0.75)
                IQR = Q3 - Q1
                dataset = dataset.filter(~((pl.col(parameters["column"]) < (Q1 - 1.5 * IQR)) | (pl.col(parameters["column"]) > (Q3 + 1.5 * IQR))))
            elif parameters["method"] == "Z-Score Method":
                dataset = dataset[(zscore(dataset[parameters["column"]]).abs() < 3)]

        elif operation_type == "Merge Datasets":
            from models import get_db, DatasetOperation, Dataset, DatasetVersion
            from sqlalchemy.orm import Session

            merge_with = parameters["merge_with"]
            merge_column = parameters["join_column"]
            join_type = parameters["join_type"]
            merge_version_num = parameters["merge_version"]
            
            db: Session = next(get_db())
            selected_dataset = db.query(Dataset).filter(Dataset.id == merge_with).first()

            selected_version = db.query(DatasetVersion).filter(
                DatasetVersion.dataset_id == selected_dataset.id,
                DatasetVersion.id == merge_version_num
            ).first()
            selected_data = load_data(selected_version.dataset.filepath)

            operations = db.query(DatasetOperation).filter(DatasetOperation.version_id == selected_version.id).all()
            if operations:
                selected_data = apply_operations_to_dataset(selected_data, operations)

            dataset = pd.merge(dataset, selected_data, on=merge_column, how=join_type)

    return dataset
