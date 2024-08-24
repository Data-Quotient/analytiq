import polars as pl
import pandas as pd

filename = "test.parquet"

# df = pl.read_csv("random_data.csv")

# # Update column data types
# df = df.with_columns([df['Age'].cast(pl.Int8)])

# # Save DataFrame as a Parquet file
# df.write_parquet('data.parquet')


new_df = pl.read_parquet('data.parquet')

# Step 2: Check the schema of the DataFrame
schema = new_df.schema

# Print the schema to verify data types
print(schema)
# print(df.with_columns([pl.col('Age').cast(pl.Int8)]))
# df.write_parquet("test.parquet")

# specify schema using schema=
# use `infer_schema_length=10000`
# set `ignore_errors` to True
# add | to null_values

# def column_summary_pd(df: pd.DataFrame, col):
#     """Generates a detailed summary for a single column."""
#     summary = {
#         'Data Type': df[col].dtype,
#         'Unique Values': df[col].nunique(),
#         'Missing Values': df[col].isnull().sum(),
#         'Mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
#         'Median': df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
#         'Mode': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
#         'Standard Deviation': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
#         'Min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
#         'Max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
#     }
#     return summary


# def column_summary_pl(df: pl.DataFrame, col: str) -> dict:
#     """Generates a detailed summary for a single column in a Polars DataFrame."""
#     # Define numeric types
#     numeric_types = [
#         pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
#         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
#         pl.Float32, pl.Float64
#     ]
    
#     # Get the column as a Series
#     column = df[col]
#     dtype = column.dtype
    
#     # Define the summary dictionary
#     summary = {
#         'Data Type': dtype,
#         'Unique Values': column.n_unique(),
#         'Missing Values': column.is_null().sum(),
#         'Mean': column.mean() if dtype in numeric_types else 'N/A',
#         'Median': column.median() if dtype in numeric_types else 'N/A',
#         'Mode': column.mode().sort()[0] if dtype in numeric_types else 'N/A',
#         'Standard Deviation': column.std() if dtype in numeric_types else 'N/A',
#         'Min': column.min() if dtype in numeric_types else 'N/A',
#         'Max': column.max() if dtype in numeric_types else 'N/A',
#     }
    
#     return summary

# dfl = pl.read_csv(filename, infer_schema_length=10000)
# dfp = pd.read_csv(filename)
# schema = dfl.schema
# schema_dict = {col: str(dtype) for col, dtype in schema.items()}
# print(type(schema_dict))

# col = "Name"
# print(column_summary_pd(dfp, col))
# print(column_summary_pl(dfl, col))
# print(column_summary(dfp))

# df = dfp
# print({
#     'Number of Rows': len(df),
#     'Number of Columns': len(df.columns),
#     'Missing Values': df.isnull().sum().sum(),
#     'Duplicate Rows': df.duplicated().sum(),
#     'Memory Usage (MB)': round(df.memory_usage(deep=True).sum() / (1024**2), 2)
# })

# df = dfl
# print({
#     'Number of Rows': len(df),
#     'Number of Columns': len(df.columns),
#     'Missing Values': df.null_count().to_pandas().sum().sum(),
#     'Duplicate Rows': int(df.is_duplicated().sum()/2),
#     'Memory Usage (MB)': round(df.estimated_size()/ (1024**2), 2)
# })
# print(int(dfl.is_duplicated().sum()/2))
# print(dfp.duplicated().sum())