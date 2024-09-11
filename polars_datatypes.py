import polars as pl

DATA_TYPE_OPTIONS = {
    "Int8": pl.Int8,
    "Int16": pl.Int16,
    "Int32": pl.Int32,
    "Int64": pl.Int64,
    "UInt8": pl.UInt8,
    "UInt16": pl.UInt16,
    "UInt32": pl.UInt32,
    "UInt64": pl.UInt64,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "Utf8": pl.Utf8,
    "Boolean": pl.Boolean,
    "Date": pl.Date,
    "Datetime": pl.Datetime,
    "Date": pl.Date,
    "Time": pl.Time,
    "Duration": pl.Duration,
    "Categorical": pl.Categorical,
    "List": pl.List,
    "Object": pl.Object,
    "String": pl.String
}

NUMERIC_TYPES = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, 
    pl.Float32, pl.Float64
]

