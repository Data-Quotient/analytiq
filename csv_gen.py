import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Number of rows to generate
num_rows = 100

# Generate sample data
data = {
    "StringColumn": [f"String_{i}" for i in range(num_rows)],
    "Float64Column": np.random.uniform(0.0, 100.0, size=num_rows).tolist(),
    "CategoryColumn": [f"Category_{random.choice(['A', 'B', 'C'])}" for _ in range(num_rows)],
    "DateColumn": [(datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d') for _ in range(num_rows)],
    "BooleanColumn": [random.choice([True, False]) for _ in range(num_rows)],
    "AdditionalColumn1": [random.randint(1, 100) for _ in range(num_rows)],
    "AdditionalColumn2": [f"Extra_{i}" for i in range(num_rows)],
    "AdditionalColumn3": np.random.normal(50.0, 10.0, size=num_rows).tolist(),
    "AdditionalColumn4": [datetime.now().strftime('%H:%M:%S') for _ in range(num_rows)],
    "AdditionalColumn5": [random.choice(['X', 'Y', 'Z']) for _ in range(num_rows)],
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_file_path = "sample_data.csv"
df.to_csv(csv_file_path, index=False)

print(f"CSV file created at {csv_file_path}")
