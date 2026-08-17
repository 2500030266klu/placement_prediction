from pathlib import Path
import pandas as pd
import numpy as np

# Set project root path relative to this script location
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "dataset" / "placement_predict_50K_Raw.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "clean_target_encode_M2.csv"

# ==========================================================
# Load Placement Prediction Dataset
# Original dataset is NOT modified
# ==========================================================

df = pd.read_csv(INPUT_PATH)

# Create a copy
data = df.copy()

# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================

for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()

# ==========================================================
# 2. Identify Missing Values
# ==========================================================

print("Missing Values Before Cleaning:")
print(data.isnull().sum())

# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================

duplicate_count = data.duplicated().sum()
data = data.drop_duplicates()

print("\nDuplicate Records Removed:", duplicate_count)

# ==========================================================
# 4. Identify Numerical and Categorical Columns
# ==========================================================

num_cols = data.select_dtypes(include=np.number).columns.tolist()
cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)

# ==========================================================
# 5. Fill Missing Numerical Values with Mean
# ==========================================================

for col in num_cols:
    mean_value = data[col].mean()
    data[col] = data[col].fillna(mean_value)

# ==========================================================
# 6. Fill Missing Categorical Values with Mode
# ==========================================================

for col in cat_cols:
    mode_value = data[col].mode()[0]
    data[col] = data[col].fillna(mode_value)

# ==========================================================
# 7. Select Target Column
# Change according to your dataset
# ==========================================================

target_column = "PlacementStatus"

# ==========================================================
# 8. Apply Target Encoding Using Pandas
# ==========================================================

target_encoded_df = pd.DataFrame()

for col in cat_cols:
    # Do not encode target column itself
    if col != target_column:
        mean_encoding = data.groupby(col)[target_column].mean()
        target_encoded_df["Target_" + col] = data[col].map(mean_encoding)

# ==========================================================
# 9. Merge Numerical Columns and Target Encoded Columns
# ==========================================================

final_output = pd.concat(
    [
        data[num_cols].reset_index(drop=True),
        target_encoded_df.reset_index(drop=True),
    ],
    axis=1,
)

# ==========================================================
# 10. Check Missing Values After Processing
# ==========================================================

print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())

# ==========================================================
# 11. Save Final Result
# ==========================================================

final_output.to_csv(OUTPUT_PATH, index=False)

print("\n======================================")
print("Target Encoding Completed Successfully")
print("Original dataset is NOT modified")
print("Output file saved to:", OUTPUT_PATH)
print("======================================")