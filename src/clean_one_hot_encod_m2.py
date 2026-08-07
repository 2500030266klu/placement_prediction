from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ==========================================================
# Dynamic Path Resolution (Works across any system/user)
# ==========================================================
# Points to project root directory (one level up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "placement_predict_50K_Raw.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "clean_one_hot_encode_M2.csv"

# Read input CSV
df = pd.read_csv(INPUT_PATH)

data = df.copy()

print("Original Dataset")
print("------------------------")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nData Types:")
print("------------------------")
print(data.dtypes)

print("\nMissing Values:")
print("------------------------")
print(data.isnull().sum())

print("\nDuplicate Records:", data.duplicated().sum())

# Clean categorical column whitespace
cat_cols = data.select_dtypes(include=["object", "string"]).columns.tolist()

for col in cat_cols:
    data[col] = data[col].astype(str).str.strip()

before_duplicates = data.shape[0]

data = data.drop_duplicates()

after_duplicates = data.shape[0]

print("\nDuplicate Records Removed:", before_duplicates - after_duplicates)

# Identify numerical and categorical columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
cat_cols = data.select_dtypes(include=["object", "string"]).columns.tolist()

# Impute missing values
if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy="mean")
    data[num_cols] = num_imputer.fit_transform(data[num_cols])

if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# Apply One-Hot Encoding
if len(cat_cols) > 0:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    encoded_values = encoder.fit_transform(data[cat_cols])

    encoded_df = pd.DataFrame(
        encoded_values, columns=encoder.get_feature_names_out(cat_cols)
    )

    encoded_df.reset_index(drop=True, inplace=True)

    numeric_df = data[num_cols].reset_index(drop=True)

    final_output = pd.concat([numeric_df, encoded_df], axis=1)
else:
    final_output = data.copy()

print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())

print("\nDataset Shape:", final_output.shape)

print("\nFirst 5 Rows:")
print(final_output.head())

# Save dataset
final_output.to_csv(OUTPUT_PATH, index=False)

print("\n======================================")
print("Original dataset is NOT modified.")
print("Cleaning and One-Hot Encoding completed successfully.")
print(f"Output file saved to:\n{OUTPUT_PATH}")
print("======================================")