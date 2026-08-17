# --------------------------------------------
# Placement Prediction Dataset Preprocessing
# --------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up dynamic paths relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

input_file = DATASET_DIR / "placement_predict_50K_Raw.csv"
output_file = DATASET_DIR / "final_preprocess_M2.csv"

# Read original dataset
df = pd.read_csv(input_file)

# Create a copy so original dataset remains unchanged
processed_df = df.copy()

print("Original Dataset Shape:", processed_df.shape)

# --------------------------------------------
# Remove Duplicate Records
# --------------------------------------------
processed_df = processed_df.drop_duplicates()

# --------------------------------------------
# Handle Missing Values
# --------------------------------------------

# Numeric Columns
numeric_cols = processed_df.select_dtypes(include=['number']).columns

for col in numeric_cols:
    processed_df[col] = processed_df[col].fillna(processed_df[col].median())

# Categorical Columns (Updated to avoid Pandas string dtype deprecation)
categorical_cols = processed_df.select_dtypes(include=['object', 'string']).columns

for col in categorical_cols:
    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])

# --------------------------------------------
# Clean Text Data
# --------------------------------------------
for col in categorical_cols:
    processed_df[col] = processed_df[col].astype(str).str.strip().str.lower()

# --------------------------------------------
# Label Encoding
# --------------------------------------------
encoder = LabelEncoder()

for col in categorical_cols:
    processed_df[col] = encoder.fit_transform(processed_df[col])

# --------------------------------------------
# Feature Scaling
# --------------------------------------------
scaler = StandardScaler()

processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])

# --------------------------------------------
# Save Preprocessed Dataset
# --------------------------------------------
processed_df.to_csv(output_file, index=False)

print("\nPreprocessing Completed Successfully!")
print("Original Dataset Shape :", df.shape)
print("Processed Dataset Shape:", processed_df.shape)
print("Saved File :", output_file)