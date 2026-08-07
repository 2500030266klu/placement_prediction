# ---------------------------------------------------------
# Numeric Column Pre-processing Techniques
# Imputation, Feature Scaling, Standardization, and Normalization
# ---------------------------------------------------------

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------
file_path = "C:/Users/anil pandey/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv"

df = pd.read_csv(file_path)

print("Original Dataset")
print("---------------------------")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nData Types:")
print("------------------------")
print(df.dtypes)

print("\nMissing Values:")
print("------------------------")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# ---------------------------------------------------
# Step 2: Remove Duplicate Records
# ---------------------------------------------------
df = df.drop_duplicates()

# ---------------------------------------------------
# Step 3: Handle Missing Values
# ---------------------------------------------------
# Numerical Columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].mean())

# Categorical Columns (Updated to avoid Pandas4Warning)
categorical_columns = df.select_dtypes(include=['object', 'string']).columns

# Fill missing values in categorical columns with mode
for column in categorical_columns:
    if not df[column].mode().empty:
        df[column] = df[column].fillna(df[column].mode()[0])

# ---------------------------------------------------
# Step 4: Remove Extra Spaces from Text Columns
# ---------------------------------------------------
for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()

# ------------------------------------------------------------
# Step 5: Feature Transformations (Standardization, Scaling, Normalization)
# ------------------------------------------------------------
# Store original numeric column names explicitly
numeric_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
print("\nNumeric Columns targeted for transformation:")
print(numeric_cols)

# 1. Standardization (Z-score: Mean = 0, Std = 1)
standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(df[numeric_cols])
for i, col in enumerate(numeric_cols):
    df[f"{col}_Standardized"] = standardized_data[:, i]

# 2. Min-Max Scaling (Range: [0, 1])
minmax_scaler = MinMaxScaler()
scaled_data = minmax_scaler.fit_transform(df[numeric_cols])
for i, col in enumerate(numeric_cols):
    df[f"{col}_Scaled"] = scaled_data[:, i]

# 3. L2 Normalization (Row-wise Unit Vector)
normalizer = Normalizer(norm='l2')
normalized_data = normalizer.fit_transform(df[numeric_cols])
for i, col in enumerate(numeric_cols):
    df[f"{col}_Normalized"] = normalized_data[:, i]

# ------------------------------------------------------------
# Step 6: Verify Results
# ------------------------------------------------------------
print("\n--- Display Preprocessed Dataset ---")
print(df.head())
print("\nDataset Shape:", df.shape)

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum().sum())

print("\nDuplicate Records After Preprocessing:", df.duplicated().sum())

# ---------------------------------------------------
# Step 7: Save Preprocessed Dataset
# ---------------------------------------------------
# FIXED: Updated path to match current project directory
output_dir = "C:/Users/anil pandey/PycharmProjects/placement_prediction/dataset"
output_path = os.path.join(output_dir, "clean_minmax_stand_normal_M2.csv")

# Create directory automatically if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

df.to_csv(output_path, index=False)

print(f"\nPreprocessed dataset successfully saved to: {output_path}")