import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------
# Configuration
# ----------------------

# Dynamic project root resolution (works regardless of system path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "EDA_Analysis_outputs")

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Plot style setup
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Load dataset
df = pd.read_csv(DATASET_PATH)

# ----------------------
# Dataset Overview
# ----------------------
print("=" * 60)
print("First Five Records:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Identify numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Identify target column
target = None
possible_targets = ["Placement", "PlacementStatus", "Status", "Placed"]
for col in possible_targets:
    if col in df.columns:
        target = col
        break

print(f"\nTarget Column Detected: {target}")

# ----------------------
# Visualization Loop
# ----------------------
for col in numeric_cols:
    # 1. Histogram (Distribution)
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=20, kde=True, color='skyblue')
    plt.title(f"Histogram - {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"{col}_histogram.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # 2. Boxplot (Outlier Analysis)
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col], color="orange")
    plt.title(f"Box Plot - {col}")
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, f"{col}_boxplot.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

print(f"\nEDA completed successfully! Graphs saved to: {OUTPUT_FOLDER}")