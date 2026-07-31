import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#----------------------
#Configuration
#----------------------

DATASET_PATH = "D:/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv"
OUTPUT_FOLDER = "D:/PycharmProjects/placement_prediction/outputs/EDA_Analysis_outputs"

#Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#Plot  style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv(DATASET_PATH)

print("=" * 60)

print ("First Five Records")
print(df.head())

print ("\nDataset Shape:", df.shape)

print("\nColumn Names")

print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)
print("\nDataset Information")
print(df.info())
print("\nMissing Values")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
target = None
possible_targets = ["Placement", "PlacementStatus", "Status", "Placed"]
for col in possible_targets:
   if col in df.columns:
       target = col
       break
for col in numeric_cols:

   plt.figure(figsize=(8,5))
   sns.histplot(df[col], bins=20, kde=True, color='skyblue')
   plt.title(f"Histogram - {col}")
   plt.xlabel(col)
   plt.ylabel("Frequency")
   plt.savefig(os.path.join(OUTPUT_FOLDER, "univariate_histogram.png"))
   plt.close()
   plt.figure(figsize=(6,4))
   sns.boxplot(y=df[col], color="orange")
   plt.title(f"Box Plot - {col}")
   plt.savefig(os.path.join(OUTPUT_FOLDER, "boxplot.png"))
   plt.savefig(os.path.join(OUTPUT_FOLDER,
                            f"{col}_boxplot.png"),
               dpi=300,
               bbox_inches='tight')
   plt.close()
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col], color='red')
    plt.title(f"Outlier Detection - {col}")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "outlier.png"))

    plt.close()



