# ================================================================
# RANDOM FOREST CLASSIFIER
# PLACEMENT PREDICTION USING RAW DATASET
# ================================================================
#
# IMPORTANT:
# The original raw dataset is NEVER modified.
#
# All preprocessing is performed on copies / inside a pipeline.
#
# OUTPUTS:
#   1. Accuracy
#   2. Precision
#   3. Recall
#   4. F1 Score
#   5. Confusion Matrix
#   6. Performance Graph
#   7. Actual vs Predicted Chart
#   8. Class Distribution Chart
#   9. Feature Importance Chart
#  10. Random Forest Tree Visualization
#  11. Classification Report
#  12. Prediction CSV
#  13. Random Forest Model
#  14. Model Parameters
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
   confusion_matrix,
   accuracy_score,
   precision_score,
   recall_score,
   f1_score,
   classification_report
)


# ================================================================
# 2. PATH CONFIGURATION (RELATIVE PATHS)
# ================================================================

# Automatically detects the root directory of your project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
   BASE_DIR,
   "dataset",
   "placement_predict_50K_Raw.csv"
)

# ================================================================
# 3. OUTPUT MAIN FOLDER
# ================================================================

OUTPUT_FOLDER = os.path.join(
   BASE_DIR,
   "outputs",
   "Random_Forest_Tree_M3_Outputs"
)


# ================================================================
# 4. OUTPUT SUBFOLDERS
# ================================================================

METRICS_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "metrics"
)

PREDICTIONS_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "predictions"
)

CONFUSION_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "confusion_matrix"
)

CHARTS_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "charts"
)

TREE_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "random_forest_tree"
)

FEATURE_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "feature_importance"
)

MODEL_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "model"
)


# ================================================================
# 5. CREATE OUTPUT FOLDERS
# ================================================================

folders = [
   METRICS_FOLDER,
   PREDICTIONS_FOLDER,
   CONFUSION_FOLDER,
   CHARTS_FOLDER,
   TREE_FOLDER,
   FEATURE_FOLDER,
   MODEL_FOLDER
]

for folder in folders:
   os.makedirs(
       folder,
       exist_ok=True
   )


# ================================================================
# 6. PROGRAM HEADER
# ================================================================

print("\n")
print("=" * 80)
print(
   "             RANDOM FOREST PLACEMENT PREDICTION"
)
print("=" * 80)


# ================================================================
# 7. CHECK DATASET
# ================================================================

if not os.path.exists(DATASET_PATH):
   print("\nERROR: Dataset not found.")
   print("\nCheck dataset path:")
   print(DATASET_PATH)
   raise SystemExit

print("\nDataset found successfully.")


# ================================================================
# 8. LOAD RAW DATASET
# ================================================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Number of rows    :", df.shape[0])
print("Number of columns :", df.shape[1])


# ================================================================
# 9. CREATE COPY
# ================================================================
#
# The original dataframe remains untouched.
#
# ================================================================

data = df.copy()


# ================================================================
# 10. DISPLAY DATASET INFORMATION
# ================================================================

print("\nDataset columns:")
print(list(data.columns))

print("\nFirst five records:")
print(data.head())


# ================================================================
# 11. IDENTIFY TARGET COLUMN
# ================================================================

possible_targets = [
   "PlacementStatus"
]

target_column = None

for column in possible_targets:
   if column in data.columns:
       target_column = column
       break


# ================================================================
# 12. TARGET NOT FOUND
# ================================================================

if target_column is None:
   print("\nTarget column was not automatically detected.")
   print("\nAvailable columns:")
   print(list(data.columns))
   print("\nSet target_column manually in the program.")
   raise SystemExit

print("\nTarget column:", target_column)


# ================================================================
# 13. REMOVE MISSING TARGET ROWS
# ================================================================

data_model = data.dropna(
   subset=[target_column]
).copy()

print("\nRecords used for modeling:", len(data_model))


# ================================================================
# 14. SEPARATE X AND y
# ================================================================

X = data_model.drop(
   columns=[target_column]
).copy()

y = data_model[target_column].copy()


# ================================================================
# 15. TARGET DISTRIBUTION
# ================================================================

print("\nTarget class distribution:")
print(y.value_counts())


# ================================================================
# 16. REMOVE COMPLETELY EMPTY COLUMNS
# ================================================================

empty_columns = X.columns[
   X.isnull().all()
].tolist()

if len(empty_columns) > 0:
   print("\nCompletely empty columns found:")
   print(empty_columns)
   X = X.drop(
       columns=empty_columns
   )


# ================================================================
# 17. IDENTIFY NUMERIC FEATURES
# ================================================================

numeric_features = X.select_dtypes(
   include=np.number
).columns.tolist()


# ================================================================
# 18. IDENTIFY CATEGORICAL FEATURES
# ================================================================

categorical_features = X.select_dtypes(
   include=[
       "object",
       "category",
       "bool"
   ]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ================================================================
# 19. NUMERIC PREPROCESSING
# ================================================================

numeric_transformer = Pipeline(
   steps=[
       (
           "imputer",
           SimpleImputer(
               strategy="median"
           )
       )
   ]
)


# ================================================================
# 20. CATEGORICAL PREPROCESSING
# ================================================================

categorical_transformer = Pipeline(
   steps=[
       (
           "imputer",
           SimpleImputer(
               strategy="most_frequent"
           )
       ),
       (
           "onehot",
           OneHotEncoder(
               handle_unknown="ignore",
               sparse_output=False
           )
       )
   ]
)


# ================================================================
# 21. COLUMN TRANSFORMER
# ================================================================

preprocessor = ColumnTransformer(
   transformers=[
       (
           "numeric",
           numeric_transformer,
           numeric_features
       ),
       (
           "categorical",
           categorical_transformer,
           categorical_features
       )
   ],
   remainder="drop"
)


# ================================================================
# 22. TRAIN TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size=0.20,
   random_state=42,
   stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ================================================================
# 23. CREATE RANDOM FOREST CLASSIFIER
# ================================================================

random_forest = RandomForestClassifier(
   n_estimators=100,
   criterion="gini",
   max_depth=10,
   min_samples_split=10,
   min_samples_leaf=5,
   max_features="sqrt",
   bootstrap=True,
   random_state=42,
   n_jobs=-1
)


# ================================================================
# 24. CREATE PIPELINE
# ================================================================

model = Pipeline(
   steps=[
       (
           "preprocessor",
           preprocessor
       ),
       (
           "classifier",
           random_forest
       )
   ]
)


# ================================================================
# 25. TRAIN RANDOM FOREST
# ================================================================

print("\nTraining Random Forest...")

model.fit(
   X_train,
   y_train
)

print("Random Forest training completed.")


# ================================================================
# 26. PREDICTION
# ================================================================

print("\nGenerating predictions...")

y_pred = model.predict(
   X_test
)

print("Prediction completed.")


# ================================================================
# 27. CALCULATE ACCURACY
# ================================================================

accuracy = accuracy_score(
   y_test,
   y_pred
)


# ================================================================
# 28. CALCULATE PRECISION
# ================================================================

precision = precision_score(
   y_test,
   y_pred,
   average="weighted",
   zero_division=0
)


# ================================================================
# 29. CALCULATE RECALL
# ================================================================

recall = recall_score(
   y_test,
   y_pred,
   average="weighted",
   zero_division=0
)


# ================================================================
# 30. CALCULATE F1 SCORE
# ================================================================

f1 = f1_score(
   y_test,
   y_pred,
   average="weighted",
   zero_division=0
)


# ================================================================
# 31. DISPLAY PERFORMANCE
# ================================================================

print("\n")
print("=" * 80)
print(
   "                RANDOM FOREST PERFORMANCE"
)
print("=" * 80)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nPercentage:")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# ================================================================
# 32. SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame({
   "Metric": [
       "Accuracy",
       "Precision",
       "Recall",
       "F1 Score"
   ],
   "Score": [
       accuracy,
       precision,
       recall,
       f1
   ],
   "Percentage": [
       accuracy * 100,
       precision * 100,
       recall * 100,
       f1 * 100
   ]
})

metrics_path = os.path.join(
   METRICS_FOLDER,
   "random_forest_metrics.csv"
)

metrics_df.to_csv(
   metrics_path,
   index=False
)


# ================================================================
# 33. CLASSIFICATION REPORT
# ================================================================

classification_report_result = classification_report(
   y_test,
   y_pred,
   output_dict=True,
   zero_division=0
)

classification_report_df = pd.DataFrame(
   classification_report_result
).transpose()

classification_report_path = os.path.join(
   METRICS_FOLDER,
   "classification_report.csv"
)

classification_report_df.to_csv(
   classification_report_path
)


# ================================================================
# 34. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
   y_test,
   y_pred
)

random_forest_classifier = (
   model.named_steps["classifier"]
)

class_labels = (
   random_forest_classifier.classes_
)

print("\n")
print("=" * 80)
print(
   "                    CONFUSION MATRIX"
)
print("=" * 80)

print(cm)


# ================================================================
# 35. SAVE CONFUSION MATRIX CSV
# ================================================================

cm_df = pd.DataFrame(
   cm,
   index=[
       "Actual_" + str(label)
       for label in class_labels
   ],
   columns=[
       "Predicted_" + str(label)
       for label in class_labels
   ]
)

cm_csv_path = os.path.join(
   CONFUSION_FOLDER,
   "confusion_matrix.csv"
)

cm_df.to_csv(
   cm_csv_path
)


# ================================================================
# 36. CONFUSION MATRIX GRAPH
# ================================================================

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(
   range(len(class_labels)),
   class_labels,
   rotation=45
)
plt.yticks(
   range(len(class_labels)),
   class_labels
)

for i in range(cm.shape[0]):
   for j in range(cm.shape[1]):
       plt.text(
           j,
           i,
           str(cm[i, j]),
           ha="center",
           va="center"
       )

plt.colorbar()
plt.tight_layout()

confusion_image_path = os.path.join(
   CONFUSION_FOLDER,
   "confusion_matrix.png"
)

plt.savefig(
   confusion_image_path,
   dpi=300,
   bbox_inches="tight"
)

plt.show()


# ================================================================
# 37. PERFORMANCE GRAPH
# ================================================================

metric_names = [
   "Accuracy",
   "Precision",
   "Recall",
   "F1 Score"
]

metric_values = [
   accuracy * 100,
   precision * 100,
   recall * 100,
   f1 * 100
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
   metric_names,
   metric_values
)

plt.title("Random Forest Performance")
plt.xlabel("Metrics")
plt.ylabel("Score (%)")
plt.ylim(0, 100)

for bar, value in zip(bars, metric_values):
   plt.text(
       bar.get_x() + bar.get_width() / 2,
       value + 1,
       f"{value:.2f}%",
       ha="center"
   )

plt.tight_layout()

performance_path = os.path.join(
   CHARTS_FOLDER,
   "performance_graph.png"
)

plt.savefig(
   performance_path,
   dpi=300,
   bbox_inches="tight"
)

plt.show()


# ================================================================
# 38. ACTUAL VS PREDICTED CHART
# ================================================================

actual_counts = y_test.value_counts()

predicted_counts = pd.Series(
   y_pred
).value_counts()

comparison_df = pd.DataFrame({
   "Actual": actual_counts,
   "Predicted": predicted_counts
}).fillna(0)

comparison_df = comparison_df.reindex(class_labels)

plt.figure(figsize=(9, 6))

x = np.arange(len(class_labels))
width = 0.35

plt.bar(
   x - width / 2,
   comparison_df["Actual"],
   width,
   label="Actual"
)

plt.bar(
   x + width / 2,
   comparison_df["Predicted"],
   width,
   label="Predicted"
)

plt.xlabel("Placement Class")
plt.ylabel("Number of Students")
plt.title("Actual vs Predicted Placement")
plt.xticks(x, class_labels)
plt.legend()
plt.tight_layout()

actual_predicted_path = os.path.join(
   CHARTS_FOLDER,
   "actual_vs_predicted.png"
)

plt.savefig(
   actual_predicted_path,
   dpi=300,
   bbox_inches="tight"
)

plt.show()


# ================================================================
# 39. CLASS DISTRIBUTION CHART
# ================================================================

class_counts = y.value_counts()

plt.figure(figsize=(8, 6))

plt.bar(
   class_counts.index.astype(str),
   class_counts.values
)

plt.xlabel("Placement Class")
plt.ylabel("Number of Students")
plt.title("Placement Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()

class_distribution_path = os.path.join(
   CHARTS_FOLDER,
   "class_distribution.png"
)

plt.savefig(
   class_distribution_path,
   dpi=300,
   bbox_inches="tight"
)

plt.show()


# ================================================================
# 40. GET FEATURE NAMES
# ================================================================

feature_names = (
   model
   .named_steps["preprocessor"]
   .get_feature_names_out()
)


# ================================================================
# 41. FEATURE IMPORTANCE
# ================================================================

feature_importances = (
   random_forest.feature_importances_
)

feature_importance_df = pd.DataFrame({
   "Feature": feature_names,
   "Importance": feature_importances
})

feature_importance_df = (
   feature_importance_df
   .sort_values(
       by="Importance",
       ascending=False
   )
)


# ================================================================
# 42. SAVE FEATURE IMPORTANCE CSV
# ================================================================

feature_importance_path = os.path.join(
   FEATURE_FOLDER,
   "feature_importance.csv"
)

feature_importance_df.to_csv(
   feature_importance_path,
   index=False
)


# ================================================================
# 43. FEATURE IMPORTANCE CHART
# ================================================================

top_features = (
   feature_importance_df
   .head(15)
   .sort_values(by="Importance")
)

plt.figure(figsize=(10, 7))

plt.barh(
   top_features["Feature"],
   top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Feature Importances")
plt.tight_layout()

feature_chart_path = os.path.join(
   FEATURE_FOLDER,
   "feature_importance.png"
)

plt.savefig(
   feature_chart_path,
   dpi=300,
   bbox_inches="tight"
)

plt.show()


# ================================================================
# 44. RANDOM FOREST TREE VISUALIZATION
# ================================================================

print("\n")
print("=" * 80)
print(
   "             RANDOM FOREST TREE VISUALIZATION"
)
print("=" * 80)

random_forest_classifier = (
   model.named_steps["classifier"]
)

first_tree = (
   random_forest_classifier.estimators_[0]
)

print("\nDisplaying Random Forest Tree 1...")

plt.figure(figsize=(30, 18))

plot_tree(
   first_tree,
   feature_names=feature_names,
   class_names=[
       str(label)
       for label in class_labels
   ],
   filled=True,
   rounded=True,
   proportion=False,
   precision=2,
   fontsize=7
)

plt.title(
   "Random Forest - Tree 1",
   fontsize=20
)

plt.tight_layout()

tree_image_path = os.path.join(
   TREE_FOLDER,
   "random_forest_tree_1.png"
)

plt.savefig(
   tree_image_path,
   dpi=300,
   bbox_inches="tight"
)

plt.show()

print("\nRandom Forest Tree 1 displayed successfully.")
print("Tree image saved at:")
print(tree_image_path)


# ================================================================
# 45. SAVE TEST PREDICTIONS
# ================================================================

test_predictions = X_test.copy()
test_predictions["Actual"] = y_test.values
test_predictions["Predicted"] = y_pred

prediction_path = os.path.join(
   PREDICTIONS_FOLDER,
   "test_predictions.csv"
)

test_predictions.to_csv(
   prediction_path,
   index=False
)


# ================================================================
# 46. SAVE TRAINED MODEL
# ================================================================

model_path = os.path.join(
   MODEL_FOLDER,
   "random_forest_model.pkl"
)

with open(model_path, "wb") as file:
   pickle.dump(model, file)


# ================================================================
# 47. SAVE RANDOM FOREST PARAMETERS
# ================================================================

parameters_df = pd.DataFrame({
   "Parameter": [
       "Algorithm",
       "Number of Trees",
       "Criterion",
       "Maximum Depth",
       "Minimum Samples Split",
       "Minimum Samples Leaf",
       "Maximum Features",
       "Bootstrap",
       "Random State"
   ],
   "Value": [
       "Random Forest Classifier",
       random_forest.n_estimators,
       random_forest.criterion,
       random_forest.max_depth,
       random_forest.min_samples_split,
       random_forest.min_samples_leaf,
       random_forest.max_features,
       random_forest.bootstrap,
       random_forest.random_state
   ]
})

parameters_path = os.path.join(
   METRICS_FOLDER,
   "random_forest_parameters.csv"
)

parameters_df.to_csv(
   parameters_path,
   index=False
)


# ================================================================
# 48. FINAL RESULT
# ================================================================

print("\n")
print("=" * 80)
print(
   "          RANDOM FOREST COMPLETED SUCCESSFULLY"
)
print("=" * 80)

print("\nOriginal RAW dataset was NOT modified.")

print("\nDataset:")
print(DATASET_PATH)

print("\nAll outputs stored in:")
print(OUTPUT_FOLDER)

print("\n")
print("FINAL PERFORMANCE")
print("-" * 50)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\n")
print("OUTPUT FOLDERS")
print("-" * 50)

print("1. metrics")
print("2. predictions")
print("3. confusion_matrix")
print("4. charts")
print("5. random_forest_tree")
print("6. feature_importance")
print("7. model")

print("\n")
print("=" * 80)
print(
   "                PROGRAM FINISHED"
)
print("=" * 80)