# ================================================================
# XGBOOST CLASSIFIER (EXTREME GRADIENT BOOSTING)
# PLACEMENT PREDICTION USING RAW DATASET
# ================================================================
#
# IMPORTANT:
# The original RAW dataset is NEVER modified.
# All preprocessing is performed on copies / inside a pipeline.
#
# OUTPUTS:
#   1. Accuracy
#   2. Precision
#   3. Recall
#   4. F1 Score
#   5. Confusion Matrix
#   6. Feature Importance Chart
#   7. Actual vs Predicted Comparison
#   8. Classification Report
#   9. Trained Model
#
# ================================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import xgboost as xgb

# ================================================================
# 1. DATASET & OUTPUT PATHS
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "XGBoost_Classifier_M3_Outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================================================================
# 2. LOAD DATASET
# ================================================================
print(f"Loading raw dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape}")

def normalize_stream(val):
    if not isinstance(val, str):
        return "CS"
    val = val.strip().upper()
    if val in ["CSE", "CS"]:
        return "CS"
    if val in ["EEE", "EE"]:
        return "EE"
    if val in ["ECE"]:
        return "ECE"
    if val in ["IT"]:
        return "IT"
    if "MECH" in val:
        return "Mechanical"
    if "CIVIL" in val:
        return "Civil"
    return "CS"

df['Stream_Norm'] = df['Stream'].apply(normalize_stream)

feature_cols = [
    'Stream_Norm',
    'CGPA',
    'HistoryOfBacklogs',
    'Internships',
    'Projects',
    'AptitudeTestScore',
    'SoftSkillsRating'
]

X = df[feature_cols].copy()
y = df['PlacementStatus'].copy()

num_cols = ['CGPA', 'Internships', 'Projects', 'AptitudeTestScore', 'SoftSkillsRating']
cat_cols = ['Stream_Norm', 'HistoryOfBacklogs']

# ================================================================
# 3. PIPELINE PREPROCESSOR
# ================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================================================
# 4. TRAIN XGBOOST MODEL
# ================================================================
print("Training XGBoost Classifier...")
xgb_model = xgb.XGBClassifier(
    n_estimators=80,
    max_depth=6,
    learning_rate=0.08,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', xgb_model)
])

pipeline.fit(X_train, y_train)

# ================================================================
# 5. EVALUATION
# ================================================================
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100
prec = precision_score(y_test, y_pred, zero_division=0) * 100
rec = recall_score(y_test, y_pred, zero_division=0) * 100
f1 = f1_score(y_test, y_pred, zero_division=0) * 100
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 50)
print("XGBOOST CLASSIFIER PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy : {acc:.2f}%")
print(f"Precision: {prec:.2f}%")
print(f"Recall   : {rec:.2f}%")
print(f"F1 Score : {f1:.2f}%")
print("=" * 50)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", cm)

# Save evaluation report text
report_path = os.path.join(OUTPUT_FOLDER, "xgboost_performance_report.txt")
with open(report_path, "w") as f:
    f.write("XGBOOST CLASSIFIER BENCHMARK REPORT\n")
    f.write("=" * 45 + "\n")
    f.write(f"Accuracy : {acc:.2f}%\n")
    f.write(f"Precision: {prec:.2f}%\n")
    f.write(f"Recall   : {rec:.2f}%\n")
    f.write(f"F1 Score : {f1:.2f}%\n\n")
    f.write("Classification Report:\n" + classification_report(y_test, y_pred))

# ================================================================
# 6. FEATURE IMPORTANCE CHART
# ================================================================
fitted_xgb = pipeline.named_steps['model']
ohe = pipeline.named_steps['prep'].named_transformers_['cat']
cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
all_features = num_cols + cat_feature_names

importances = fitted_xgb.feature_importances_
top_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(9, 5))
plt.barh(np.array(all_features)[top_idx][:8], importances[top_idx][:8], color='#1e40af')
plt.xlabel('XGBoost Relative Feature Importance (Gain)')
plt.title('XGBoost Top Features in Placement Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
chart_path = os.path.join(OUTPUT_FOLDER, "xgboost_feature_importance.png")
plt.savefig(chart_path, dpi=150)
plt.close()

# ================================================================
# 7. SAVE MODEL
# ================================================================
model_save_path = os.path.join(OUTPUT_FOLDER, "xgboost_pipeline.joblib")
joblib.dump(pipeline, model_save_path)
print(f"\nTrained XGBoost pipeline saved to: {model_save_path}")
print(f"Feature chart saved to: {chart_path}")
print(f"Report saved to: {report_path}")
