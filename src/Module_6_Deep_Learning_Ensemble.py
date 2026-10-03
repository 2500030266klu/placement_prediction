import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "M6_Deep_Learning_Ensemble_outputs")
CHARTS_DIR = os.path.join(OUTPUTS_DIR, "charts")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

for d in [MODELS_DIR, CHARTS_DIR, TABLES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("--- Module 6: Neural Networks (MLP) & Advanced Stacking Ensembling ---")
df = pd.read_csv(DATASET_PATH)

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

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor.fit(X_train)
X_train_prep = preprocessor.transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# 1. Train Multi-Layer Perceptron (Deep Neural Network Architecture)
print("Training Multi-Layer Perceptron (MLP) Neural Network...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=128,
    learning_rate='adaptive',
    learning_rate_init=0.005,
    max_iter=120,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=8
)
mlp_model.fit(X_train_prep, y_train)

mlp_preds = mlp_model.predict(X_test_prep)
mlp_probs = mlp_model.predict_proba(X_test_prep)[:, 1]

mlp_acc = round(accuracy_score(y_test, mlp_preds) * 100, 2)
mlp_prec = round(precision_score(y_test, mlp_preds, zero_division=0) * 100, 2)
mlp_rec = round(recall_score(y_test, mlp_preds, zero_division=0) * 100, 2)
mlp_f1 = round(f1_score(y_test, mlp_preds, zero_division=0) * 100, 2)
mlp_auc = round(roc_auc_score(y_test, mlp_probs) * 100, 2)

mlp_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', mlp_model)
])
joblib.dump(mlp_pipeline, os.path.join(MODELS_DIR, "mlp_pipeline.joblib"))
print(f"MLP Neural Network Performance: Acc={mlp_acc}%, Prec={mlp_prec}%, Rec={mlp_rec}%, F1={mlp_f1}%, AUC={mlp_auc}%")

# 2. Train Stacking Classifier (Meta-Learner Architecture)
print("Training Stacking Classifier (Meta-Learner)...")
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)),
    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)),
    ('gnb', GaussianNB()),
    ('lr', LogisticRegression(max_iter=500, random_state=42))
]

# Use 15,000 samples for stacking meta-fit to keep cross-validation swift
idx_stack = np.random.RandomState(42).choice(len(X_train_prep), size=15000, replace=False)
stack_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
    cv=3,
    n_jobs=-1
)
stack_model.fit(X_train_prep[idx_stack], y_train.iloc[idx_stack])

stack_preds = stack_model.predict(X_test_prep)
stack_probs = stack_model.predict_proba(X_test_prep)[:, 1]

stack_acc = round(accuracy_score(y_test, stack_preds) * 100, 2)
stack_prec = round(precision_score(y_test, stack_preds, zero_division=0) * 100, 2)
stack_rec = round(recall_score(y_test, stack_preds, zero_division=0) * 100, 2)
stack_f1 = round(f1_score(y_test, stack_preds, zero_division=0) * 100, 2)
stack_auc = round(roc_auc_score(y_test, stack_probs) * 100, 2)

stack_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', stack_model)
])
joblib.dump(stack_pipeline, os.path.join(MODELS_DIR, "stacking_pipeline.joblib"))
print(f"Stacking Meta-Learner Performance: Acc={stack_acc}%, Prec={stack_prec}%, Rec={stack_rec}%, F1={stack_f1}%, AUC={stack_auc}%")

# 3. Generate Visualizations for Module 6
print("Generating Module 6 Charts...")

# Chart 1: MLP Cross-Entropy Loss Convergence Curve
plt.figure(figsize=(9, 5))
loss_curve = mlp_model.loss_curve_
val_scores = [round(s * 100, 2) for s in mlp_model.validation_scores_] if hasattr(mlp_model, 'validation_scores_') and mlp_model.validation_scores_ is not None else []
epochs = list(range(1, len(loss_curve) + 1))

plt.plot(epochs, loss_curve, color='#4f46e5', linewidth=2.5, marker='o', markersize=4, label='Cross-Entropy Training Loss')
plt.title(f"Module 6: MLP Deep Neural Network Loss Convergence ({len(epochs)} Epochs)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Training Iterations (Epochs)", fontsize=11)
plt.ylabel("Loss Magnitude", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "mlp_loss_convergence_curve.png"), dpi=200)
plt.close()

# Chart 2: Neural Network Weight Intensity Heatmap (Layer 1 Weights)
plt.figure(figsize=(10, 5))
w1 = mlp_model.coefs_[0][:len(num_cols), :16]  # Numerical features x First 16 neurons
sns.heatmap(w1, annot=True, fmt=".2f", cmap="vlag", xticklabels=[f"N{i+1}" for i in range(16)], yticklabels=num_cols)
plt.title("Module 6: First Dense Layer (Input -> Hidden 1) Synaptic Weights", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Hidden Layer 1 Neurons (First 16 shown)", fontsize=11)
plt.ylabel("Input Features", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "neural_network_weights_heatmap.png"), dpi=200)
plt.close()

# Chart 3: Stacking Meta-Estimator Logistic Regression Coefficients
plt.figure(figsize=(9, 5))
meta_coefs = stack_model.final_estimator_.coef_[0]
estimator_names = [f"Base: {name.upper()}" for name, _ in base_estimators]
colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6']

bars = plt.barh(estimator_names, meta_coefs, color=colors, alpha=0.85)
plt.title("Module 6: Stacking Meta-Classifier Blending Weights", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Meta-Learner Logistic Regression Coefficient", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
for bar, coef in zip(bars, meta_coefs):
    plt.text(bar.get_width() + (0.05 if coef >= 0 else -0.2), bar.get_y() + bar.get_height()/2., f"{coef:.2f}", va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "stacking_meta_weights.png"), dpi=200)
plt.close()

# 4. Generate Tabular CSVs
m6_metrics_df = pd.DataFrame([
    {
        'Module': 'Module 6',
        'Model': 'Multi-Layer Perceptron (MLP Deep Neural Net)',
        'Accuracy (%)': mlp_acc,
        'Precision (%)': mlp_prec,
        'Recall (%)': mlp_rec,
        'F1-Score (%)': mlp_f1,
        'ROC-AUC (%)': mlp_auc,
        'Parameters': 'Architecture=[64, 32, 16], activation=relu, solver=adam, alpha=0.001'
    },
    {
        'Module': 'Module 6',
        'Model': 'Stacking Meta-Ensemble Classifier',
        'Accuracy (%)': stack_acc,
        'Precision (%)': stack_prec,
        'Recall (%)': stack_rec,
        'F1-Score (%)': stack_f1,
        'ROC-AUC (%)': stack_auc,
        'Parameters': 'Base=[RF, KNN, GNB, LR], Meta=LogisticRegression, CV=3'
    }
])
m6_metrics_df.to_csv(os.path.join(TABLES_DIR, "m6_deep_learning_metrics.csv"), index=False)

# Hyperparameter specification table
hyperparams_df = pd.DataFrame([
    {'Layer': 'Input Layer', 'Neurons / Dimension': X_train_prep.shape[1], 'Activation': 'Linear / Identity', 'Function': 'Standardized Student Academic & Skill Vectors'},
    {'Layer': 'Hidden Layer 1', 'Neurons / Dimension': 64, 'Activation': 'ReLU', 'Function': 'High-dimensional non-linear feature interaction extraction'},
    {'Layer': 'Hidden Layer 2', 'Neurons / Dimension': 32, 'Activation': 'ReLU', 'Function': 'Latent employability abstract feature abstraction'},
    {'Layer': 'Hidden Layer 3', 'Neurons / Dimension': 16, 'Activation': 'ReLU', 'Function': 'Pre-classification compression & regularization'},
    {'Layer': 'Output Layer', 'Neurons / Dimension': 1, 'Activation': 'Logistic Sigmoid', 'Function': 'Binary placement probability prediction [0, 1]'}
])
hyperparams_df.to_csv(os.path.join(TABLES_DIR, "mlp_layer_hyperparameters.csv"), index=False)

# 5. Technical Report
report_text = f"""================================================================================
MODULE 6 TECHNICAL REPORT: DEEP NEURAL NETWORKS (MLP) & STACKING ENSEMBLE
================================================================================
Academic Context: Deep Feedforward Architectures, Backpropagation & Meta-Ensembling
Dataset: University Placement Prediction Dataset (50,000 records)
Holdout Test Size: 10,000 records (20%)

--------------------------------------------------------------------------------
1. MULTI-LAYER PERCEPTRON (MLP) DEEP NEURAL NETWORK
--------------------------------------------------------------------------------
Architecture Topology:
    - Input Features : {X_train_prep.shape[1]} standardized inputs
    - Hidden Layer 1 : 64 Dense Neurons (Activation: Rectified Linear Unit - ReLU)
    - Hidden Layer 2 : 32 Dense Neurons (Activation: ReLU)
    - Hidden Layer 3 : 16 Dense Neurons (Activation: ReLU)
    - Output Neuron  : 1 Sigmoid Logit Neuron (Placement Probability)
Optimization & Regularization:
    - Loss Function      : Binary Cross-Entropy Loss
    - Optimizer          : Adam (Adaptive Moment Estimation, beta1=0.9, beta2=0.999)
    - Initial Step Size  : 0.005 with adaptive decay
    - L2 Regularization  : alpha = 0.001 (prevents co-adaptation of weights)
    - Early Stopping     : Enabled (patience = 8 epochs)
Performance Results:
    - Test Accuracy  : {mlp_acc}%
    - Test Precision : {mlp_prec}%
    - Test Recall    : {mlp_rec}%
    - Test F1-Score  : {mlp_f1}%
    - Area Under ROC : {mlp_auc}%
    - Total Epochs   : {len(loss_curve)} iterations

--------------------------------------------------------------------------------
2. STACKING META-ENSEMBLE CLASSIFIER
--------------------------------------------------------------------------------
Architecture Topology:
    - Level-0 Estimators : Random Forest (30 trees), KNN (k=5), GaussianNB, Logistic Regression
    - Cross-Validation   : 3-Fold out-of-fold meta-prediction generation
    - Level-1 Meta-Model : Regularized Logistic Regression
Performance Results:
    - Test Accuracy  : {stack_acc}%
    - Test Precision : {stack_prec}%
    - Test Recall    : {stack_rec}%
    - Test F1-Score  : {stack_f1}%
    - Area Under ROC : {stack_auc}%
Key Insight:
    Stacking combines non-linear partitioning (Random Forest), local distance geometry
    (KNN), and Bayesian likelihood into an optimal meta-decision boundary.

Status: Artifacts generated and persisted successfully.
================================================================================
"""

with open(os.path.join(REPORTS_DIR, "m6_neural_networks_report.txt"), "w") as f:
    f.write(report_text)

print("Module 6 execution complete! Models and outputs generated successfully.")
