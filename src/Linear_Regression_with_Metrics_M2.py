# ============================================================
# LINEAR REGRESSION - PLACEMENT PREDICTION DATASET
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
   mean_squared_error,
   mean_absolute_error,
   r2_score
)

# ============================================================
# 1. FILE PATHS (DYNAMIC & RELATIVE)
# ============================================================

# Automatically determine project root directory (one level up from src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Linear_Regression_with_Metrics_M2")
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
   raise FileNotFoundError(
       f"Dataset not found: {DATASET_PATH}"
   )

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("LINEAR REGRESSION - PLACEMENT PREDICTION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# 3. DISPLAY COLUMN NAMES
# ============================================================

print("\nDataset Columns:")
for column in df.columns:
   print(column)


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

feature_columns = [
   "CGPA",
   "AptitudeTestScore",
   "CodingTestScore",
   "MockInterviewScore"
]

target_column = "PlacementStatus"


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = feature_columns + [target_column]

missing_columns = [
   column
   for column in required_columns
   if column not in df.columns
]

if missing_columns:
   print("\nERROR!")
   print("The following columns were not found:")
   print(missing_columns)

   print("\nAvailable columns are:")
   print(list(df.columns))

   raise ValueError(
       "Please change feature_columns and target_column "
       "according to your dataset."
   )


# ============================================================
# 6. CREATE MODEL DATA
# ============================================================

model_df = df[required_columns].copy()


# ============================================================
# 8. DEFINE X AND Y
# ============================================================

X = model_df[feature_columns]
y = model_df[target_column]


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
   X,
   y,
   test_size=0.20,
   random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 11. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nModel training completed.")


# ============================================================
# 12. MODEL COEFFICIENTS
# ============================================================

print("\nIntercept is b0:")
print(model.intercept_)

print("\nCoefficients (b1,b2,b3,b4) values:")
coefficient_df = pd.DataFrame({
   "Feature": feature_columns,
   "Coefficient": model.coef_
})
print(coefficient_df)


# ============================================================
# 13. LINEAR REGRESSION EQUATION
# ============================================================

equation = f"{target_column} = {model.intercept_:.4f}"

for feature, coefficient in zip(
       feature_columns,
       model.coef_):

   equation += (
       f" + ({coefficient:.4f} × {feature})"
   )

print("\nLinear Regression Equation:")
print(equation)


# ============================================================
# 14. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 15. EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 16. CREATE PREDICTION RESULTS
# ============================================================

results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred
results["Residual"] = (
   results["Actual"] -
   results["Predicted"]
)
results["Absolute_Error"] = abs(
   results["Residual"]
)


# ============================================================
# 17. SAVE PREDICTION RESULTS
# ============================================================

prediction_file = os.path.join(
   OUTPUT_FOLDER,
   "linear_regression_predictions.csv"
)

results.to_csv(
   prediction_file,
   index=False
)

print("\nPrediction results saved to:")
print(prediction_file)


# ============================================================
# 18. SAVE MODEL COEFFICIENTS
# ============================================================

coefficient_file = os.path.join(
   OUTPUT_FOLDER,
   "linear_regression_coefficients.csv"
)

coefficient_df.to_csv(
   coefficient_file,
   index=False
)


# ============================================================
# 19. SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame({
   "Metric": [
       "MAE",
       "MSE",
       "RMSE",
       "R2"
   ],
   "Value": [
       mae,
       mse,
       rmse,
       r2
   ]
})

metrics_file = os.path.join(
   OUTPUT_FOLDER,
   "linear_regression_metrics.csv"
)

metrics_df.to_csv(
   metrics_file,
   index=False
)


# ============================================================
# 20. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
   y_test,
   y_pred,
   alpha=0.6
)

minimum = min(
   y_test.min(),
   y_pred.min()
)

maximum = max(
   y_test.max(),
   y_pred.max()
)

plt.plot(
   [minimum, maximum],
   [minimum, maximum],
   linestyle="--"
)

plt.xlabel("Actual placement")
plt.ylabel("Predicted placement")
plt.title(
   "Linear Regression: Actual vs Predicted placement"
)
plt.grid(True)

actual_predicted_image = os.path.join(
   IMAGE_FOLDER,
   "actual_vs_predicted.png"
)

plt.savefig(
   actual_predicted_image,
   dpi=300,
   bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. RESIDUAL GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
   y_pred,
   results["Residual"],
   alpha=0.6
)

plt.axhline(
   y=0,
   linestyle="--"
)

plt.xlabel("Predicted Placement")
plt.ylabel("Residual")
plt.title(
   "Residual Plot - Linear Regression"
)
plt.grid(True)

residual_image = os.path.join(
   IMAGE_FOLDER,
   "residual_plot.png"
)

plt.savefig(
   residual_image,
   dpi=300,
   bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. COEFFICIENT GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
   coefficient_df["Feature"],
   coefficient_df["Coefficient"]
)

plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title(
   "Linear Regression Feature Coefficients"
)

plt.xticks(
   rotation=30,
   ha="right"
)

plt.grid(
   axis="y"
)

coefficient_image = os.path.join(
   IMAGE_FOLDER,
   "feature_coefficients.png"
)

plt.savefig(
   coefficient_image,
   dpi=300,
   bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. SAVE EQUATION
# ============================================================

equation_file = os.path.join(
   OUTPUT_FOLDER,
   "linear_regression_equation.txt"
)

with open(
   equation_file,
   "w",
   encoding="utf-8"
) as file:

   file.write(
       "Linear Regression Equation\n"
   )
   file.write(
       "=" * 40 + "\n"
   )
   file.write(
       equation
   )


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput Folder:")
print(OUTPUT_FOLDER)

print("\nGenerated Files:")
print("- linear_regression_predictions.csv")
print("- linear_regression_coefficients.csv")
print("- linear_regression_metrics.csv")
print("- linear_regression_equation.txt")

print("\nGenerated Images:")
print("- actual_vs_predicted.png")
print("- residual_plot.png")
print("- feature_coefficients.png")