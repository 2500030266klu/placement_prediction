import pandas as pd


# ============================================================
# PLACEMENT PREDICTION DATASET
# HANDLING MISSING VALUES USING PANDAS ONLY
#
# Techniques:
# 1. Deletion
# 2. Mean Imputation
# 3. Median Imputation
# 4. Model-Based Imputation
# 5. Missing Indicator Features
#
# Original dataset is NOT modified.
# All results are stored in ONE CSV file.
# ============================================================




# ------------------------------------------------------------
# 1. READ DATASET
# ------------------------------------------------------------


input_file = "C:/Users/Dell/PycharmProjects/ML_Project_New/dataset/placement_predict_50K_Raw.csv"
output_file = "C:/Users/Dell/PycharmProjects/ML_Project_New/dataset/clean_del_mean_model_M2.csv"


df = pd.read_csv(input_file)


print("=" * 70)
print("ORIGINAL PLACEMENT PREDICTION DATASET")
print("=" * 70)


print(df.head())


print("\nDataset Shape:")
print(df.shape)




# ------------------------------------------------------------
# 2. CHECK MISSING VALUES
# ------------------------------------------------------------


print("\nMissing Values in Original Dataset:")
print(df.isnull().sum())




# ------------------------------------------------------------
# 3. CREATE COPY
# ------------------------------------------------------------


# The original dataset is never modified.


df_original = df.copy()




# ============================================================
# METHOD 1: DELETION
# ============================================================


# Delete rows containing any missing value.
#dropna() deletes any row that has at least one missing value.
# dropna() removes values lin Nan or None  from data.
df_deletion = df_original.dropna().copy()


print("\n" + "=" * 70)
print("1. DELETION")
print("=" * 70)


print("Original rows:", len(df_original))
print("Rows after deletion:", len(df_deletion))
print("Rows deleted:", len(df_original) - len(df_deletion))




# ------------------------------------------------------------
# Create a deletion indicator
# ------------------------------------------------------------


# 1 = row contains missing value and would be deleted
# 0 = row contains no missing value


deletion_indicator = (
   df_original.isnull().any(axis=1)
).astype(int)




# ============================================================
# METHOD 2: MEAN IMPUTATION
# ============================================================


df_mean = df_original.copy()


# Find numerical columns


numeric_columns = df_original.select_dtypes(
   include="number"
).columns.tolist()


print("\nNumerical Columns:")
print(numeric_columns)




for column in numeric_columns:


   if df_mean[column].isnull().any():
#calculates the average value of all numbers in a specific column of a Pandas DataFrame
       mean_value = df_mean[column].mean()
# this calculates the mean, excluding any missing values, replace the empty spots.
       df_mean[column] = df_mean[column].fillna(
           mean_value
       )


       print(
           "Mean used for",
           column,
           "=",
           mean_value
       )




print("\n" + "=" * 70)
print("2. MEAN IMPUTATION")
print("=" * 70)


print(df_mean.head())




# ============================================================
# METHOD 3: MEDIAN IMPUTATION
# ============================================================


df_median = df_original.copy()




for column in numeric_columns:


   if df_median[column].isnull().any():
# calculates the middle value (median) of a specific column in a Pandas DataFrame.
       median_value = df_median[column].median()


# df_median is a typo for df finds all missing or empty NaN values in a specific column of a Pandas DataFrame
# and replaces them with that column's statistical median (middle) value
       df_median[column] = df_median[column].fillna(
           median_value
       )


       print(
           "Median used for",
           column,
           "=",
           median_value
       )




print("\n" + "=" * 70)
print("3. MEDIAN IMPUTATION")
print("=" * 70)


print(df_median.head())




# ============================================================
# METHOD 4: MODEL-BASED IMPUTATION
# ============================================================
#
# Pandas does not contain a machine-learning model.
#
# Therefore, a simple linear regression calculation is
# implemented using Pandas operations.
#
# One numerical predictor is selected automatically.
# The first numerical column having sufficient data is used.
# ============================================================


df_model = df_original.copy()


print("\n" + "=" * 70)
print("4. MODEL-BASED IMPUTATION")
print("=" * 70)




if len(numeric_columns) >= 2:


   for target_column in numeric_columns:


       # Check if target  column "PlacementStatus"contains missing values


       if not df_model[target_column].isnull().any():
           continue




       # Select another numerical column as predictor


       predictor_column = None


       for column in numeric_columns:


           if column != target_column:


               # Make sure predictor has enough values


               if (
                   df_model[column].notnull().sum()
                   >= 2
               ):


                   predictor_column = column
                   break




       # If predictor is not available


       if predictor_column is None:
           continue




       # ----------------------------------------------------
       # Training data
       # ----------------------------------------------------


       training_data = df_model[
           df_model[target_column].notnull()
           &
           df_model[predictor_column].notnull()
       ].copy()




       # Need at least two records


       if len(training_data) < 2:
           continue




       x = training_data[predictor_column]
       y = training_data[target_column]




       # ----------------------------------------------------
       # Calculate linear regression manually
       #
       # y = slope*x + intercept
       # ----------------------------------------------------
# The linear regression formula for a straight line is y = mx + b.
       # Here,  y is the predicted value, x is the input variable,
       # m is the slope of the line, and b is the y-intercept.
# linear Regression formula:
       x_mean = x.mean()
       y_mean = y.mean()




       numerator = (
           (x - x_mean) *
           (y - y_mean)
       ).sum()




       denominator = (
           (x - x_mean) ** 2
       ).sum()




       if denominator == 0:
           continue




       slope = numerator / denominator


       intercept = y_mean - (
           slope * x_mean
       )




       # ----------------------------------------------------
       # Find rows where target is missing
       # ----------------------------------------------------


       missing_rows = df_model[
           df_model[target_column].isnull()
           &
           df_model[predictor_column].notnull()
       ].copy()




       # ----------------------------------------------------
       # Predict missing values
       # ----------------------------------------------------


       if len(missing_rows) > 0:


           predicted_values = (
               slope *
               missing_rows[predictor_column]
               +
               intercept
           )




           df_model.loc[
               missing_rows.index,
               target_column
           ] = predicted_values




           print(
               "\nTarget column:",
               target_column
           )


           print(
               "Predictor column:",
               predictor_column
           )


           print(
               "Slope:",
               slope
           )


           print(
               "Intercept:",
               intercept
           )


           print(
               "Number of values predicted:",
               len(missing_rows)
           )




# ------------------------------------------------------------
# If some missing values remain, use median as fallback
# ------------------------------------------------------------


for column in numeric_columns:


   if df_model[column].isnull().any():


       median_value = df_model[column].median()


       df_model[column] = df_model[column].fillna(
           median_value
       )




print("\nModel-Based Imputation Result:")
print(df_model.head())




# ============================================================
# HANDLE CATEGORICAL COLUMNS FOR MODEL RESULT
# ============================================================


categorical_columns = df_original.select_dtypes(
   exclude="number"
).columns.tolist()




# Fill categorical missing values using mode
# This is only a supporting step for categorical data.


for column in categorical_columns:


   if df_model[column].isnull().any():


       mode_values = df_model[column].mode()


       if len(mode_values) > 0:


           df_model[column] = df_model[column].fillna(
               mode_values.iloc[0]
           )




# ============================================================
# METHOD 5: MISSING INDICATOR FEATURES
# ============================================================


print("\n" + "=" * 70)
print("5. MISSING INDICATOR FEATURES")
print("=" * 70)




df_indicator = df_original.copy()




# Create an indicator for every column


for column in df_original.columns:


   indicator_column = (
       column + "_Missing"
   )


   df_indicator[indicator_column] = (
       df_original[column].isnull()
   ).astype(int)




print("\nMissing Indicator Result:")
print(df_indicator.head())




# ============================================================
# CREATE ONE FINAL OUTPUT DATAFRAME
# ============================================================


final_result = df_original.copy()




# ------------------------------------------------------------
# A. Deletion indicator
# ------------------------------------------------------------


final_result[
   "Deletion_Row_Removed"
] = deletion_indicator




# ------------------------------------------------------------
# B. Mean-imputed values
# ------------------------------------------------------------


for column in numeric_columns:


   final_result[
       column + "_Mean_Imputed"
   ] = df_mean[column]




# ------------------------------------------------------------
# C. Median-imputed values
# ------------------------------------------------------------


for column in numeric_columns:


   final_result[
       column + "_Median_Imputed"
   ] = df_median[column]




# ------------------------------------------------------------
# D. Model-based imputed values
# ------------------------------------------------------------


for column in numeric_columns:


   final_result[
       column + "_Model_Imputed"
   ] = df_model[column]




# ------------------------------------------------------------
# E. Missing Indicator features
# ------------------------------------------------------------


for column in df_original.columns:


   final_result[
       column + "_Missing"
   ] = (
       df_original[column].isnull()
   ).astype(int)




# ============================================================
# SAVE ALL RESULTS INTO ONE CSV FILE
# ============================================================


final_result.to_csv(
   output_file,
   index=False
)




# ============================================================
# DISPLAY FINAL OUTPUT
# ============================================================


print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)


print(final_result.head())


print("\nFinal Dataset Shape:")
print(final_result.shape)




# ============================================================
# CHECK REMAINING MISSING VALUES
# ============================================================


print("\nMissing Values in Mean-Imputed Dataset:")
print(df_mean.isnull().sum())




print("\nMissing Values in Median-Imputed Dataset:")
print(df_median.isnull().sum())




print("\nMissing Values in Model-Imputed Dataset:")
print(df_model.isnull().sum())




# ============================================================
# VERIFY ORIGINAL DATASET IS NOT MODIFIED
# ============================================================


df_check = pd.read_csv(input_file)


if df_original.equals(df_check):


   print(
       "\nOriginal dataset was NOT modified."
   )


else:


   print(
       "\nWARNING: Original dataset was modified!"
   )




# ============================================================
# COMPLETION MESSAGE
# ============================================================


print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nInput file:")
print(input_file)


print("\nOutput file:")
print(output_file)


print(
   "\nAll missing-value handling techniques "
   "are stored in ONE CSV file."
)
