import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# Define the path to your specific CSV dataset
# 1. Load Dataset


print("1. Load the Dataset")
file_path = 'D:/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv'


try:
   # Read the CSV file into a DataFrame
   df = pd.read_csv(file_path)


   # Display the complete table
   print("-----------------------------------")
   print("1. Dataset Contents:")
   print("-----------------------------------")
   print(df)


   # Display additional information
   print("-----------------------------------")
   print("\n 2. Number of Rows and Columns:", df.shape)
   print("-----------------------------------")


   print("\n3. Column Names:")
   print("-----------------------------------")
   print(df.columns.tolist())
   # Configure pandas to display all columns in the console window
   pd.set_option('display.max_columns', None)
   pd.set_option('display.width', 1000)


   # Display the first 10 rows of the table dataset
   print("-----------------------------------")
   print("\n 4. --- Placement Predict CSV Dataset Table View ---")
   print("-----------------------------------")
   print("Dataset first 10 records")
   print("-----------------------------------")
   print(df.head(10))
   print("********************************************************************************************************************************************************************************************************************************************************************")
   print("Dataset Last 10 records")
   print("-----------------------------------")
   print(df.tail(10))


   # 2. Understand the Dataset
   print("-----------------------------------")
   print("2. Understand the Dataset")
   print("-----------------------------------")


   # Display the data types of each column
   print("-----------------------------------")
   print("\n 5. Data Types of Columns:")
   print(df.dtypes)




   # to print a line of 60 equal signs (=) to the screen.
   print("=" * 60)


   # Display column names with data types in a formatted way
   print(" 6. Display column names with data types in a formatted way")
   print("\nColumn Name\t\tData Type")
   print("-" * 35)
   for column in df.columns:
       print(f"{column:<20} {df[column].dtype}")


   # Understand the datatypes, column names, and missing values
   print("-----------------------------------")
   print("7. Dataset Summary and Information")
   print("-----------------------------------")
   print(df.info())
   print("\n"+"="*50+"\n")


   # display only numeric columns
   print("8. Display Numeric Columns")
   print("-----------------------------------")
   # Select only numerical columns
   numeric_df = df.select_dtypes(include=['int64', 'float64'])
   # Display numerical columns
   print("Numerical Columns:")
   print(numeric_df)
   print("9. Missing Values in Numeric Attributes")
   print(numeric_df.isnull().sum())
   print("\n10. Total Missing Numeric Values:",
         numeric_df.isnull().sum().sum())




   # Get float column names
   float_columns = df.select_dtypes(include=['float64']).columns
   print("-----------------------------------")
   print("10. Float Attribute Names:")
   print("-----------------------------------")
   for column in float_columns:
       print(column)
   print("11. Missing Values in Float Attributes")
   print("=" * 50)
   print(float_columns.isnull().sum())
   print("\n12. Total Missing Float Values:",
         float_columns.isnull().sum().sum())




   # Select categorical(object) columns
   categorical_df = df.select_dtypes(include=['object'])
   # Display categorical columns
   print("-----------------------------------")
   print("12. Display Categorical (Object) Attributes:")
   print("-----------------------------------")
   print(categorical_df)
   print("Missing Values in Categorical Attributes")
   print("=" * 50)
   print(categorical_df.isnull().sum())
   print("\n13. Total Missing Categorical Values:",
         categorical_df.isnull().sum().sum())




   # Display missing values in each column
   print("-----------------------------------")
   print("14. Missing Values in Each Column")
   print("-" * 40)
   print(df.isnull().sum())


   # Total missing values
   total_missing = df.isnull().sum().sum()
   print("-----------------------------------")
   print("15. Total Missing Values:", total_missing)


   # Count duplicate rows
   print("-----------------------------------")
   duplicate_count = df.duplicated().sum()
   print("16. Number of Duplicate Records:", duplicate_count)


   # GEnerate stastical metrics for numeric columns
   print("-----------------------------------")
   print(" 17. Statistical Overview")
   print("-----------------------------------")
   print(df.describe())


   # display histogram for CGPA numerical Attribute
   print("18. Display Histogram of CGPA Attribute")
   plt.figure(figsize=(8, 5))
   plt.hist(df['CGPA'], bins=10, edgecolor='black')
   plt.title("Histogram of CGPA")
   plt.xlabel("CGPA")
   plt.ylabel("Frequency")
   plt.grid(True)
   plt.show()


except FileNotFoundError:
   print(f"Error: The file at '{file_path}' was not found.")
except Exception as e:
   print(f"An error occurred: {e}")
