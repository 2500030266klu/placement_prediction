# ============================================================
# PLACEMENT PREDICTION - PREPROCESSED DATASET
# K-MEANS AND K-MEANS++ CLUSTERING
#
# Features:
#   CGPA
#   HistoryOfBacklogs
#   Internships
#   AptituteTestScore
#
# Methods:
#   1. K-Means
#   2. K-Means++
#   3. Elbow Method
#   4. Silhouette Score
#   5. PCA visualization
#
# IMPORTANT:
# - Uses the PREPROCESSED dataset
# - Original dataset is NOT modified
# - Outputs are stored in separate folders
# ============================================================


import os
import warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from sklearn.metrics import (
   silhouette_score,
   calinski_harabasz_score,
   davies_bouldin_score,
   accuracy_score
)


warnings.filterwarnings("ignore")




# ============================================================
# 1. INPUT FILE
# ============================================================


INPUT_FILE = (
   r"C:/Users/Dell/PycharmProjects/ML_Project_New/dataset"
   r"\final_preprocess_M2.csv"
)




# ============================================================
# 2. OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = (
    r"C:/Users/anil pandey/PycharmProjects/placement_prediction/outputs"
    r"/K_Means_K++Means_Elbow_Silhoute_M4_Outputs"
)



# ============================================================
# 3. CREATE SEPARATE OUTPUT FOLDERS
# ============================================================


KMEANS_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "KMeans"
)


KMEANS_PP_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "KMeansPlusPlus"
)


ELBOW_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "Elbow"
)


SILHOUETTE_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "Silhouette"
)


ACCURACY_FOLDER = os.path.join(
   OUTPUT_FOLDER,
   "Accuracy"
)




for folder in [
   KMEANS_FOLDER,
   KMEANS_PP_FOLDER,
   ELBOW_FOLDER,
   SILHOUETTE_FOLDER,
   ACCURACY_FOLDER
]:
   os.makedirs(folder, exist_ok=True)




# ============================================================
# 4. LOAD PREPROCESSED DATASET
# ============================================================


print("=" * 75)
print("K-MEANS AND K-MEANS++")
print("PLACEMENT PREDICTION - PREPROCESSED DATASET")
print("=" * 75)


if not os.path.exists(INPUT_FILE):


   raise FileNotFoundError(
       "\nPreprocessed dataset was not found.\n\n"
       "Check this path:\n"
       + INPUT_FILE
   )




data = pd.read_csv(INPUT_FILE)


print("\nPreprocessed dataset loaded successfully.")


print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])




# ============================================================
# 5. DISPLAY COLUMNS
# ============================================================


print("\nAvailable columns:")
print("-" * 75)


for i, column in enumerate(data.columns, 1):
   print(i, ".", column)




# ============================================================
# 6. COLUMN-NAME NORMALIZATION
# ============================================================


def normalize_column_name(column):


   return (
       str(column)
       .strip()
       .lower()
       .replace("_", "")
       .replace(" ", "")
       .replace("-", "")
   )




normalized_columns = {
   normalize_column_name(column): column
   for column in data.columns
}




# ============================================================
# 7. REQUIRED FEATURES
# ============================================================


required_features = [
   "CGPA",
   "HistoryOfBacklogs",
   "Internships",
   "AptituteTestScore"
]




# ============================================================
# 8. FIND FEATURES
# ============================================================


feature_columns = []




for feature in required_features:


   normalized_feature = normalize_column_name(
       feature
   )


   if normalized_feature in normalized_columns:


       feature_columns.append(
           normalized_columns[
               normalized_feature
           ]
       )


   else:


       # Handle Aptitude spelling variations
       if feature == "AptituteTestScore":


           alternatives = [
               "AptitudeTestScore",
               "Aptitude_Test_Score",
               "AptitudeScore",
               "AptituteScore"
           ]


           found = None


           for alternative in alternatives:


               normalized_alt = normalize_column_name(
                   alternative
               )


               if normalized_alt in normalized_columns:


                   found = normalized_columns[
                       normalized_alt
                   ]


                   break


           if found is not None:


               feature_columns.append(found)


           else:


               raise ValueError(
                   "\nAptitude Test Score column was not found."
                   "\n\nAvailable columns:\n"
                   + "\n".join(
                       data.columns.astype(str)
                   )
               )


       else:


           raise ValueError(
               f"\nRequired column '{feature}' "
               "was not found."
           )




# ============================================================
# 9. DISPLAY SELECTED FEATURES
# ============================================================


print("\n" + "=" * 75)
print("FEATURES USED FOR CLUSTERING")
print("=" * 75)


for feature in feature_columns:


   print("✓", feature)




# ============================================================
# 10. SELECT FOUR FEATURES
# ============================================================


X = data[
   feature_columns
].copy()




# ============================================================
# 11. CONVERT TO NUMERIC
# ============================================================


for column in feature_columns:


   X[column] = pd.to_numeric(
       X[column],
       errors="coerce"
   )




# ============================================================
# 12. HANDLE INFINITE VALUES
# ============================================================


X = X.replace(
   [np.inf, -np.inf],
   np.nan
)




# ============================================================
# 13. HANDLE MISSING VALUES
# ============================================================


print("\nMissing values:")


print(
   X.isnull().sum()
)




for column in feature_columns:


   X[column] = X[column].fillna(
       X[column].median()
   )




print("\nMissing values after preprocessing:")


print(
   X.isnull().sum()
)




# ============================================================
# 14. STANDARDISATION
# ============================================================
#
# Even though this is a preprocessed dataset, K-Means is
# distance-based. Therefore, the four clustering variables
# are standardised before clustering.
# ============================================================


scaler = StandardScaler()


X_scaled = scaler.fit_transform(
   X
)




print("\nFeature standardisation completed.")




# ============================================================
# 15. K RANGE
# ============================================================


K_VALUES = range(
   2,
   11
)




# ============================================================
# 16. K-MEANS - ELBOW AND SILHOUETTE
# ============================================================


print("\n" + "=" * 75)
print("K-MEANS ANALYSIS")
print("=" * 75)


kmeans_inertia = []
kmeans_silhouette = []




for k in K_VALUES:


   model = KMeans(
       n_clusters=k,
       init="random",
       n_init=10,
       random_state=42
   )


   labels = model.fit_predict(
       X_scaled
   )


   inertia = model.inertia_


   silhouette = silhouette_score(
       X_scaled,
       labels
   )


   kmeans_inertia.append(
       inertia
   )


   kmeans_silhouette.append(
       silhouette
   )


   print(
       f"K = {k:2d} | "
       f"Inertia = {inertia:.4f} | "
       f"Silhouette = {silhouette:.4f}"
   )




# ============================================================
# 17. K-MEANS++ - ELBOW AND SILHOUETTE
# ============================================================


print("\n" + "=" * 75)
print("K-MEANS++ ANALYSIS")
print("=" * 75)


kmeans_pp_inertia = []
kmeans_pp_silhouette = []




for k in K_VALUES:


   model = KMeans(
       n_clusters=k,
       init="k-means++",
       n_init=10,
       random_state=42
   )


   labels = model.fit_predict(
       X_scaled
   )


   inertia = model.inertia_


   silhouette = silhouette_score(
       X_scaled,
       labels
   )


   kmeans_pp_inertia.append(
       inertia
   )


   kmeans_pp_silhouette.append(
       silhouette
   )


   print(
       f"K = {k:2d} | "
       f"Inertia = {inertia:.4f} | "
       f"Silhouette = {silhouette:.4f}"
   )




# ============================================================
# 18. SELECT BEST K
# ============================================================
#
# Maximum Silhouette Score is used to select K automatically.
# The Elbow graph should also be inspected.
# ============================================================


best_k_kmeans = list(K_VALUES)[
   np.argmax(
       kmeans_silhouette
   )
]




best_k_kmeans_pp = list(K_VALUES)[
   np.argmax(
       kmeans_pp_silhouette
   )
]




print("\n" + "=" * 75)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 75)


print(
   "K-Means best K       :",
   best_k_kmeans
)


print(
   "K-Means++ best K     :",
   best_k_kmeans_pp
)




# ============================================================
# 19. ELBOW GRAPH - K-MEANS
# ============================================================


plt.figure(
   figsize=(10, 6)
)


plt.plot(
   list(K_VALUES),
   kmeans_inertia,
   marker="o"
)


plt.xlabel(
   "Number of Clusters (K)"
)


plt.ylabel(
   "Inertia"
)


plt.title(
   "Elbow Method - K-Means"
)


plt.xticks(
   list(K_VALUES)
)


plt.grid(True)


plt.tight_layout()


plt.savefig(
   os.path.join(
       ELBOW_FOLDER,
       "elbow_kmeans.png"
   ),
   dpi=300,
   bbox_inches="tight"
)


plt.close()




# ============================================================
# 20. ELBOW GRAPH - K-MEANS++
# ============================================================


plt.figure(
   figsize=(10, 6)
)


plt.plot(
   list(K_VALUES),
   kmeans_pp_inertia,
   marker="o"
)


plt.xlabel(
   "Number of Clusters (K)"
)


plt.ylabel(
   "Inertia"
)


plt.title(
   "Elbow Method - K-Means++"
)


plt.xticks(
   list(K_VALUES)
)


plt.grid(True)


plt.tight_layout()


plt.savefig(
   os.path.join(
       ELBOW_FOLDER,
       "elbow_kmeans_plus_plus.png"
   ),
   dpi=300,
   bbox_inches="tight"
)


plt.close()




# ============================================================
# 21. SILHOUETTE GRAPH - K-MEANS
# ============================================================


plt.figure(
   figsize=(10, 6)
)


plt.plot(
   list(K_VALUES),
   kmeans_silhouette,
   marker="o"
)


plt.xlabel(
   "Number of Clusters (K)"
)


plt.ylabel(
   "Silhouette Score"
)


plt.title(
   "Silhouette Score - K-Means"
)


plt.xticks(
   list(K_VALUES)
)


plt.grid(True)


plt.tight_layout()


plt.savefig(
   os.path.join(
       SILHOUETTE_FOLDER,
       "silhouette_kmeans.png"
   ),
   dpi=300,
   bbox_inches="tight"
)


plt.close()




# ============================================================
# 22. SILHOUETTE GRAPH - K-MEANS++
# ============================================================


plt.figure(
   figsize=(10, 6)
)


plt.plot(
   list(K_VALUES),
   kmeans_pp_silhouette,
   marker="o"
)


plt.xlabel(
   "Number of Clusters (K)"
)


plt.ylabel(
   "Silhouette Score"
)


plt.title(
   "Silhouette Score - K-Means++"
)


plt.xticks(
   list(K_VALUES)
)


plt.grid(True)


plt.tight_layout()


plt.savefig(
   os.path.join(
       SILHOUETTE_FOLDER,
       "silhouette_kmeans_plus_plus.png"
   ),
   dpi=300,
   bbox_inches="tight"
)


plt.close()




# ============================================================
# 23. FINAL K-MEANS
# ============================================================


kmeans_model = KMeans(
   n_clusters=best_k_kmeans,
   init="random",
   n_init=10,
   random_state=42
)


kmeans_labels = kmeans_model.fit_predict(
   X_scaled
)




# ============================================================
# 24. FINAL K-MEANS++
# ============================================================


kmeans_pp_model = KMeans(
   n_clusters=best_k_kmeans_pp,
   init="k-means++",
   n_init=10,
   random_state=42
)


kmeans_pp_labels = kmeans_pp_model.fit_predict(
   X_scaled
)




# ============================================================
# 25. SAVE K-MEANS CLUSTERED DATA
# ============================================================


kmeans_result = X.copy()


kmeans_result[
   "KMeans_Cluster"
] = kmeans_labels + 1




kmeans_result.to_csv(
   os.path.join(
       KMEANS_FOLDER,
       "kmeans_clustered_data.csv"
   ),
   index=False
)




# ============================================================
# 26. SAVE K-MEANS++ CLUSTERED DATA
# ============================================================


kmeans_pp_result = X.copy()


kmeans_pp_result[
   "KMeansPlusPlus_Cluster"
] = kmeans_pp_labels + 1




kmeans_pp_result.to_csv(
   os.path.join(
       KMEANS_PP_FOLDER,
       "kmeans_plus_plus_clustered_data.csv"
   ),
   index=False
)




# ============================================================
# 27. FINAL METRICS - K-MEANS
# ============================================================


km_silhouette = silhouette_score(
   X_scaled,
   kmeans_labels
)


km_calinski = calinski_harabasz_score(
   X_scaled,
   kmeans_labels
)


km_davies = davies_bouldin_score(
   X_scaled,
   kmeans_labels
)




# ============================================================
# 28. FINAL METRICS - K-MEANS++
# ============================================================


pp_silhouette = silhouette_score(
   X_scaled,
   kmeans_pp_labels
)


pp_calinski = calinski_harabasz_score(
   X_scaled,
   kmeans_pp_labels
)


pp_davies = davies_bouldin_score(
   X_scaled,
   kmeans_pp_labels
)




# ============================================================
# 29. SAVE K-MEANS METRICS
# ============================================================


kmeans_metrics = pd.DataFrame({


   "Method": ["K-Means"],


   "Number_of_Clusters": [
       best_k_kmeans
   ],


   "Inertia": [
       kmeans_model.inertia_
   ],


   "Silhouette_Score": [
       km_silhouette
   ],


   "Calinski_Harabasz_Score": [
       km_calinski
   ],


   "Davies_Bouldin_Score": [
       km_davies
   ]


})




kmeans_metrics.to_csv(
   os.path.join(
       KMEANS_FOLDER,
       "kmeans_metrics.csv"
   ),
   index=False
)




# ============================================================
# 30. SAVE K-MEANS++ METRICS
# ============================================================


kmeans_pp_metrics = pd.DataFrame({


   "Method": ["K-Means++"],


   "Number_of_Clusters": [
       best_k_kmeans_pp
   ],


   "Inertia": [
       kmeans_pp_model.inertia_
   ],


   "Silhouette_Score": [
       pp_silhouette
   ],


   "Calinski_Harabasz_Score": [
       pp_calinski
   ],


   "Davies_Bouldin_Score": [
       pp_davies
   ]


})




kmeans_pp_metrics.to_csv(
   os.path.join(
       KMEANS_PP_FOLDER,
       "kmeans_plus_plus_metrics.csv"
   ),
   index=False
)




# ============================================================
# 31. 3-D K-MEANS CLUSTERING GRAPH
# ============================================================
#
# Four features cannot be directly plotted in a normal 3-D
# graph. The first three features are shown here.
#
# All FOUR features are used for actual clustering.
# ============================================================


fig = plt.figure(
   figsize=(11, 8)
)


ax = fig.add_subplot(
   111,
   projection="3d"
)




scatter = ax.scatter(
   X[feature_columns[0]],
   X[feature_columns[1]],
   X[feature_columns[2]],
   c=kmeans_labels,
   s=20,
   alpha=0.7
)




ax.set_xlabel(
   feature_columns[0]
)


ax.set_ylabel(
   feature_columns[1]
)


ax.set_zlabel(
   feature_columns[2]
)


ax.set_title(
   "K-Means Clustering"
)




fig.colorbar(
   scatter,
   ax=ax,
   label="Cluster"
)




plt.tight_layout()


plt.savefig(
   os.path.join(
       KMEANS_FOLDER,
       "kmeans_3D_clustering.png"
   ),
   dpi=300,
   bbox_inches="tight"
)


plt.close()




# ============================================================
# 32. 3-D K-MEANS++ CLUSTERING GRAPH
# ============================================================


fig = plt.figure(
   figsize=(11, 8)
)


ax = fig.add_subplot(
   111,
   projection="3d"
)




scatter = ax.scatter(
   X[feature_columns[0]],
   X[feature_columns[1]],
   X[feature_columns[2]],
   c=kmeans_pp_labels,
   s=20,
   alpha=0.7
)




ax.set_xlabel(
   feature_columns[0]
)


ax.set_ylabel(
   feature_columns[1]
)


ax.set_zlabel(
   feature_columns[2]
)


ax.set_title(
   "K-Means++ Clustering"
)




fig.colorbar(
   scatter,
   ax=ax,
   label="Cluster"
)




plt.tight_layout()


plt.savefig(
   os.path.join(
       KMEANS_PP_FOLDER,
       "kmeans_plus_plus_3D_clustering.png"
   ),
   dpi=300,
   bbox_inches="tight"
)


plt.close()




# ============================================================
# 33. PCA 2-D VISUALISATION
# ============================================================
#
# PCA is ONLY used here to visualize the four-dimensional
# clustering results in two dimensions.
#
# PCA is NOT used to perform the clustering.
# ============================================================


pca = PCA(
   n_components=2
)


X_pca = pca.fit_transform(
   X_scaled
)




# ============================================================
# 34. K-MEANS PCA GRAPH
# ============================================================


plt.figure(
   figsize=(10, 7)
)


scatter = plt.scatter(
   X_pca[:, 0],
   X_pca[:, 1],
   c=kmeans_labels,
   s=20,
   alpha=0.7
)


plt.xlabel(
   "Principal Component 1"
)


plt.ylabel(
   "Principal Component 2"
)


plt.title(
   "K-Means Clusters - PCA Visualization"
)


plt.colorbar(
   scatter,
   label="Cluster"
)


plt.grid(True)


plt.tight_layout()


plt.savefig(
   os.path.join(
       KMEANS_FOLDER,
       "kmeans_PCA_clustering.png"
   ),
   dpi=300,
   bbox_inches="tight"
)


plt.close()




# ============================================================
# 35. K-MEANS++ PCA GRAPH
# ============================================================


plt.figure(
   figsize=(10, 7)
)


scatter = plt.scatter(
   X_pca[:, 0],
   X_pca[:, 1],
   c=kmeans_pp_labels,
   s=20,
   alpha=0.7
)


plt.xlabel(
   "Principal Component 1"
)


plt.ylabel(
   "Principal Component 2"
)


plt.title(
   "K-Means++ Clusters - PCA Visualization"
)


plt.colorbar(
   scatter,
   label="Cluster"
)


plt.grid(True)


plt.tight_layout()


plt.savefig(
   os.path.join(
       KMEANS_PP_FOLDER,
       "kmeans_plus_plus_PCA_clustering.png"
   ),
   dpi=300,
   bbox_inches="tight"
)


plt.close()




# ============================================================
# 36. CLUSTER COUNTS
# ============================================================


print("\n" + "=" * 75)
print("K-MEANS CLUSTER COUNTS")
print("=" * 75)


print(
   kmeans_result[
       "KMeans_Cluster"
   ].value_counts().sort_index()
)




print("\n" + "=" * 75)
print("K-MEANS++ CLUSTER COUNTS")
print("=" * 75)


print(
   kmeans_pp_result[
       "KMeansPlusPlus_Cluster"
   ].value_counts().sort_index()
)




# ============================================================
# 37. FIND PLACEMENT TARGET
# ============================================================


target_candidates = [
   "Placement",
   "Placed",
   "Status",
   "Target"
]




target_column = None




for candidate in target_candidates:


   normalized_candidate = normalize_column_name(
       candidate
   )


   if normalized_candidate in normalized_columns:


       target_column = normalized_columns[
           normalized_candidate
       ]


       break




# ============================================================
# 38. CLUSTER MATCHING ACCURACY
# ============================================================


def cluster_matching_accuracy(
       true_values,
       cluster_values
):


   true_values = pd.Series(
       true_values
   ).reset_index(drop=True)


   cluster_values = pd.Series(
       cluster_values
   ).reset_index(drop=True)


   predicted_values = np.empty(
       len(cluster_values),
       dtype=object
   )




   for cluster in np.unique(
       cluster_values
   ):


       indexes = np.where(
           cluster_values == cluster
       )[0]




       cluster_targets = (
           true_values.iloc[
               indexes
           ]
       )




       majority = (
           cluster_targets
           .mode()
       )




       if len(majority) > 0:


           predicted_values[
               indexes
           ] = majority.iloc[0]




   return accuracy_score(
       true_values,
       predicted_values
   )




# ============================================================
# 39. CALCULATE ACCURACY IF TARGET EXISTS
# ============================================================


if target_column is not None:


   print("\n" + "=" * 75)
   print("PLACEMENT CLUSTER MATCHING ACCURACY")
   print("=" * 75)


   target = data[
       target_column
   ].copy()




   valid = target.notna()




   target_valid = target[
       valid
   ].reset_index(drop=True)




   km_labels_valid = pd.Series(
       kmeans_labels,
       index=data.index
   )[valid].reset_index(drop=True)




   pp_labels_valid = pd.Series(
       kmeans_pp_labels,
       index=data.index
   )[valid].reset_index(drop=True)




   km_accuracy = cluster_matching_accuracy(
       target_valid,
       km_labels_valid
   )




   pp_accuracy = cluster_matching_accuracy(
       target_valid,
       pp_labels_valid
   )




   print(
       "Target column:",
       target_column
   )


   print(
       f"K-Means   : {km_accuracy * 100:.2f}%"
   )


   print(
       f"K-Means++ : {pp_accuracy * 100:.2f}%"
   )




   accuracy_result = pd.DataFrame({


       "Method": [
           "K-Means",
           "K-Means++"
       ],


       "Cluster_Matching_Accuracy": [
           km_accuracy,
           pp_accuracy
       ],


       "Accuracy_Percentage": [
           km_accuracy * 100,
           pp_accuracy * 100
       ]


   })




   accuracy_result.to_csv(
       os.path.join(
           ACCURACY_FOLDER,
           "placement_cluster_matching_accuracy.csv"
       ),
       index=False
   )




else:


   print(
       "\nNo Placement/Placed/Status/Target column "
       "was found."
   )


   print(
       "Accuracy is not calculated because K-Means "
       "is an unsupervised algorithm."
   )




# ============================================================
# 40. FINAL RESULTS
# ============================================================


print("\n" + "=" * 75)
print("FINAL CLUSTERING RESULTS")
print("=" * 75)




print("\nK-MEANS")
print("-" * 40)


print(
   "Best K:",
   best_k_kmeans
)


print(
   "Inertia:",
   round(
       kmeans_model.inertia_,
       4
   )
)


print(
   "Silhouette:",
   round(
       km_silhouette,
       4
   )
)


print(
   "Calinski-Harabasz:",
   round(
       km_calinski,
       4
   )
)


print(
   "Davies-Bouldin:",
   round(
       km_davies,
       4
   )
)




print("\nK-MEANS++")
print("-" * 40)


print(
   "Best K:",
   best_k_kmeans_pp
)


print(
   "Inertia:",
   round(
       kmeans_pp_model.inertia_,
       4
   )
)


print(
   "Silhouette:",
   round(
       pp_silhouette,
       4
   )
)


print(
   "Calinski-Harabasz:",
   round(
       pp_calinski,
       4
   )
)


print(
   "Davies-Bouldin:",
   round(
       pp_davies,
       4
   )
)




# ============================================================
# 41. OUTPUT LOCATION
# ============================================================


print("\n" + "=" * 75)
print("OUTPUTS SAVED")
print("=" * 75)


print(
   "\nMain folder:"
)


print(
   OUTPUT_FOLDER
)


print(
   "\nK-Means:"
)


print(
   KMEANS_FOLDER
)


print(
   "\nK-Means++:"
)


print(
   KMEANS_PP_FOLDER
)


print(
   "\nElbow:"
)


print(
   ELBOW_FOLDER
)


print(
   "\nSilhouette:"
)


print(
   SILHOUETTE_FOLDER
)


print(
   "\nAccuracy:"
)


print(
   ACCURACY_FOLDER
)




print(
   "\nOriginal/preprocessed input dataset was NOT modified."
)


print(
   "\nPROGRAM COMPLETED SUCCESSFULLY."
)
