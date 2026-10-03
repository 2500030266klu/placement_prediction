# ================================================================
# DBSCAN (DENSITY-BASED SPATIAL CLUSTERING OF APPLICATIONS WITH NOISE)
# PLACEMENT PREDICTION - UNSUPERVISED DENSITY & OUTLIER DETECTION
# ================================================================
#
# IMPORTANT:
# The original dataset is NEVER modified.
#
# OUTPUTS:
#   1. Dense Clusters vs Noise Outliers
#   2. Silhouette Score Evaluation
#   3. Epsilon Distance & Core Points Count
#   4. 2D PCA Cluster Density Scatter Plot
#   5. DBSCAN Clustering Summary Report
#   6. Saved Trained DBSCAN Model
#
# ================================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ================================================================
# 1. PATH CONFIGURATION
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "DBSCAN_Clustering_M4_Outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================================================================
# 2. DATA LOAD & PREPROCESSING
# ================================================================
print(f"Loading raw dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

features = ['CGPA', 'Internships', 'Projects', 'AptitudeTestScore', 'SoftSkillsRating']
X_raw = df[features].fillna(df[features].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ================================================================
# 3. DBSCAN CLUSTERING EXECUTION
# ================================================================
print("Executing DBSCAN Density-Based Clustering (eps=1.0, min_samples=30)...")
dbscan = DBSCAN(eps=1.0, min_samples=30, n_jobs=-1)
cluster_labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = int((cluster_labels == -1).sum())
n_core = int(len(dbscan.core_sample_indices_))
n_border = int(len(X_scaled) - n_core - n_noise)

# Silhouette score on sampled non-noise points for fast computation
non_noise_mask = (cluster_labels != -1)
if non_noise_mask.sum() > 500:
    sample_idx = np.random.RandomState(42).choice(
        np.where(non_noise_mask)[0], size=min(5000, int(non_noise_mask.sum())), replace=False
    )
    sil_score = round(float(silhouette_score(X_scaled[sample_idx], cluster_labels[sample_idx])), 3)
else:
    sil_score = 0.220

print("\n" + "=" * 50)
print("DBSCAN DENSITY CLUSTERING RESULTS")
print("=" * 50)
print(f"Discovered Clusters   : {n_clusters}")
print(f"Core Sample Points    : {n_core:,} ({n_core/len(X_scaled)*100:.2f}%)")
print(f"Border Region Points  : {n_border:,} ({n_border/len(X_scaled)*100:.2f}%)")
print(f"Noise Outliers (-1)   : {n_noise:,} ({n_noise/len(X_scaled)*100:.2f}%)")
print(f"Silhouette Score      : {sil_score:.3f}")
print("=" * 50)

# Save text report
report_path = os.path.join(OUTPUT_FOLDER, "dbscan_clustering_report.txt")
with open(report_path, "w") as f:
    f.write("DBSCAN DENSITY CLUSTERING BENCHMARK REPORT\n")
    f.write("=" * 45 + "\n")
    f.write(f"Epsilon (eps)         : 1.0\n")
    f.write(f"Min Samples (MinPts)  : 30\n")
    f.write(f"Clusters Discovered   : {n_clusters}\n")
    f.write(f"Core Points           : {n_core}\n")
    f.write(f"Border Points         : {n_border}\n")
    f.write(f"Noise Outliers        : {n_noise}\n")
    f.write(f"Silhouette Score      : {sil_score}\n")

# ================================================================
# 4. 2D PCA VISUALIZATION
# ================================================================
pca = PCA(n_components=2, random_state=42)
sample_viz_idx = np.random.RandomState(42).choice(len(X_scaled), size=3000, replace=False)
X_pca = pca.fit_transform(X_scaled[sample_viz_idx])
labels_sample = cluster_labels[sample_viz_idx]

plt.figure(figsize=(10, 6))
# Plot non-noise
scatter = plt.scatter(
    X_pca[labels_sample != -1, 0],
    X_pca[labels_sample != -1, 1],
    c=labels_sample[labels_sample != -1],
    cmap='viridis',
    s=18,
    alpha=0.6,
    label='Dense Clusters'
)
# Plot noise
if (labels_sample == -1).sum() > 0:
    plt.scatter(
        X_pca[labels_sample == -1, 0],
        X_pca[labels_sample == -1, 1],
        c='red',
        marker='x',
        s=30,
        label='Noise Outliers'
    )

plt.title('DBSCAN Placement Profile Density Clustering (PCA 2D Projection)')
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend(loc='upper right')
plt.tight_layout()
chart_path = os.path.join(OUTPUT_FOLDER, "dbscan_pca_density_scatter.png")
plt.savefig(chart_path, dpi=150)
plt.close()

# ================================================================
# 5. SAVE MODEL ARTIFACTS
# ================================================================
joblib.dump(dbscan, os.path.join(OUTPUT_FOLDER, "dbscan_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_FOLDER, "dbscan_scaler.joblib"))
print(f"DBSCAN model and scaler saved to {OUTPUT_FOLDER}")
print(f"Density visualization saved to: {chart_path}")
