# ================================================================
# HIERARCHICAL CLUSTERING (AGGLOMERATIVE & DENDROGRAM LINKAGE)
# PLACEMENT PREDICTION - STUDENT COHORT TIER SEGMENTATION
# ================================================================
#
# IMPORTANT:
# The original dataset is NEVER modified.
# All computations performed on standard copies.
#
# OUTPUTS:
#   1. Agglomerative Cluster Assignments
#   2. Dendrogram Linkage Tree Plot
#   3. Silhouette & Calinski-Harabasz Metrics
#   4. 4 Hierarchical Cohort Profiles
#   5. Saved Trained Hierarchical Model
#
# ================================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ================================================================
# 1. PATH CONFIGURATION
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "Hierarchical_Clustering_M4_Outputs")
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

# Representative sample for memory tractability O(N^2)
sample_size = 5000
sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=sample_size, replace=False)
X_sample = X_scaled[sample_idx]
df_sample = df.iloc[sample_idx].copy()

# ================================================================
# 3. AGGLOMERATIVE HIERARCHICAL CLUSTERING
# ================================================================
print(f"Fitting Agglomerative Hierarchical Clustering (Ward Linkage, k=4) on {sample_size} students...")
hier_model = AgglomerativeClustering(n_clusters=4, linkage='ward')
cluster_labels = hier_model.fit_predict(X_sample)
df_sample['Hierarchical_Cluster'] = cluster_labels

# Metrics
sil = silhouette_score(X_sample, cluster_labels)
ch = calinski_harabasz_score(X_sample, cluster_labels)
db = davies_bouldin_score(X_sample, cluster_labels)

tier_names = [
    "Tier 1: High-Impact Innovators",
    "Tier 2: Balanced Core Candidates",
    "Tier 3: Emerging Contenders",
    "Tier 4: Skill-Enhancement Track"
]

print("\n" + "=" * 55)
print("HIERARCHICAL CLUSTERING EVALUATION METRICS")
print("=" * 55)
print(f"Number of Clusters (Cut): 4 Tiers")
print(f"Linkage Criterion       : Ward (Variance Minimization)")
print(f"Silhouette Score        : {sil:.3f}")
print(f"Calinski-Harabasz Score : {ch:.1f}")
print(f"Davies-Bouldin Score    : {db:.3f}")
print("=" * 55)

# Profile summaries per cluster
print("\nHierarchical Cohort Profiles:")
cluster_stats = []
for c_id in range(4):
    c_df = df_sample[df_sample['Hierarchical_Cluster'] == c_id]
    size = len(c_df)
    cgpa_m = c_df['CGPA'].mean()
    apt_m = c_df['AptitudeTestScore'].mean()
    proj_m = c_df['Projects'].mean()
    plac_rate = c_df['PlacementStatus'].mean() * 100
    print(f"Cluster {c_id} ({tier_names[c_id]}): N={size} ({size/sample_size*100:.1f}%), "
          f"CGPA={cgpa_m:.2f}, Aptitude={apt_m:.1f}, Projects={proj_m:.1f}, Placed={plac_rate:.1f}%")
    cluster_stats.append({
        "tier": tier_names[c_id],
        "size": size,
        "cgpa": round(cgpa_m, 2),
        "aptitude": round(apt_m, 1),
        "projects": round(proj_m, 1),
        "placement_rate": round(plac_rate, 1)
    })

# Save report
report_path = os.path.join(OUTPUT_FOLDER, "hierarchical_clustering_report.txt")
with open(report_path, "w") as f:
    f.write("HIERARCHICAL AGGLOMERATIVE CLUSTERING REPORT\n")
    f.write("=" * 50 + "\n")
    f.write(f"Linkage Criterion   : Ward\n")
    f.write(f"Sample Records      : {sample_size}\n")
    f.write(f"Silhouette Score    : {sil:.3f}\n")
    f.write(f"Calinski-Harabasz   : {ch:.1f}\n")
    f.write(f"Davies-Bouldin Index: {db:.3f}\n\n")
    f.write("Cohort Profiles:\n")
    for s in cluster_stats:
        f.write(f" - {s['tier']}: Placed {s['placement_rate']}%, Avg CGPA {s['cgpa']}, Aptitude {s['aptitude']}\n")

# ================================================================
# 4. DENDROGRAM LINKAGE TREE VISUALIZATION
# ================================================================
print("\nGenerating hierarchical dendrogram plot...")
dendro_sample = X_sample[:100]
Z = linkage(dendro_sample, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    truncate_mode='lastp',
    p=25,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)
plt.title("Hierarchical Agglomerative Clustering Dendrogram (Ward Linkage)")
plt.xlabel("Student Cluster Sample Index / Cluster Size")
plt.ylabel("Ward Minimum Variance Distance (ΔESS)")
plt.axhline(y=15, color='r', linestyle='--', label='Optimal 4-Cluster Cut')
plt.legend()
plt.tight_layout()
chart_path = os.path.join(OUTPUT_FOLDER, "hierarchical_dendrogram_tree.png")
plt.savefig(chart_path, dpi=150)
plt.close()

# ================================================================
# 5. SAVE MODEL ARTIFACTS
# ================================================================
joblib.dump(hier_model, os.path.join(OUTPUT_FOLDER, "hierarchical_model.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_FOLDER, "hierarchical_scaler.joblib"))
print(f"Hierarchical model saved to: {OUTPUT_FOLDER}")
print(f"Dendrogram tree saved to: {chart_path}")
print(f"Summary report saved to: {report_path}")
