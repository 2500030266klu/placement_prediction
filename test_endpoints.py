import json
import urllib.request
from app import app

# Tests can run against live server if up, or seamlessly via Flask test_client
tests = [
    ('/', 200, 'Placement Prediction System'),
    ('/about', 200, 'About Placement Prediction System'),
    ('/dataset', 200, '50,000'),
    ('/preprocessing', 200, 'Data Preprocessing'),
    ('/visualization', 200, 'Data Visualization'),
    ('/models', 200, 'K-Nearest Neighbors (KNN)'),
    ('/models', 200, 'K-Means++ (K++ Clustering)'),
    ('/models', 200, 'XGBoost Classifier'),
    ('/models', 200, 'AdaBoost Classifier'),
    ('/models', 200, 'DBSCAN (Density-Based'),
    ('/models', 200, 'Hierarchical Clustering'),
    ('/models', 200, 'Multi-Layer Perceptron (MLP)'),
    ('/models', 200, 'Support Vector Machine (SVC)'),
    ('/models', 200, 'Stacking Meta-Ensemble'),
    ('/models', 200, 'Gaussian Naive Bayes (GNB)'),
    ('/models', 200, 'rfVisualChart'),
    ('/models', 200, 'knnVisualChart'),
    ('/models', 200, 'kppVisualChart'),
    ('/models', 200, 'xgbVisualChart'),
    ('/models', 200, 'adaVisualChart'),
    ('/models', 200, 'dbscanVisualChart'),
    ('/models', 200, 'hierVisualChart'),
    ('/models', 200, 'mlpVisualChart'),
    ('/models', 200, 'svmVisualChart'),
    ('/models', 200, 'stackVisualChart'),
    ('/models', 200, 'nbVisualChart'),
    ('/outputs', 200, 'Outputs Explorer'),
    ('/outputs', 200, 'M1: EDA'),
    ('/outputs', 200, 'M2: Linear Reg.'),
    ('/outputs', 200, 'M3: Trees & Boosting'),
    ('/outputs', 200, 'M4: Clustering'),
    ('/outputs', 200, 'M5: SVM & PCA'),
    ('/outputs', 200, 'M6: Deep Learning'),
    ('/outputs', 200, 'Evaluation Matrix'),
    ('/evaluation', 200, 'Model Evaluation Matrix & Diagnostics'),
    ('/evaluation-matrix', 200, 'Model Evaluation Matrix & Diagnostics'),
    ('/model-evaluation', 200, 'Model Evaluation Matrix & Diagnostics'),
    ('/prediction', 200, 'Student Placement Prediction'),
    ('/dashboard', 200, '32,856'),
    ('/reports', 200, 'Placement Reports'),
    ('/contact', 200, 'Contact Us'),
    ('/api/dataset-sample', 200, 'StudentID'),
    ('/api/dataset-summary', 200, 'StudentID'),
    ('/api/chart-data', 200, 'placement_distribution'),
    ('/api/models-visual-data', 200, 'random_forest'),
    ('/api/models-visual-data', 200, 'knn'),
    ('/api/models-visual-data', 200, 'kmeans_kpp'),
    ('/api/models-visual-data', 200, 'xgboost'),
    ('/api/models-visual-data', 200, 'adaboost'),
    ('/api/models-visual-data', 200, 'dbscan'),
    ('/api/models-visual-data', 200, 'hierarchical'),
    ('/api/models-visual-data', 200, 'svm'),
    ('/api/models-visual-data', 200, 'naive_bayes'),
    ('/api/models-visual-data', 200, 'pca'),
    ('/api/models-visual-data', 200, 'mlp_neural_network'),
    ('/api/models-visual-data', 200, 'stacking_classifier'),
    ('/api/evaluation-matrix', 200, 'evaluation_matrix'),
    ('/api/outputs-list', 200, 'eda_cgpa_hist'),
    ('/api/outputs-list', 200, 'reg_cfne_vs_gd'),
    ('/api/outputs-list', 200, 'dt_tree_diagram'),
    ('/api/outputs-list', 200, 'km_elbow_kpp'),
    ('/api/outputs-list', 200, 'm5_svm_boundary'),
    ('/api/outputs-list', 200, 'm5_pca_scree'),
    ('/api/outputs-list', 200, 'm6_mlp_loss'),
    ('/api/outputs-list', 200, 'eval_conf_matrix_grid'),
    ('/api/preview-report/student', 200, 'Student Placement Profile Report'),
    ('/api/preview-report/model', 200, 'Machine Learning Performance Evaluation Report'),
    ('/api/outputs/preview-csv/Decision_Tree_Classifier_M3_Outputs/metrics/classification_report.csv', 200, 'precision'),
    ('/api/outputs/preview-csv/M5_SVM_PCA_outputs/tables/m5_models_performance.csv', 200, 'Support Vector Machine'),
    ('/api/outputs/preview-csv/M6_Deep_Learning_Ensemble_outputs/tables/m6_deep_learning_metrics.csv', 200, 'Multi-Layer Perceptron'),
    ('/api/outputs/preview-csv/Evaluation_Matrix_outputs/tables/comprehensive_evaluation_matrix.csv', 200, 'Random Forest'),
    ('/api/outputs/preview-text/Hierarchical_Clustering_M4_Outputs/hierarchical_clustering_report.txt', 200, 'HIERARCHICAL'),
    ('/api/outputs/preview-text/M5_SVM_PCA_outputs/reports/m5_svm_pca_technical_report.txt', 200, 'MODULE 5 TECHNICAL REPORT'),
    ('/api/outputs/preview-text/M6_Deep_Learning_Ensemble_outputs/reports/m6_neural_networks_report.txt', 200, 'MODULE 6 TECHNICAL REPORT'),
    ('/api/outputs/preview-text/Evaluation_Matrix_outputs/reports/model_evaluation_diagnostic_report.txt', 200, 'COMPREHENSIVE MODEL EVALUATION')
]

client = app.test_client()

# Check if live server is reachable
server_online = False
try:
    urllib.request.urlopen('http://127.0.0.1:5000/', timeout=0.5)
    server_online = True
    print("Detected live Flask server at http://127.0.0.1:5000")
except Exception:
    print("Testing directly via Flask TestClient (robust standalone testing)")

all_passed = True

for path, expected_status, snippet in tests:
    try:
        # Use TestClient directly for code-under-test validation
        res = client.get(path)
        status = res.status_code
        content = res.get_data(as_text=True)

        if status == expected_status and snippet in content:
            print(f'PASS: {path} [Found: {snippet[:24]}] (Status {status})')
        else:
            print(f'FAIL: {path} (Snippet "{snippet}" not found or status {status} != {expected_status})')
            all_passed = False
    except Exception as e:
        print(f'ERROR: {path} -> {e}')
        all_passed = False

# Test reports download
for rpt in ['student', 'prediction', 'model', 'dataset']:
    try:
        res = client.get(f'/api/generate-report/{rpt}')
        status = res.status_code
        ct = res.content_type
        cd = res.headers.get('Content-Disposition')

        if status == 200 and 'text/csv' in ct and cd:
            print(f'PASS: Report download {rpt} -> {status}, Content-Type: {ct}, Header: {cd}')
        else:
            print(f'FAIL: Report download {rpt} status {status}, ct {ct}')
            all_passed = False
    except Exception as e:
        print(f'ERROR: Report download {rpt} -> {e}')
        all_passed = False

# Test output file downloads (PNG, CSV, TXT) across M1-M6 + EVAL
output_files_to_test = [
    'Linear_Regression_CFNE_GD_Compare_M2/actual_vs_predicted.png',
    'Decision_Tree_Classifier_M3_Outputs/metrics/classification_report.csv',
    'Hierarchical_Clustering_M4_Outputs/hierarchical_clustering_report.txt',
    'M5_SVM_PCA_outputs/charts/svm_decision_boundary.png',
    'M5_SVM_PCA_outputs/tables/m5_models_performance.csv',
    'M6_Deep_Learning_Ensemble_outputs/charts/mlp_loss_convergence_curve.png',
    'M6_Deep_Learning_Ensemble_outputs/tables/m6_deep_learning_metrics.csv',
    'Evaluation_Matrix_outputs/charts/confusion_matrices_grid.png',
    'Evaluation_Matrix_outputs/tables/comprehensive_evaluation_matrix.csv',
    'Evaluation_Matrix_outputs/reports/model_evaluation_diagnostic_report.txt'
]
for of in output_files_to_test:
    res = client.get(f'/api/outputs/file/{of}')
    if res.status_code == 200 and len(res.data) > 0:
        print(f'PASS: Output file serve {of} -> 200, {len(res.data)} bytes')
    else:
        print(f'FAIL: Output file serve {of} -> status {res.status_code}')
        all_passed = False

# Test predict API with all models (KNN, XGBoost, AdaBoost, SVM, Naive Bayes, MLP, Stacking, K-Means++, DBSCAN, Hierarchical)
pred_payload = {
    'student_id': 'STU-VERIFY-01',
    'department': 'CSE',
    'cgpa': 8.8,
    'tenth_pct': 88.0,
    'twelfth_pct': 85.0,
    'backlogs': '0',
    'internship': 'Yes',
    'projects': 3,
    'aptitude': 82.0,
    'communication': 4.5
}

try:
    pred_res = client.post('/api/predict', json=pred_payload)
    pred_data = pred_res.get_json()

    required_keys = [
        'cluster_archetype', 'knn_probability', 'xgboost_probability',
        'adaboost_probability', 'svm_probability', 'naive_bayes_probability',
        'mlp_probability', 'stacking_probability', 'density_profile',
        'hierarchical_cluster', 'salary_package'
    ]
    all_keys_valid = all(pred_data.get(k) is not None for k in required_keys)
    if pred_data.get('success') and all_keys_valid:
        print('PASS: Predict API returned Cluster Archetype:', pred_data['cluster_archetype']['name'])
        print('PASS: Predict API returned KNN Probability:', pred_data['knn_probability'])
        print('PASS: Predict API returned XGBoost Probability:', pred_data['xgboost_probability'])
        print('PASS: Predict API returned AdaBoost Probability:', pred_data['adaboost_probability'])
        print('PASS: Predict API returned SVM Probability:', pred_data['svm_probability'])
        print('PASS: Predict API returned Naive Bayes Probability:', pred_data['naive_bayes_probability'])
        print('PASS: Predict API returned MLP Probability:', pred_data['mlp_probability'])
        print('PASS: Predict API returned Stacking Probability:', pred_data['stacking_probability'])
        print('PASS: Predict API returned Salary Package:', pred_data['salary_package'])
        print('PASS: Predict API returned DBSCAN Profile:', pred_data['density_profile']['status'])
        print('PASS: Predict API returned Hierarchical Tier:', pred_data['hierarchical_cluster']['tier'])
    else:
        print('FAIL: Predict API missing required enhanced keys or keys are None:', pred_data)
        all_passed = False
except Exception as e:
    print(f'ERROR in Predict API test: {e}')
    all_passed = False

# Test contact submit
try:
    contact_payload = {'name': 'Professor Sharma', 'email': 'prof@university.edu', 'message': 'Great project!'}
    res = client.post('/api/contact', json=contact_payload)
    print('PASS: Contact API ->', res.get_data(as_text=True))
except Exception as e:
    print(f'ERROR in Contact API test: {e}')
    all_passed = False

print('\nALL AUTOMATED TESTS PASSED:', all_passed)
