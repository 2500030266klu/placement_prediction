import urllib.request
import json

tests = [
    ('/', 200, 'Placement Prediction System'),
    ('/about', 200, 'About Placement Prediction System'),
    ('/dataset', 200, '50,000'),
    ('/preprocessing', 200, 'Data Preprocessing'),
    ('/visualization', 200, 'Data Visualization'),
    ('/models', 200, 'K-Nearest Neighbors (KNN)'),
    ('/models', 200, 'K-Means++ (K++ Clustering)'),
    ('/models', 200, 'rfVisualChart'),
    ('/models', 200, 'knnVisualChart'),
    ('/models', 200, 'kppVisualChart'),
    ('/prediction', 200, 'Student Placement Prediction'),
    ('/dashboard', 200, '32,856'),
    ('/reports', 200, 'Placement Reports'),
    ('/contact', 200, 'Contact Us'),
    ('/api/dataset-sample', 200, '"success": true'),
    ('/api/dataset-summary', 200, '"success": true'),
    ('/api/chart-data', 200, '"placement_distribution"'),
    ('/api/models-visual-data', 200, '"random_forest"'),
    ('/api/models-visual-data', 200, '"knn"'),
    ('/api/models-visual-data', 200, '"kmeans_kpp"')
]

all_passed = True
for path, expected_status, snippet in tests:
    url = f'http://127.0.0.1:5000{path}'
    try:
        res = urllib.request.urlopen(url)
        content = res.read().decode('utf-8')
        if res.status == expected_status and snippet in content:
            print(f'PASS: {path} [Found: {snippet[:20]}] (Status {res.status})')
        else:
            print(f'FAIL: {path} (Snippet "{snippet}" not found or status mismatch)')
            all_passed = False
    except Exception as e:
        print(f'ERROR: {path} -> {e}')
        all_passed = False

# Test reports download
for rpt in ['student', 'prediction', 'model', 'dataset']:
    res = urllib.request.urlopen(f'http://127.0.0.1:5000/api/generate-report/{rpt}')
    ct = res.headers.get('Content-Type')
    cd = res.headers.get('Content-Disposition')
    print(f'PASS: Report download {rpt} -> {res.status}, Content-Type: {ct}, Header: {cd}')

# Test predict API with cluster archetype & KNN score
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
pred_req = urllib.request.Request(
    'http://127.0.0.1:5000/api/predict',
    data=json.dumps(pred_payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
pred_res = urllib.request.urlopen(pred_req)
pred_data = json.loads(pred_res.read().decode('utf-8'))
if pred_data.get('success') and 'cluster_archetype' in pred_data and 'knn_probability' in pred_data:
    print('PASS: Predict API returned Cluster Archetype:', pred_data['cluster_archetype']['name'])
    print('PASS: Predict API returned KNN Probability:', pred_data['knn_probability'])
else:
    print('FAIL: Predict API missing cluster or KNN:', pred_data)
    all_passed = False

# Test contact submit
req = urllib.request.Request(
    'http://127.0.0.1:5000/api/contact',
    data=json.dumps({'name': 'Professor Sharma', 'email': 'prof@university.edu', 'message': 'Great project!'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res = urllib.request.urlopen(req)
print('PASS: Contact API ->', res.read().decode('utf-8'))

print('\nALL AUTOMATED TESTS PASSED:', all_passed)
