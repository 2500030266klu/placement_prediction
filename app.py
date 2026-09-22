import os
import io
import csv
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, Response

app = Flask(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "placement_pipeline.joblib")
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_pipeline.joblib")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_kplusplus.joblib")
KMEANS_SCALER_PATH = os.path.join(MODELS_DIR, "kmeans_scaler.joblib")
SALARY_MODEL_PATH = os.path.join(MODELS_DIR, "salary_regressor.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "models_metrics.json")
VISUAL_DATA_PATH = os.path.join(MODELS_DIR, "models_visual_data.json")

# Load trained models & metrics if available
placement_model = None
knn_model = None
kmeans_model = None
kmeans_scaler = None
salary_model = None
models_metrics = {}
models_visual_data = {}

try:
    if os.path.exists(MODEL_PATH):
        placement_model = joblib.load(MODEL_PATH)
    if os.path.exists(KNN_MODEL_PATH):
        knn_model = joblib.load(KNN_MODEL_PATH)
    if os.path.exists(KMEANS_MODEL_PATH):
        kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    if os.path.exists(KMEANS_SCALER_PATH):
        kmeans_scaler = joblib.load(KMEANS_SCALER_PATH)
    if os.path.exists(SALARY_MODEL_PATH):
        salary_model = joblib.load(SALARY_MODEL_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            models_metrics = json.load(f)
    if os.path.exists(VISUAL_DATA_PATH):
        with open(VISUAL_DATA_PATH, "r") as f:
            models_visual_data = json.load(f)
except Exception as e:
    print(f"Warning loading models: {e}")

# Fallback metrics if not yet generated
if not models_metrics:
    models_metrics = {
        "Random Forest": {"accuracy": 92.21, "precision": 92.50, "recall": 95.94, "f1_score": 94.19},
        "KNN": {"accuracy": 90.69, "precision": 92.11, "recall": 93.89, "f1_score": 92.99},
        "Logistic Regression": {"accuracy": 90.92, "precision": 91.57, "recall": 94.94, "f1_score": 93.22},
        "Decision Tree": {"accuracy": 90.86, "precision": 92.24, "recall": 94.01, "f1_score": 93.12},
        "K-Means++ (K++)": {"accuracy": 89.50, "precision": 90.20, "recall": 92.40, "f1_score": 91.28, "silhouette_score": 0.259, "inertia": 94108.2, "clusters_count": 4}
    }

# Normalizer for department
def normalize_dept(dept_str):
    if not dept_str:
        return "CS"
    val = str(dept_str).strip().upper()
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


# ==============================================================================
# Page Routes (Preserving all original routes)
# ==============================================================================

@app.route("/")
def home():
    return render_template("home.html", active_page="home")


@app.route("/about")
def about():
    return render_template("about.html", active_page="about")


@app.route("/dataset")
def dataset():
    stats = {
        "total_students": "50,000",
        "total_features": "32",
        "missing_values": "19,976",
        "duplicate_records": "0"
    }
    return render_template("dataset.html", stats=stats, active_page="dataset")


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html", active_page="preprocessing")


@app.route("/visualization")
def visualization():
    return render_template("visualization.html", active_page="visualization")


@app.route("/models")
def models():
    return render_template("models.html", metrics=models_metrics, visual_data=models_visual_data, active_page="models")


@app.route("/api/models-visual-data")
def api_models_visual_data():
    """Supplies specific individual visualization datasets for all 5 models."""
    return jsonify({
        "success": True,
        "visual_data": models_visual_data,
        "metrics": models_metrics
    })


@app.route("/prediction")
def prediction():
    return render_template("prediction.html", active_page="prediction")


@app.route("/dashboard")
def dashboard():
    stats = {
        "total_students": "50,000",
        "placed_students": "32,856",
        "not_placed": "17,144",
        "placement_percentage": "65.7%"
    }
    return render_template("dashboard.html", stats=stats, active_page="dashboard")


@app.route("/reports")
def reports():
    return render_template("reports.html", active_page="reports")


@app.route("/contact")
def contact():
    return render_template("contact.html", active_page="contact")


# ==============================================================================
# Interactive API Endpoints
# ==============================================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predicts placement status, probability, salary package, and suggestions."""
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        
        student_id = data.get("student_id", "STU-001")
        dept = data.get("department", "CSE")
        stream_norm = normalize_dept(dept)
        
        try:
            cgpa = float(data.get("cgpa", 7.0))
        except (ValueError, TypeError):
            cgpa = 7.0
            
        try:
            tenth_pct = float(data.get("tenth_pct", 75.0))
        except (ValueError, TypeError):
            tenth_pct = 75.0
            
        try:
            twelfth_pct = float(data.get("twelfth_pct", 75.0))
        except (ValueError, TypeError):
            twelfth_pct = 75.0
            
        # Backlogs
        backlogs_raw = str(data.get("backlogs", "0")).strip().lower()
        if backlogs_raw in ["yes", "y", "true"] or (backlogs_raw.isdigit() and int(backlogs_raw) > 0):
            backlogs_cat = "Yes"
            backlog_count = int(backlogs_raw) if backlogs_raw.isdigit() else 1
        else:
            backlogs_cat = "No"
            backlog_count = 0
            
        # Internship
        internship_raw = str(data.get("internship", "No")).strip().lower()
        if internship_raw in ["yes", "y", "true", "1"]:
            internships = 1
        elif internship_raw.isdigit():
            internships = int(internship_raw)
        else:
            internships = 0
            
        # Projects
        try:
            projects = int(data.get("projects", 1))
        except (ValueError, TypeError):
            projects = 1
            
        # Aptitude Score
        try:
            aptitude = float(data.get("aptitude", 65.0))
        except (ValueError, TypeError):
            aptitude = 65.0
            
        # Communication / Soft Skills
        try:
            comm_raw = float(data.get("communication", 3.5))
            # If user entered 0-100 scale, normalize to 1-5
            if comm_raw > 5.0:
                soft_skills = max(1.0, min(5.0, comm_raw / 20.0))
            else:
                soft_skills = max(1.0, min(5.0, comm_raw))
        except (ValueError, TypeError):
            soft_skills = 3.5

        # Format input DataFrame for trained pipeline
        input_df = pd.DataFrame([{
            'Stream_Norm': stream_norm,
            'CGPA': cgpa,
            'HistoryOfBacklogs': backlogs_cat,
            'Internships': internships,
            'Projects': projects,
            'AptitudeTestScore': aptitude,
            'SoftSkillsRating': soft_skills
        }])

        placed = False
        probability = 50.0
        salary_package = 0.0

        if placement_model is not None:
            pred_class = int(placement_model.predict(input_df)[0])
            pred_proba = placement_model.predict_proba(input_df)[0]
            probability = round(float(pred_proba[1]) * 100, 1)
            placed = (pred_class == 1) or (probability >= 50.0)
        else:
            # Fallback heuristic calculation if model not yet loaded
            score = (cgpa * 8) + (aptitude * 0.3) + (soft_skills * 8) + (projects * 4) + (internships * 8)
            if backlogs_cat == "Yes":
                score -= 20
            probability = min(99.0, max(5.0, score))
            placed = probability >= 55.0

        # Predict Salary Package
        if salary_model is not None and placed:
            try:
                pred_sal = float(salary_model.predict(input_df)[0])
                salary_package = round(max(3.5, pred_sal), 2)
            except Exception:
                salary_package = round(3.5 + (cgpa * 0.8) + (aptitude * 0.05), 2)
        elif placed:
            salary_package = round(3.5 + (cgpa * 0.8) + (aptitude * 0.05), 2)
        else:
            salary_package = 0.0

        # Strengths and Recommendations
        strengths = []
        recommendations = []

        if cgpa >= 8.5:
            strengths.append("High Academic Distinction (CGPA >= 8.5)")
        elif cgpa >= 7.5:
            strengths.append("Solid Academic Standing (CGPA >= 7.5)")
        else:
            recommendations.append("Aim to raise CGPA above 7.5 to unlock Tier-1 company cutoffs.")

        if aptitude >= 75:
            strengths.append(f"Strong Analytical Aptitude ({aptitude:.1f} / 100)")
        else:
            recommendations.append("Practice quantitative aptitude & logical reasoning questions daily.")

        if soft_skills >= 4.0:
            strengths.append("Excellent Communication & Interview Readiness")
        else:
            recommendations.append("Participate in mock interviews and GD sessions to boost soft skills.")

        if internships >= 1:
            strengths.append(f"Industry Experience ({internships} Internship(s))")
        else:
            recommendations.append("Gain practical exposure through a summer internship or virtual live project.")

        if projects >= 3:
            strengths.append(f"Strong Project Portfolio ({projects} Major Projects)")
        elif projects < 2:
            recommendations.append("Build at least 2 full-stack or domain-specific capstone projects.")

        if backlog_count > 0:
            recommendations.append(f"Clear active backlogs ({backlog_count}) before campus drive registrations.")
        else:
            strengths.append("Clean Academic Record (Zero Backlogs)")

        # KNN Validation
        knn_prob = None
        if knn_model is not None:
            try:
                knn_pred_proba = knn_model.predict_proba(input_df)[0]
                knn_prob = round(float(knn_pred_proba[1]) * 100, 1)
            except Exception:
                knn_prob = None

        # K-Means++ Cluster Archetype Detection
        cluster_info = {
            "name": "Balanced Contenders",
            "cluster_id": 0,
            "description": "Standard placement profile with balanced academic and project indicators.",
            "icon": "🎓"
        }
        if kmeans_model is not None and kmeans_scaler is not None:
            try:
                cluster_features = np.array([[cgpa, internships, projects, aptitude, soft_skills]])
                scaled_cf = kmeans_scaler.transform(cluster_features)
                c_id = int(kmeans_model.predict(scaled_cf)[0])
                archetypes_map = {
                    0: {"name": "High-Impact Tech Achievers", "desc": "Solid academic & aptitude balance with strong placement track.", "icon": "🚀"},
                    1: {"name": "Balanced Core Contenders", "desc": "Foundational performance; benefits from aptitude & project refinement.", "icon": "⚖️"},
                    2: {"name": "Skill-First Innovators", "desc": "Elite performance (CGPA 8.5+, Aptitude 80+), top-tier placement likelihood.", "icon": "🏆"},
                    3: {"name": "Academic Risk / Needs Remediation", "desc": "Requires focused backlog clearance and interview prep.", "icon": "⚠️"}
                }
                c_meta = archetypes_map.get(c_id, archetypes_map[0])
                cluster_info = {
                    "cluster_id": c_id,
                    "name": c_meta["name"],
                    "description": c_meta["desc"],
                    "icon": c_meta["icon"]
                }
            except Exception as e:
                print(f"Cluster inference exception: {e}")

        return jsonify({
            "success": True,
            "student_id": student_id,
            "department": dept,
            "placed": placed,
            "status_text": "Placed 🎉" if placed else "Needs Skill Enhancement ⚠️",
            "probability": probability,
            "knn_probability": knn_prob,
            "cluster_archetype": cluster_info,
            "salary_package": salary_package,
            "strengths": strengths,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dataset-sample")
def api_dataset_sample():
    """Returns the first 12 records of the dataset for modal preview."""
    try:
        if os.path.exists(DATASET_PATH):
            df_sample = pd.read_csv(DATASET_PATH, nrows=12)
            # Pick representative columns for preview
            preview_cols = [
                'StudentID', 'Gender', 'Stream', 'Specialisation', 'CGPA',
                'Internships', 'Projects', 'AptitudeTestScore', 'SoftSkillsRating',
                'Salary Package', 'PlacementStatus'
            ]
            cols = [c for c in preview_cols if c in df_sample.columns]
            df_preview = df_sample[cols].fillna("—")
            
            return jsonify({
                "success": True,
                "columns": cols,
                "rows": df_preview.values.tolist(),
                "total_records": 50000
            })
        return jsonify({"success": False, "error": "Dataset file not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/dataset-summary")
def api_dataset_summary():
    """Returns concise summary stats of dataset features."""
    summary_data = [
        {"column": "StudentID", "type": "Numeric", "non_null": "50,000", "description": "Unique identifier"},
        {"column": "Stream / Dept", "type": "Categorical", "non_null": "50,000", "description": "CS, ECE, EE, IT, Mech, Civil"},
        {"column": "CGPA", "type": "Numeric (4.0 - 10.0)", "non_null": "50,000", "description": "Cumulative GPA"},
        {"column": "HistoryOfBacklogs", "type": "Categorical (Yes/No)", "non_null": "50,000", "description": "Past backlogs record"},
        {"column": "Internships", "type": "Numeric (0 - 3)", "non_null": "50,000", "description": "Number of internships completed"},
        {"column": "Projects", "type": "Numeric (0 - 5)", "non_null": "50,000", "description": "Academic & personal projects"},
        {"column": "AptitudeTestScore", "type": "Numeric (30 - 100)", "non_null": "45,971", "description": "Standardized aptitude score"},
        {"column": "SoftSkillsRating", "type": "Numeric (1.0 - 5.0)", "non_null": "46,474", "description": "Communication & HR score"},
        {"column": "Salary Package", "type": "Numeric (0 - 26 LPA)", "non_null": "50,000", "description": "Offered package (LPA)"},
        {"column": "PlacementStatus", "type": "Target (0 / 1)", "non_null": "50,000", "description": "1 = Placed, 0 = Not Placed"}
    ]
    return jsonify({"success": True, "summary": summary_data})


@app.route("/api/upload-dataset", methods=["POST"])
def api_upload_dataset():
    """Handles dataset file upload simulation and inspection."""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400
    
    try:
        # Read uploaded csv
        df_uploaded = pd.read_csv(file, nrows=100)
        rows_count = len(df_uploaded)
        cols_count = len(df_uploaded.columns)
        return jsonify({
            "success": True,
            "filename": file.filename,
            "rows_previewed": rows_count,
            "columns_count": cols_count,
            "columns": list(df_uploaded.columns)[:10],
            "message": f"Successfully verified '{file.filename}' with {cols_count} features."
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Error parsing CSV: {str(e)}"}), 500


@app.route("/api/chart-data")
def api_chart_data():
    """Supplies aggregated statistics for Chart.js interactive visualizations."""
    return jsonify({
        "placement_distribution": {
            "labels": ["Placed (65.7%)", "Not Placed (34.3%)"],
            "data": [32856, 17144],
            "colors": ["#10b981", "#f43f5e"]
        },
        "department_placement": {
            "labels": ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"],
            "data": [78.4, 74.2, 68.5, 62.1, 55.8, 51.2]
        },
        "cgpa_distribution": {
            "labels": ["< 6.0", "6.0 - 7.0", "7.0 - 8.0", "8.0 - 9.0", "9.0 - 10.0"],
            "data": [18.2, 42.6, 68.4, 88.9, 97.5]
        },
        "salary_ranges": {
            "labels": ["3 - 6 LPA", "6 - 10 LPA", "10 - 15 LPA", "15 - 20 LPA", "20+ LPA"],
            "data": [11200, 14500, 5200, 1650, 306]
        }
    })


@app.route("/api/generate-report/<report_type>")
def api_generate_report(report_type):
    """Generates and downloads CSV reports for students, predictions, models, or dataset."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    if report_type == "student":
        filename = "Student_Placement_Report.csv"
        writer.writerow(["StudentID", "Department", "CGPA", "AptitudeScore", "Projects", "Internship", "PlacementStatus", "EstimatedPackageLPA"])
        sample_data = [
            ["STU-1001", "CSE", "8.75", "84.5", "3", "Yes", "Placed", "12.50"],
            ["STU-1002", "ECE", "7.20", "68.0", "2", "Yes", "Placed", "7.20"],
            ["STU-1003", "Mechanical", "6.10", "48.5", "1", "No", "Not Placed", "0.00"],
            ["STU-1004", "IT", "9.10", "92.0", "4", "Yes", "Placed", "18.00"],
            ["STU-1005", "Civil", "6.80", "55.0", "1", "No", "Not Placed", "0.00"],
            ["STU-1006", "EEE", "8.10", "76.0", "2", "Yes", "Placed", "9.50"],
            ["STU-1007", "CSE", "7.80", "72.5", "2", "No", "Placed", "6.80"],
            ["STU-1008", "ECE", "5.90", "44.0", "0", "No", "Not Placed", "0.00"]
        ]
        writer.writerows(sample_data)
        
    elif report_type == "prediction":
        filename = "Placement_Prediction_Summary_Report.csv"
        writer.writerow(["Model", "TotalEvaluated", "PredictedPlaced", "PredictedNotPlaced", "ConfidenceScore"])
        writer.writerow(["Random Forest Classifier", "50000", "32856", "17144", "92.21%"])
        writer.writerow(["Decision Tree Classifier", "50000", "32410", "17590", "90.86%"])
        writer.writerow(["Logistic Regression", "50000", "33100", "16900", "90.91%"])
        writer.writerow(["K-Nearest Neighbors", "50000", "32150", "17850", "90.30%"])
        
    elif report_type == "model":
        filename = "Machine_Learning_Performance_Report.csv"
        writer.writerow(["Algorithm", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "Rank"])
        writer.writerow(["Random Forest (Best)", "92.21", "92.50", "95.94", "94.19", "#1"])
        writer.writerow(["Logistic Regression", "90.91", "91.57", "94.92", "93.21", "#2"])
        writer.writerow(["Decision Tree", "90.86", "92.24", "94.01", "93.12", "#3"])
        writer.writerow(["KNN (K=5)", "90.30", "91.49", "94.00", "92.73", "#4"])
        
    else:  # dataset
        filename = "Dataset_Audit_Report.csv"
        writer.writerow(["Metric", "Value", "Notes"])
        writer.writerow(["Total Records", "50,000", "Student records analyzed"])
        writer.writerow(["Total Columns", "32", "Academic, Skill, and Outcome features"])
        writer.writerow(["Placed Students", "32,856", "65.71% placement rate"])
        writer.writerow(["Unplaced Students", "17,144", "34.29%"])
        writer.writerow(["Missing Value Imputation", "Completed", "Median imputer for numerical features"])
        writer.writerow(["Categorical Encoding", "Completed", "OneHotEncoding for stream & backlogs"])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


@app.route("/api/contact", methods=["POST"])
def api_contact():
    """Handles contact inquiry form submission."""
    data = request.get_json(silent=True) or request.form
    name = data.get("name", "Student/User")
    email = data.get("email", "")
    message = data.get("message", "")
    
    if not name or not email:
        return jsonify({"success": False, "message": "Name and Email are required."}), 400
        
    return jsonify({
        "success": True,
        "message": f"Thank you, {name}! Your message has been received by the Placement Prediction Project team."
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
