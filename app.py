from flask import Flask, render_template


app = Flask(__name__)




@app.route("/")
def home():
   return render_template("home.html")




@app.route("/about")
def about():
   return render_template("about.html")




@app.route("/dataset")
def dataset():
   return render_template("dataset.html")




@app.route("/preprocessing")
def preprocessing():
   return render_template("preprocessing.html")




@app.route("/visualization")
def visualization():
   return render_template("visualization.html")




@app.route("/models")
def models():
   return render_template("models.html")




@app.route("/prediction")
def prediction():
   return render_template("prediction.html")




@app.route("/dashboard")
def dashboard():
   return render_template("dashboard.html")




@app.route("/reports")
def reports():
   return render_template("reports.html")




@app.route("/contact")
def contact():
   return render_template("contact.html")




if __name__ == "__main__":
   app.run(debug=True)
