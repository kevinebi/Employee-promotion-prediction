from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('hr.pkl', 'rb'))


@app.route('/')
def loadpage():
    return render_template("home.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict')
def pred():
    return render_template("predict.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        department = int(request.form['department'])
        education = int(request.form['education'])
        no_of_trainings = int(request.form['no_of_trainings'])
        age = int(request.form['age'])
        previous_year_rating = float(request.form['previous_year_rating'])
        length_of_service = float(request.form['length_of_service'])
        kpis_met = int(request.form['kpis_met'])
        awards_won = int(request.form['awards_won'])
        avg_training_score = float(request.form['avg_training_score'])

        # Create feature array for prediction
        features = np.array([[department, education, no_of_trainings, age,
                              previous_year_rating, length_of_service, kpis_met,
                              awards_won, avg_training_score]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Determine the message based on prediction
        if prediction == 1:
            prediction = "Great! You are promoted."
        else:
            prediction = "Sorry, you are not promoted."

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
