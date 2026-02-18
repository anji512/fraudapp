from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("payments.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    step = float(request.form['step'])
    type_val = request.form['type']
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])

    type_map = {
        "PAYMENT": 0,
        "TRANSFER": 1,
        "CASH_OUT": 2,
        "DEBIT": 3,
        "CASH_IN": 4
    }

    features = np.array([[step, type_map[type_val], amount,
                          oldbalanceOrg, newbalanceOrig,
                          oldbalanceDest, newbalanceDest]])

    prediction = model.predict(features)[0]

    result = "Fraudulent Transaction ❌" if prediction == 1 else "Safe Transaction ✅"

    return render_template('submit.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)