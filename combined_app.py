from flask import Flask, request, render_template
from pycaret.regression import load_model as load_regression_model, predict_model as predict_regression_model
from pycaret.classification import load_model as load_classification_model, predict_model as predict_classification_model
import pandas as pd

app = Flask(__name__)

# Load the regression model and define columns
regression_model = load_regression_model('resale-pipeline')
regression_cols = ['flat_type', 'storey_range', 'floor_area_sqm', 'cbd_dist', 'min_dist_mrt', 'remaining_years_lease']

# Load the classification model and define columns
classification_model = load_classification_model('cardiovascular_health_model')
classification_cols = ['Age', 'Gender', 'Chest Pain Type', 'Resting Blood Pressure (mm Hg)', 'Cholesterol (mg/dL)',
                       'Fasting Blood Sugar', 'Resting ECG Results', 'Maximum Heart Rate', 'Exercise-Induced Angina']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/home2', methods=['GET', 'POST'])
def home2():
    return render_template("home2.html")

@app.route('/predict_resale', methods=['POST'])
def predict_resale():
    try:
        int_features = [request.form.get(col) for col in regression_cols]
        final = pd.DataFrame([int_features], columns=regression_cols)
        prediction = predict_regression_model(regression_model, data=final, round=0)
        predicted_price = prediction.iloc[0]['prediction_label']
        return render_template('home.html', pred='Expected Resale Price will be ${}'.format(predicted_price))
    except Exception as e:
        return str(e)

@app.route('/predict_health', methods=['POST'])
def predict_health():
    try:
        int_features = [request.form.get(col) for col in classification_cols]
        final = pd.DataFrame([int_features], columns=classification_cols)
        prediction = predict_classification_model(classification_model, data=final)
        predicted_result = "Positive" if prediction.Label[0] == 1 else "Negative"
        return render_template('home2.html', prediction=predicted_result)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)