from flask import Flask, request, render_template
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained model and other necessary components

model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    precipitation = float(request.form['precipitation'])
    cloud_cover = int(request.form['cloud_cover'])
    atmospheric_pressure = float(request.form['atmospheric_pressure'])
    season = int(request.form['season'])
    visibility = float(request.form['visibility'])
    location = int(request.form['location'])

    # Prepare input for prediction
    user_input = np.array([[temperature, humidity, wind_speed, precipitation,
                            cloud_cover, atmospheric_pressure, season,
                            visibility, location]])

    # Scale the input
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_scaled)

    # Reverse encode the prediction
    weather_type = label_encoder.inverse_transform(prediction)[0]

    # Display the prediction
    return f'Predicted Weather Type: {weather_type}'

if __name__ == '__main__':
    app.run(debug=True)
