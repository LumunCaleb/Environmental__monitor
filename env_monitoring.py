import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Define the paths to your joblib files
model_file_path = 'knnmainnew_model.joblib'
label_encoder_path = 'encoder.joblib'
scaler_path = 'scaler.joblib'

# Load the model, label encoder, and scaler using joblib
model = joblib.load(model_file_path)
label_encoder = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)

st.title("Environmental Monitoring Model:monitor:")

st.write("Enter feature values for prediction:")

# Input fields
Week = st.number_input('Week', value=0.0)
Temperature = st.number_input('Temperature', value=0.0)
Humidity = st.number_input('Humidity', value=0.0)
GasLevel = st.number_input('GasLevel', value=0.0)

# Create a DataFrame for input features
input_data = pd.DataFrame([[Week, Temperature, Humidity, GasLevel]], columns=['Week', 'Temp', 'Hum', 'Gas'])

# Transform the input data for scaling
input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    # Assume the prediction is used to update Previous_Status
    previous_status = prediction[0]
    
    # Transform 'Previous_Status' using the label encoder
    try:
        # Update input_data with the predicted Previous_Status
        input_data_with_status = pd.DataFrame([[Week, previous_status, Temperature, Humidity, GasLevel]],
                                              columns=['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas'])
    except ValueError:
        # Assign a default value if category is unknown
        input_data_with_status = pd.DataFrame([[Week, 'unknown', Temperature, Humidity, GasLevel]],
                                              columns=['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas'])

    st.write(f'Prediction: {previous_status}')
    
    # Display the result in a non-editable format
    st.write(f'Previous Status (Predicted): {previous_status}')
