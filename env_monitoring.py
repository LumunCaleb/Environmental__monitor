import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Define the paths to your joblib files
model_file_path = 'knnmainnew_model.joblib'
encoder_file_path = 'encoder.joblib'
scaler_file_path = 'scaler.joblib'

# Load the model, label encoder, and scaler using joblib
model = joblib.load(model_file_path)
label_encoder = joblib.load(encoder_file_path)
scaler = joblib.load(scaler_file_path)

st.title("Environmental Monitoring Model :monitor:")

# Input fields
Week = st.number_input('Week', value=0.0)
Temperature = st.number_input('Temperature', value=0.0)
Humidity = st.number_input('Humidity', value=0.0)
GasLevel = st.number_input('GasLevel', value=0.0)

# Create a DataFrame for input features
input_data = pd.DataFrame([[Week, np.nan, Temperature, Humidity, GasLevel]],
                          columns=['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas'])

# Placeholder for displaying the result
prev_status_display = st.text('Previous Status: Not Available')

if st.button('Predict'):
    # Prepare data for prediction
    input_data_for_scaling = input_data[['Week', 'Temp', 'Hum', 'Gas']]
    input_data_scaled = scaler.transform(input_data_for_scaling)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Convert prediction to corresponding 'Prev_Status' value
    predicted_status = label_encoder.inverse_transform([prediction[0]])[0]
    
    # Update DataFrame and display result
    input_data['Prev_Status'] = predicted_status
    prev_status_display.text(f'Previous Status: {predicted_status}')
    
    st.write(f'Prediction: {predicted_status}')

# Display input data including 'Prev_Status'
st.write("Input Data:", input_data)
