import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Define the paths to your joblib files
model_file_path = 'knnmainnew_model.joblib'
encoder_file_path = 'encoder.joblib'
scaler_file_path = 'scaler.joblib'

# Load the model, label encoder, and scaler using joblib
try:
    model = joblib.load(model_file_path)
    label_encoder = joblib.load(encoder_file_path)
    scaler = joblib.load(scaler_file_path)
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()

st.title("Environmental Monitoring Model:")

st.write("Enter feature values for prediction:")

# Input fields
Week = st.number_input('Week', value=0.0)
Previous_Status = st.text_input('Previous_Status', 'M')
Temperature = st.number_input('Temperature', value=0.0)
Humidity = st.number_input('Humidity', value=0.0)
GasLevel = st.number_input('GasLevel', value=0.0)

# Create a DataFrame for input features
cols = ['Week', 'Prev. Status', 'Temp', 'Hum', 'Gas']
input_data = pd.DataFrame([[Week, Previous_Status, Temperature, Humidity, GasLevel]], columns=cols)

# Transform 'Prev. Status' using the label encoder
try:
    input_data['Prev_Status'] = label_encoder.transform(input_data[['Prev. Status']])
except ValueError:
    st.error("Unknown category in 'Previous_Status'. Please use a known category.")
    st.stop()

# Select the required features and scale them
input_data = input_data[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]
input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Prediction: {prediction[0]}')
