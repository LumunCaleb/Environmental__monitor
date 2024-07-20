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

# Initialize Streamlit session state variables
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = 'M'  # Default value

st.title("Environmental Monitoring Model:monitor:")

st.write("Enter feature values for prediction:")

# Input fields
Week = st.number_input('Week', value=0.0)
Temperature = st.number_input('Temperature', value=0.0)
Humidity = st.number_input('Humidity', value=0.0)
GasLevel = st.number_input('GasLevel', value=0.0)

# Automatically update Previous_Status with the result of a prediction
# Create a DataFrame for input features
cols = ['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']
input_data = pd.DataFrame([[Week, st.session_state.previous_prediction, Temperature, Humidity, GasLevel]], columns=cols)

# Transform 'Prev_Status' using the label encoder
try:
    input_data['Prev_Status'] = label_encoder.transform(input_data[['Prev_Status']])
except ValueError:
    # Assign a default value if category is unknown
    input_data['Prev_Status'] = label_encoder.transform([['M']])[0]

# Ensure the order of columns matches the trained scaler's expectations
input_data = input_data[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]

# Scale the features
input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    # Perform prediction
    prediction = model.predict(input_data_scaled)
    
    # Update session state with the latest prediction
    st.session_state.previous_prediction = prediction[0]
    
    # Display the result
    st.write(f'Prediction: {prediction[0]}')

    # Update 'Previous_Status' with the prediction result
    st.write(f'Updated Previous Status: {st.session_state.previous_prediction}')
