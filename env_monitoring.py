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

# Custom CSS to style the page
st.markdown("""
    <style>
        .reportview-container {
            background-color: #d0f0c0; /* Light green background */
        }
        .sidebar .sidebar-content {
            background-color: #d0f0c0; /* Light green sidebar */
        }
        h1 {
            color: #006400; /* Dark green heading */
            font-size: 2.5em; /* Increase font size */
        }
        .stButton>button {
            background-color: #006400; /* Dark green button */
            color: white; /* White text */
            font-size: 1.2em; /* Increase font size */
            padding: 10px 20px; /* Add padding */
            border-radius: 5px; /* Rounded corners */
        }
    </style>
""", unsafe_allow_html=True)

st.title("Environmental Monitoring Model :monitor:")

st.write("Enter feature values for prediction:")

# Create a dropdown for 'Prev_Status' and make it non-editable
status_options = label_encoder.classes_.tolist()
Previous_Status = st.selectbox('Previous_Status', options=status_options)

# Input fields
Week = st.number_input('Week', value=0.0)
Temperature = st.number_input('Temperature', value=0.0)
Humidity = st.number_input('Humidity', value=0.0)
GasLevel = st.number_input('GasLevel', value=0.0)

# Create a DataFrame for input features
cols = ['Prev. Status', 'Week', 'Temp', 'Hum', 'Gas']
input_data = pd.DataFrame([[Previous_Status, Week, Temperature, Humidity, GasLevel]], columns=cols)

# Transform 'Prev. Status' using the label encoder
input_data['Prev_Status'] = label_encoder.transform(input_data[['Prev. Status']])

# Select the required features and scale them
input_data = input_data[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]
input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    predicted_status = label_encoder.inverse_transform([prediction[0]])[0]
    st.write(f'Prediction: {predicted_status}')

