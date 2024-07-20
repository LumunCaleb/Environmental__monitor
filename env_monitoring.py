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

# Streamlit page configuration
st.set_page_config(page_title="Environmental Monitoring", page_icon=":bar_chart:", layout="wide")
st.title("Environmental Monitoring Model :monitor:")

# Input fields
Week = st.number_input('Week', value=0.0)
Previous_Status = st.selectbox('Previous_Status', options=['M', 'U', 'S'])
Temperature = st.number_input('Temperature', value=0.0)
Humidity = st.number_input('Humidity', value=0.0)
GasLevel = st.number_input('GasLevel', value=0.0)

# Create a DataFrame for input features with the correct order
input_data = pd.DataFrame([[Week, Previous_Status, Temperature, Humidity, GasLevel]], 
                          columns=['Week', 'Prev_status', 'Temperature', 'Humidity', 'Gas'])

# Transform 'Prev_status' using the label encoder
try:
    input_data['Prev_status'] = label_encoder.transform(input_data['Prev_status'])
except ValueError:
    # Assign a default value if category is unknown
    input_data['Prev_status'] = label_encoder.transform(['M'])[0]  # Default value

# Select the required features and scale them
input_data = input_data[['Week', 'Prev_status', 'Temperature', 'Humidity', 'Gas']]
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button('Predict', key='predict_button'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Prediction: {prediction[0]}')

# Style the page using CSS
st.markdown("""
    <style>
    .stButton button {
        background-color: #28a745; /* Green color for the button */
        color: white;
        font-size: 16px;
        font-weight: bold;
    }
    .stApp {
        background-color: #e6f9e6; /* Light green background for the page */
    }
    .stTitle {
        color: #28a745; /* Thick green color for the heading */
    }
    </style>
    """, unsafe_allow_html=True)

