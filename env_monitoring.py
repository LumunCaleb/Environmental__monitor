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

st.title("Environmental Monitoring Model :monitor:")

st.write("Enter feature values for prediction:")

# Input fields
Week = st.number_input('Week', value=0.0)
Temperature = st.number_input('Temperature', value=0.0)
Humidity = st.number_input('Humidity', value=0.0)
GasLevel = st.number_input('GasLevel', value=0.0)

# Create a DataFrame for input features
input_data = pd.DataFrame([[Week, Temperature, Humidity, GasLevel]], columns=['Week', 'Temp', 'Hum', 'Gas'])

if st.button('Predict'):
    # Transform the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Perform prediction
    prediction = model.predict(input_data_scaled)
    
    # Convert the numeric prediction to the corresponding label
    try:
        prev_status = label_encoder.inverse_transform([prediction[0]])[0]
    except ValueError:
        # Handle the case where the prediction might be out of bounds
        prev_status = 'unknown'
    
    # Display the result
    st.write(f'Prediction: {prev_status}')
    
    # Update the non-editable Previous_Status field
    st.text_input('Previous_Status', value=prev_status, disabled=True)

    # Optionally, display the full input data with predicted status for verification
    st.write("Input Data with Predicted Status:")
    input_data_with_status = pd.DataFrame([[Week, prev_status, Temperature, Humidity, GasLevel]],
                                          columns=['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas'])
    st.write(input_data_with_status)
