import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Define the paths to your joblib files
model_file_path = 'knnmainnew_model.joblib'
encoder_file_path = 'encoder.joblib'
scaler_file_path = 'scaler.joblib'

# Load the model, label encoder, and scaler using joblib
model = joblib.load(joblib_file_path)
label_encoder = joblib.load(encoder_file_path)
scaler = joblib.load(scaler_file_path)

st.title("Environmental Monitoring Model:monitor:")

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
    # Assign a default value if category is unknown
    input_data['Prev_Status'] = label_encoder.transform([['unknown']])[0]

# Select the required features and scale them
input_data = input_data[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]
input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Prediction: {prediction[0]}')




# def predict(): 
#     row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
#     X = pd.DataFrame([row], columns = columns)
#     prediction = model.predict(X)
#     # if prediction[0] == 1: 
#     #     st.success('Passenger Survived :thumbsup:')
#     # else: 
#     #     st.error('Passenger did not Survive :thumbsdown:') 

#trigger = st.button('Predict', on_click=predict)
