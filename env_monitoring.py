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

# Define the available options for 'Prev_Status'
prev_status_options = ['M', 'U', 'S']

# Set up the Streamlit page configuration
st.set_page_config(page_title="Environmental Monitoring", page_icon=":bar_chart:", layout="wide")

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #e0f2f1; /* Light green background */
    }
    .css-1d391kg {
        color: #004d40; /* Thick green heading */
    }
    .css-1n2t48i {
        background-color: #004d40; /* Thick green button */
        color: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Environmental Monitoring Model :monitor:")

# Display the 'Prev_Status' field, which will be auto-updated
prev_status = st.selectbox('Previous Status', options=prev_status_options, index=0, disabled=True)

# Input fields
Week = st.number_input('Week', value=0.0)
Temperature = st.number_input('Temperature', value=0.0)
Humidity = st.number_input('Humidity', value=0.0)
GasLevel = st.number_input('GasLevel', value=0.0)

# Create a DataFrame for input features
cols = ['Prev_Status', 'Week', 'Temp', 'Hum', 'Gas']
input_data = pd.DataFrame([[prev_status, Week, Temperature, Humidity, GasLevel]], columns=cols)

# Transform 'Prev_Status' using the label encoder
input_data['Prev_Status'] = label_encoder.transform(input_data[['Prev_Status']])

# Select the required features and scale them
input_data = input_data[['Prev_Status', 'Week', 'Temp', 'Hum', 'Gas']]
input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    predicted_status = label_encoder.inverse_transform([prediction[0]])[0]
    st.write(f'Prediction: {predicted_status}')
    # Update the 'Prev_Status' field based on prediction
    st.selectbox('Previous Status', options=prev_status_options, index=prev_status_options.index(predicted_status), disabled=True)

