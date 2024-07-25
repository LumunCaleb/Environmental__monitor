import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Define the paths to your joblib files
joblib_file_path = 'knnmainnew_model.joblib'
label_encoder_path = 'encoder.joblib'
scaler_path = 'scaler.joblib'

# Load the model, label encoder, and scaler using joblib
model = joblib.load(joblib_file_path)
label_encoder = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)

st.title("Environmental Monitoring Model :monitor:")

# Initialize session state variables to track previous prediction
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = 'M'  # Default value or set to 'unknown'
if 'previous_predictions' not in st.session_state:
    st.session_state.previous_predictions = pd.DataFrame(columns=['Week', 'Temperature', 'Humidity', 'GasLevel', 'Predicted Status'])

# Sidebar for navigation
option = st.sidebar.selectbox(
    "Choose an option",
    ("Predict with User Input", "Update a CSV File")
)

if option == "Predict with User Input":
    st.write("Enter feature values for prediction:")

    # Input fields
    Week = st.number_input('Week', value=0, step=1)  # Use step=1 for integer input
    Previous_Status = st.text_input('Previous_Status', st.session_state.previous_prediction)
    Temperature = st.number_input('Temperature', value=0.0)
    Humidity = st.number_input('Humidity', value=0.0)
    GasLevel = st.number_input('GasLevel', value=0.0)

    # Create a DataFrame for input features
    cols = ['Week', 'Prev. Status', 'Temp', 'Hum', 'Gas']
    input_data = pd.DataFrame([[Week, Previous_Status, Temperature, Humidity, GasLevel]], columns=cols)

    # Print the DataFrame to debug
    st.write("Input DataFrame for Prediction:")
    st.write(input_data)

    # Transform 'Prev. Status' using the label encoder
    try:
        # Ensure the 'Prev. Status' column is treated as a 2D array with one feature
        prev_status_encoded = label_encoder.transform(input_data[['Prev. Status']].values.reshape(-1, 1))
        input_data['Prev_Status'] = prev_status_encoded
    except ValueError as e:
        # Handle categories not in encoder
        st.error(f"Encoding error: {e}")
        input_data['Prev_Status'] = label_encoder.transform([['unknown']])[0]

    # Select the required features and scale them
    input_data = input_data[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]
    input_data_scaled = scaler.transform(input_data)

    if st.button('Predict'):
        prediction = model.predict(input_data_scaled)[0]
        st.write(f'Prediction: {prediction}')

        # Update session state with the latest prediction
        st.session_state.previous_prediction = prediction

        # Update previous predictions DataFrame
        new_prediction = pd.DataFrame({
            'Week': [Week],
            'Temperature': [Temperature],
            'Humidity': [Humidity],
            'GasLevel': [GasLevel],
            'Predicted Status': [prediction]
        })
        
        st.session_state.previous_predictions = pd.concat([st.session_state.previous_predictions, new_prediction], ignore_index=True)

        # Display previous predictions and the legend side by side
        col1, col2 = st.columns([4, 2])

        with col1:
            st.write("**Previous Predictions**")
            st.dataframe(st.session_state.previous_predictions)

        with col2:
            st.write("**Prediction Legend**")
            st.write("**U**: Unsafe", "**M**: Moderately Safe", "**S**: Safe")

elif option == "Update a CSV File":
    st.write("Upload a CSV file to update:")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(uploaded_file)

            # Display the first few rows of the DataFrame to confirm it's loaded correctly
            st.write("CSV file loaded successfully. Here are the first few rows:")
            st.write(df.head())

            # Add a button to trigger the prediction
            if st.button('Predict from CSV'):
                # Check if 'Prev. Status' column exists
                if 'Prev. Status' not in df.columns:
                    st.warning("'Prev. Status' column is missing. It will be created.")
                    df['Prev. Status'] = 'M'  # Initialize with a default value

                # Handle NaN values in 'Prev. Status'
                df['Prev. Status'].fillna('M', inplace=True)

                # Define the expected columns and their types
                expected_columns = {
                    'Week': 'int',
                    'Prev. Status': 'object',
                    'Temp': 'float',
                    'Hum': 'float',
                    'Gas': 'float'
                }

                # Validate columns
                missing_columns = [col for col in expected_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing columns: {', '.join(missing_columns)}")
                else:
                    # Validate and correct data types
                    for col, dtype in expected_columns.items():
                        if col in df.columns:
                            if dtype == 'float':
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                            elif dtype == 'int':
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # 'Int64' to allow for missing values
                            elif dtype == 'object':
                                df[col] = df[col].astype(str)

                    # Check for any remaining issues
                    if df.isnull().values.any():
                        st.error("Data contains null values. Please clean the data.")
                        st.write(df[df.isnull().any(axis=1)])
                    else:
                        # Ensure 'Prev. Status' is correctly encoded
                        prev_status_unique = df['Prev. Status'].unique()
                        categories = label_encoder.categories_[0]
                        missing_categories = [x for x in prev_status_unique if x not in categories]
                        
                        if missing_categories:
                            st.warning(f"Some categories in 'Prev. Status' are not in the encoder: {missing_categories}")
                            # Replace missing categories with 'unknown'
                            df['Prev. Status'] = df['Prev. Status'].apply(lambda x: 'unknown' if x not in categories else x)

                        # Transform 'Prev. Status' using the label encoder
                        try:
                            df['Prev_Status'] = label_encoder.transform(df[['Prev. Status']].values.reshape(-1, 1))
                        except ValueError as e:
                            st.error(f"Encoding error: {e}")
                            df['Prev_Status'] = label_encoder.transform([['unknown']])[0]

                        # Process DataFrame
                        features = df[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]
                        features_scaled = scaler.transform(features)
                        df['Prediction'] = model.predict(features_scaled)

                        # Update 'Prev. Status' with the latest prediction
                        prev_prediction = 'U'
                        for i in range(len(df)):
                            df.at[i, 'Prev. Status'] = prev_prediction
                            prev_prediction = df.at[i, 'Prediction']

                        # Map prediction results to colors
                        color_map = {'S': 'green', 'M': 'orange', 'U': 'red'}
                        df['Color'] = df['Prediction'].map(color_map)
                        # Plot results
                        fig, ax = plt.subplots()
                        for label, color in color_map.items():
                            subset = df[df['Prediction'] == label]
                            ax.scatter(subset.index, [label] * len(subset), color=color, label=label)

                        ax.set_xlabel('Index')
                        ax.set_ylabel('Prediction')
                        ax.set_title('Prediction Results')
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        st.pyplot(fig)

                        # Save the CSV file locally (optional)
                        #save_path = r'C:\Users\CALEB\uploaded_file.csv'
                        #3df.to_csv(save_path, index=False)
                        #st.success(f'CSV file saved to {save_path}')
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")

# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib

# # Define the paths to your joblib files
# model_file_path = 'knnmainnew_model.joblib'
# encoder_file_path = 'encoder.joblib'
# scaler_file_path = 'scaler.joblib'

# # Load the model, label encoder, and scaler using joblib
# model = joblib.load(model_file_path)
# label_encoder = joblib.load(encoder_file_path)
# scaler = joblib.load(scaler_file_path)

# # Initialize a DataFrame to store previous predictions
# if 'previous_predictions' not in st.session_state:
#     st.session_state.previous_predictions = pd.DataFrame(columns=['Week', 'Temperature', 'Humidity', 'GasLevel', 'Predicted Status'])

# # Custom CSS to set the background color, text color, and add a watermark
# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #004d00;  /* Dark green background */
#         color: white;  /* White text color */
#     }
#     .css-18e3th9 { /* Header */
#         background-color: #003300;  /* Darker green for header */
#         color: white;  /* White text color */
#     }
#     .css-1j7a9z2 { /* Button */
#         background-color: #006400;  /* Dark green for button */
#         color: white;  /* White text color on button */
#     }
#     .css-1n7v3w8 input { /* Input fields */
#         color: white;  /* White text color in input fields */
#         background-color: gray;  /* Gray background for input fields */
#     }
#     .css-1v3h5q2 { /* Text Input */
#         color: white;  /* White text color in text input fields */
#         background-color: gray;  /* Gray background for text input fields */
#     }
#     .css-10trblm { /* Title */
#         color: white;  /* White text color for the title */
#     }
#     .watermark {
#         position: fixed;
#         bottom: 10px;
#         right: 10px;
#         opacity: 0.5;
#         z-index: -1;
#         pointer-events: none;
#     }
#     </style>
#     <div class="watermark">
#         <img src="https://example.com/path-to-your-watermark-image.png" alt="Poultry Farm" width="200"/>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# st.title("Environmental Monitoring App")

# st.write("Enter feature values for prediction:")

# # Input fields
# Week = st.number_input('Week', value=1)
# Temperature = st.number_input('Temperature', value=0.0)
# Humidity = st.number_input('Humidity', value=0.0)
# GasLevel = st.number_input('GasLevel', value=0.0)

# # Create a DataFrame for input features
# cols = ['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']
# input_data = pd.DataFrame([[Week, 'M', Temperature, Humidity, GasLevel]], columns=cols)

# # Transform 'Prev_Status' using the label encoder
# try:
#     input_data['Prev_Status'] = label_encoder.transform(input_data[['Prev_Status']])
# except ValueError:
#     # Assign a default value if category is unknown
#     input_data['Prev_Status'] = label_encoder.transform([['M']])[0]

# # Ensure the order of columns matches the trained scaler's expectations
# input_data = input_data[['Week', 'Prev_Status', 'Temp', 'Hum', 'Gas']]

# # Scale the features
# input_data_scaled = scaler.transform(input_data)

# if st.button('Predict'):
#     # Perform prediction
#     prediction = model.predict(input_data_scaled)
#     predicted_status = prediction[0]
    
#     # Display the result
#     st.write(f'Prediction: {predicted_status}')
    
#     # Update 'Prev_Status' with the prediction result
#     new_prediction = pd.DataFrame({
#         'Week': [Week],
#         'Temperature': [Temperature],
#         'Humidity': [Humidity],
#         'GasLevel': [GasLevel],
#         'Predicted Status': [predicted_status]
#     })
    
#     st.session_state.previous_predictions = pd.concat([st.session_state.previous_predictions, new_prediction], ignore_index=True)

# # Display previous predictions and the legend side by side
# col1, col2 = st.columns([4,2])

# with col1:
#     st.write("**Previous Predictions**")
#     st.dataframe(st.session_state.previous_predictions)

# with col2:
#     st.write("**Prediction Legend**")
#     st.write("**U**: Unsafe ", "**M**: Moderately safe ")
#     st.write("**S**: Safe")
