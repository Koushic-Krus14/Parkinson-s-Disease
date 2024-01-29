import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the logistic regression model
lr_loaded = joblib.load(r'C:\Users\rshic\Downloads\logistic_regression_model.joblib')

# Feature names
feature_names = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", 
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# Default values based on the provided dataset
default_values = [119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
                  0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654]

def main():
    st.title('Parkinson\'s Disease Prediction App')

    st.sidebar.header('User Input Features')

    # Create text boxes for each feature with default values
    features = [st.sidebar.text_input(feature_name, value=default_value) for feature_name, default_value in zip(feature_names, default_values)]

    # Convert user inputs to float
    features = [float(feature) if feature else 0.0 for feature in features]

    # Create a DataFrame with the user input features
    input_data = pd.DataFrame({feature_name: [feature] for feature_name, feature in zip(feature_names, features)})

    # Scale the user input features
    scaler = MinMaxScaler((-1, 1))
    input_data_scaled = scaler.fit_transform(input_data)

    # Add a button for submission
    if st.sidebar.button('Submit'):
        # Make a prediction using the logistic regression model
        prediction = lr_loaded.predict(input_data_scaled.reshape(1, -1))

        st.subheader('Prediction')
        if prediction[0] == 1:
            st.write("The model predicts Parkinson's Disease.")
        else:
            st.write("The model predicts No Parkinson's Disease.")

if __name__ == '__main__':
    main()
