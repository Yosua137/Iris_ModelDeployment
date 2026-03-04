import streamlit as st
import joblib
import numpy as np
import pandas as np

scaler = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load ("artifacts/model.pkl")

def main():
    st.title('Machine Learning Iris Prediction Model Deployment')

    sepal_length = st.number_input("Input nilai sepal_length", min_value = 0.0, max_value = 10.0, value = 0.1)
    sepal_width = st.number_input("Input nilai sepal_width", min_value = 0.0, max_value = 10.0, value = 0.1)
    petal_length = st.number_input("Input nilai petal_length", min_value = 0.0, max_value = 10.0, value = 0.1)
    petal_width = st.number_input("Input nilai petal_width", min_value = 0.0, max_value = 10.0, value = 0.1)

    if st.button('Make Prediction'):
        features = [sepal_length, sepal_width, petal_length, petal_width]
        result = make_prediction(features)
        st.success(f"The prediciton is: {result}")

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(input_array)
    prediction = model.predict(X_scaled)
    return prediction[0]

if __name__ == '__main__':
    main()