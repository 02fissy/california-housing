import streamlit as st
import joblib
import numpy as np

try:
    model = joblib.load('best_house_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Model or Scaler files not found. Please run 'python ml_model.py' first.")
    st.stop()



feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]
st.title('🏠 California House Price Predictor (Best Model)') 
st.markdown("""
    This app uses the best-performing regression model (selected from Linear, Ridge, and LASSO)
    to predict the Median House Value in a given California block group.
    (Prediction output is in hundreds of thousands of dollars - $100,000s)
""")

st.header('Input House Group Features')

input_data = {}
col1, col2 = st.columns(2)

input_map = {
    'MedInc': ('Median Income (e.g., 8.3252 for $83,252)', 0.0, 15.0, 3.87067),
    'HouseAge': ('Median House Age', 1.0, 52.0, 28.639486),
    'AveRooms': ('Average Number of Rooms', 0.0, 100.0, 5.429),
    'AveBedrms': ('Average Number of Bedrooms', 0.0, 30.0, 1.096),
    'Population': ('Block Population', 0.0, 35682.0, 1425.476),
    'AveOccup': ('Average House Occupancy', 0.0, 100.0, 3.07),
    'Latitude': ('Latitude (e.g., 34.0)', 32.0, 42.0, 34.05),
    'Longitude': ('Longitude (e.g., -118.0)', -125.0, -114.0, -118.24)
}

# Distribute inputs across two columns
for i, (name, details) in enumerate(input_map.items()):
    label, min_val, max_val, default_val = details
    if i % 2 == 0:
        with col1:
            input_data[name] = st.number_input(label, min_value=min_val, max_value=max_val, value=default_val)
    else:
        with col2:
            input_data[name] = st.number_input(label, min_value=min_val, max_value=max_val, value=default_val)


if st.button('Predict House Value'):
    features = np.array([input_data[name] for name in feature_names])
    features = features.reshape(1, -1)
    
    scaled_features = scaler.transform(features)
    
    prediction_raw = model.predict(scaled_features)[0]
    
    prediction_dollars = prediction_raw * 100000 
    
    
    st.success(f"### Predicted Median House Value:")
    st.success(f"**${prediction_dollars:,.2f}**")
    st.info(f"The raw model output (in $100,000s) is: {prediction_raw:.4f}")