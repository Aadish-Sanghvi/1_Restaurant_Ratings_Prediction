import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
MODEL_PATH = "/Users/sanghvi/Desktop/Coding/ML/1_Restaurant_Ratings_Prediction/model_2.pkl"  # Ensure the trained model is saved as 'model.pkl'
OHE_PATH = "/Users/sanghvi/Desktop/Coding/ML/1_Restaurant_Ratings_Prediction/ohe.pkl"  # OneHotEncoder used for 'Average Cost for two'
SCALER_PATH = "/Users/sanghvi/Desktop/Coding/ML/1_Restaurant_Ratings_Prediction/scaler.pkl"  # StandardScaler for 'Votes'

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    ohe = joblib.load(OHE_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, ohe, scaler

# Streamlit UI with advanced animations and visualizations
st.set_page_config(page_title="Restaurant Rating Prediction", layout="centered", page_icon="🍽️")

# Custom CSS for advanced UI
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.9);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            margin: 40px auto;
            backdrop-filter: blur(10px);
            position: relative;
            z-index: 1;
        }
        .stTitle {
            color: #2E3B4E;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
            animation: fadeIn 1s ease-in-out;
        }
        .stButton > button {
            background: linear-gradient(135deg, #ff5733, #c70039);
            color: white;
            font-size: 18px;
            padding: 12px 28px;
            border-radius: 50px;
            border: none;
            transition: 0.3s;
            width: 100%;
            cursor: pointer;
            box-shadow: 0px 4px 10px rgba(255, 87, 51, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0px 6px 15px rgba(255, 87, 51, 0.5);
        }
        .stInput label, .stSelectbox label, .stRadio label {
            font-weight: bold;
            color: #2E3B4E;
            font-size: 16px;
        }
        .stNumberInput input, .stSelectbox select, .stRadio label {
            font-size: 16px;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ced4da;
            transition: 0.3s;
        }
        .stNumberInput input:focus, .stSelectbox select:focus, .stRadio input:focus {
            border-color: #ff5733;
            box-shadow: 0 0 10px rgba(255, 87, 51, 0.5);
        }
        .prediction-result {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #ff5733;
            margin-top: 30px;
            animation: slideIn 0.5s ease-in-out;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #6c757d;
            font-size: 14px;
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .cursor-effect {
            position: fixed;
            width: 20px;
            height: 20px;
            background: rgba(255, 87, 51, 0.5);
            border-radius: 50%;
            pointer-events: none;
            transform: translate(-50%, -50%);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.5); opacity: 0.5; }
            100% { transform: scale(1); opacity: 1; }
        }
        .visualization {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }
        .visualization .circle {
            position: absolute;
            width: 20px;
            height: 20px;
            background: rgba(255, 87, 51, 0.3);
            border-radius: 50%;
            animation: float 5s infinite;
        }
        @keyframes float {
            0% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# Main container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='stTitle'>🍽️ Restaurant Rating Prediction</h1>", unsafe_allow_html=True)
st.write("Enter the restaurant details below to predict its aggregate rating. This tool uses advanced machine learning to provide accurate predictions!")

# User input fields
with st.container():
    avg_cost = st.number_input("Average Cost for Two ($)", min_value=0, step=50, help="Enter the average cost for two people.")
    has_table_booking = st.radio("Has Table Booking?", ["No", "Yes"], horizontal=True, help="Does the restaurant offer table booking?")
    has_online_delivery = st.radio("Has Online Delivery?", ["No", "Yes"], horizontal=True, help="Does the restaurant offer online delivery?")
    is_delivering_now = st.radio("Is Delivering Now?", ["No", "Yes"], horizontal=True, help="Is the restaurant currently delivering?")
    price_range = st.selectbox("Price Range", [1, 2, 3, 4], help="Select the price range (1 = low, 4 = high).")
    votes = st.number_input("Votes", min_value=0, step=1, help="Enter the number of votes the restaurant has received.")

# Predict button
if st.button("Predict Rating"):
    model, ohe, scaler = load_artifacts()
    
    # Encode categorical inputs
    has_table_booking = 1 if has_table_booking == "Yes" else 0
    has_online_delivery = 1 if has_online_delivery == "Yes" else 0
    is_delivering_now = 1 if is_delivering_now == "Yes" else 0
    
    # One-Hot Encode 'Average Cost for two'
    avg_cost_encoded = ohe.transform([[avg_cost]])
    avg_cost_encoded_df = pd.DataFrame(avg_cost_encoded, columns=ohe.get_feature_names_out(['Average Cost for two']))
    
    # Scale 'Votes'
    votes_scaled = scaler.transform(np.array(votes).reshape(-1, 1))[0][0]
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Has Table booking': [has_table_booking],
        'Has Online delivery': [has_online_delivery],
        'Is delivering now': [is_delivering_now],
        'Price range': [price_range],
        'Votes': [votes_scaled]
    })
    
    # Concatenate one-hot encoded cost data
    input_data = pd.concat([input_data, avg_cost_encoded_df], axis=1)
    
    # Ensure all feature columns match the model's expected features
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match model's training data
    input_data = input_data[model.feature_names_in_]
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Display result with animation
    st.markdown(f"<div class='prediction-result'>🌟 Predicted Rating: {round(prediction, 2)}</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Made with ❤️ by Aadish Sanghvi | Powered by Streamlit</div>", unsafe_allow_html=True)

# Close main container
st.markdown("</div>", unsafe_allow_html=True)

# Cursor effect
st.markdown("<div class='cursor-effect'></div>", unsafe_allow_html=True)

# Visualizations
st.markdown("<div class='visualization'><div class='circle' style='top: 20%; left: 10%;'></div><div class='circle' style='top: 50%; left: 30%;'></div><div class='circle' style='top: 80%; left: 70%;'></div></div>", unsafe_allow_html=True)