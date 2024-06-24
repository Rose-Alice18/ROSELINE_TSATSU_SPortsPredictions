import gzip
import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the scaler and model from compressed files
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with gzip.open('ensemble_model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title("Player Data Prediction")

st.sidebar.header("Player Data Input")

def user_input_features():
    movement_reactions = st.sidebar.number_input("Movement Reactions", min_value=0, max_value=100, value=50)
    potential = st.sidebar.number_input("Potential", min_value=0, max_value=100, value=70)
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
    power_stamina = st.sidebar.number_input("Power Stamina", min_value=0, max_value=100, value=50)
    dribbling = st.sidebar.number_input("Dribbling", min_value=0, max_value=100, value=50)
    physic = st.sidebar.number_input("Physic", min_value=0, max_value=100, value=50)
    movement_sprint_speed = st.sidebar.number_input("Movement Sprint Speed", min_value=0, max_value=100, value=50)
    mentality_composure = st.sidebar.number_input("Mentality Composure", min_value=0, max_value=100, value=50)
    skill_ball_control = st.sidebar.number_input("Skill Ball Control", min_value=0, max_value=100, value=50)

    data = {
        'movement_reactions': movement_reactions,
        'potential': potential,
        'age': age,
        'power_stamina': power_stamina,
        'dribbling': dribbling,
        'physic': physic,
        'movement_sprint_speed': movement_sprint_speed,
        'mentality_composure': mentality_composure,
        'skill_ball_control': skill_ball_control
    }
    return data

input_data = user_input_features()

if st.sidebar.button('Predict'):
    # Convert data to the appropriate format and apply scaling
    features = np.array([[
        input_data['movement_reactions'], input_data['potential'], input_data['age'], input_data['power_stamina'], 
        input_data['dribbling'], input_data['physic'], input_data['movement_sprint_speed'], input_data['mentality_composure'], 
        input_data['skill_ball_control']
    ]], dtype=float)

    # Scale and transform features
    features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features)

    st.subheader("Prediction Result")
    st.write(f"Predicted Value: {prediction[0]}")
