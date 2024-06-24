import streamlit as st
import pickle
import numpy as np


# Load the model
with open('best_gb_model (1).pkl', 'rb') as file:
    model = pickle.load(file)


# Streamlit app
st.title("Player Data Prediction")

st.sidebar.header("Player Data Input")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
    wage_eur = st.sidebar.number_input("Wage (EUR)", min_value=0, max_value=1000000, value=50000)
    potential = st.sidebar.number_input("Potential", min_value=0, max_value=100, value=70)
    movement_reactions = st.sidebar.number_input("Movement Reactions", min_value=0, max_value=100, value=50)
    power_stamina = st.sidebar.number_input("Power Stamina", min_value=0, max_value=100, value=50)
    skill_ball_control = st.sidebar.number_input("Skill Ball Control", min_value=0, max_value=100, value=50)
    attacking_heading_accuracy = st.sidebar.number_input("Attacking Heading Accuracy", min_value=0, max_value=100, value=50)
    dribbling = st.sidebar.number_input("Dribbling", min_value=0, max_value=100, value=50)
    defending = st.sidebar.number_input("Defending", min_value=0, max_value=100, value=50)
    physic = st.sidebar.number_input("Physic", min_value=0, max_value=100, value=50)
    movement_sprint_speed = st.sidebar.number_input("Movement Sprint Speed", min_value=0, max_value=100, value=50)
    goalkeeping_reflexes = st.sidebar.number_input("Goalkeeping Reflexes", min_value=0, max_value=100, value=50)
    mentality_composure = st.sidebar.number_input("Mentality Composure", min_value=0, max_value=100, value=50)

    data = {
        'age': age,
        'wage_eur': wage_eur,
        'potential': potential,
        'movement_reactions': movement_reactions,
        'power_stamina': power_stamina,
        'skill_ball_control': skill_ball_control,
        'attacking_heading_accuracy': attacking_heading_accuracy,
        'dribbling': dribbling,
        'defending': defending,
        'physic': physic,
        'movement_sprint_speed': movement_sprint_speed,
        'goalkeeping_reflexes': goalkeeping_reflexes,
        'mentality_composure': mentality_composure
    }
    return data

input_data = user_input_features()

if st.sidebar.button('Predict'):
    # Convert data to the appropriate format and apply scaling
    features = np.array([[
        input_data['age'], input_data['wage_eur'], input_data['potential'], input_data['movement_reactions'], input_data['power_stamina'], 
        input_data['skill_ball_control'], input_data['attacking_heading_accuracy'], input_data['dribbling'], input_data['defending'], 
        input_data['physic'], input_data['movement_sprint_speed'], input_data['goalkeeping_reflexes'], input_data['mentality_composure']
    ]], dtype=float)

 
    # Make prediction
    prediction = model.predict(features)

    st.subheader("Prediction Result")
    st.write(f"Predicted Value: {prediction[0]}")