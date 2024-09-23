import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Function to predict stroke based on input features
def predict_stroke(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    return prediction, probability

# Create a Streamlit web app
def main():
    # Set app title and description
    st.title("Stroke Prediction Web App")
    st.write("Enter the required information to predict the likelihood of stroke.")

    # Create input fields for user to enter information
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    hypertension = st.selectbox("Hypertension", ("Yes", "No"))
    heart_disease = st.selectbox("Heart Disease", ("Yes", "No"))
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, value=20.0)
    gender = st.selectbox("Gender", ("Male", "Female"))
    smoking_status = st.selectbox("Smoking Status", ("Unknown", "Formerly Smoked", "Never Smoked", "Smokes"))
    ever_married = st.selectbox("ever_married", ("Yes", "No"))
    work_type = st.selectbox("work_type Status", ("Private", "Self-employed", "children", "Govt_job","Never_worked"))
    Residence_type = st.selectbox("Residence_type", ("Urban", "Rural"))

    # Convert categorical inputs to numerical values
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    gender = 1 if gender == "Male" else 0
    ever_married = 1 if ever_married == "Yes" else 0
    Residence_type = 1 if gender == "Urban" else 0

    # Map smoking status to numerical values
    smoking_map = {
        "Unknown": 0,
        "Formerly Smoked": 1,
        "Never Smoked": 2,
        "Smokes": 3
    }
    smoking_status = smoking_map[smoking_status]

    # Map work_type status to numerical values
    work_type_map = {
        "Govt_job": 0,
        "Never_worked": 1,
        "Private": 2,
        "Self-employed": 3,
        "children": 4,
    }
    work_type = work_type_map[work_type]

    # Create a button to predict stroke
    if st.button("Predict Stroke"):
        # Gather input features
        features = [age, hypertension, heart_disease, avg_glucose_level, bmi, gender, smoking_status, ever_married, work_type, Residence_type]

        # Predict stroke and probability
        prediction, probability = predict_stroke(features)

        # Display the prediction
        if prediction[0] == 0:
            st.write("Congratulations! You have a low risk of stroke.")
        else:
            st.write("Warning! You are at a high risk of stroke.")
            st.write("Probability of stroke:", probability)

# Run the web app
if __name__ == "__main__":
    main()
