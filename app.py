from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Load the pre-trained RandomForest model and scaler
model = pickle.load(open('./rf_model.pkl', 'rb'))
scaler = pickle.load(open('./scaler.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Define the prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get user inputs
            data = request.json
            age = float(data['age'])
            avg_glucose_level = float(data['avg_glucose_level'])
            bmi = float(data['bmi'])
            gender = int(data['gender'])
            ever_married = int(data['ever_married'])
            work_type = int(data['work_type'])
            residence_type = int(data['residence_type'])
            smoking_status = int(data['smoking_status'])
            hypertension = int(data['hypertension'])
            heart_disease = int(data['heart_disease'])

            # Prepare and scale numerical features
            numerical_features = np.array([age, avg_glucose_level, bmi]).reshape(1, -1)
            numerical_features_scaled = scaler.transform(numerical_features)

            # Prepare categorical features
            categorical_features = np.array([gender, ever_married, work_type, residence_type, smoking_status, hypertension, heart_disease])

            # Combine scaled numerical and categorical features
            final_features = np.concatenate((numerical_features_scaled[0], categorical_features))

            # Make prediction
            prediction = model.predict(final_features.reshape(1, -1))[0]
            probability = model.predict_proba(final_features.reshape(1, -1))[0][1]  # Probability of stroke (class 1)

            # Prepare result message
            result = {
                'stroke': int(prediction),
                'probability': float(probability) * 100  # Convert to percentage
            }
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)})
    
    # Render the correct HTML file for GET request
    return render_template('Strock.html')

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
