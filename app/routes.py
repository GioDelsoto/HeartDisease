# app/routes.py
from flask import Blueprint, current_app, request, jsonify
import numpy as np
import pandas as pd
from app.database import get_values_from_table


main_bp = Blueprint('main', __name__)

# Endpoint para adicionar dados
@main_bp.route('/get_dataframe', methods=['GET'])
def get_dataframe():
    
    data, columns = get_values_from_table(table_name = 'heart_disease', database_name= 'database_heart_disease', where_filter = '')
    dataframe = pd.DataFrame(data, columns = columns)
    
    return jsonify(dataframe.to_dict(orient='records'))



@main_bp.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    # Get JSON data from the request
    
    model = current_app.config['MODEL']
    
    data = request.get_json()
    features = pd.DataFrame.from_dict([data])
    print(features)
    
    # Check if data is None or empty
    if data is None:
        return jsonify({'error': 'No data provided'}), 400
    
    # Convert JSON data to numpy array (assuming input is a list of dictionaries)
    #input_data = np.array([list(item.values()) for item in data])
    
    # Get predictions from the model
    probabilities = model.predict_heart_disease(features)
    probability_disease = probabilities.tolist()[0][1]
    
    if probability_disease < 0.2:
        text_warning = "Low risk of heart disease."
    elif probability_disease < 0.5:
        text_warning = "Medium risk of heart disease."
    elif probability_disease >= 0.5:
        text_warning = "High risk of heart disease."
        
    
    # Prepare response
    response = {
        'probabilities': probabilities.tolist()[0]
    }
    
    # JSON RESPONSE
    response = {
            'status': 'success',
            'message': 'Prediction successful',
            'probability': f"{probabilities.tolist()[0][1]*100:.2f}%",
            'warning':text_warning
        }

    return jsonify(response), 200

@main_bp.route('/model_evaluation', methods=['POST'])
def model_evaluation():
    model = current_app.config['MODEL']
    
    data = request.get_json()
    df = pd.DataFrame.from_dict(data)

    X_eval = df.drop('target', axis = 1)
    y_eval = df['target']
    model.evaluate_model(X_eval, y_eval)
    
    response = {
            'status': 'success',
            'message': 'Evaluation finished with success',
        }

    return jsonify(response), 200