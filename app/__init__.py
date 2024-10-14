#app/__init__.py
from flask import Flask
from flask_cors import CORS
import os
import pickle
from model.model import Model
from app.config import Config
import pandas as pd
#from .database import init_db
from app.database import get_values_from_table




def create_app():
    
    # Load the model
    
    file_name = 'model.pkl'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    file_path = os.path.join(base_dir, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            
        print("Model exists and its loaded.")
        
    else:
        
        data, columns = get_values_from_table(table_name = 'heart_disease', database_name= 'database_heart_disease', where_filter = '')
        df = pd.DataFrame(data, columns = columns)
        
        y_fit = df['target']
        X_fit = df.drop('target', axis = 1)
    
        model = Model()
        model.fit_model(X_fit = X_fit, y_fit = y_fit, save_model = True)
        
        print("Model was fitted and saved")

        #model.evaluate_model(X_fit, y_fit)
    
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)
    
    #Saving the model
    
    app.config['MODEL'] = model
    
    # Inicializa o banco de dados
    #init_db(app)
    
    # Registra as rotas
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app



app = create_app()

if __name__ == '__main__':
    # Executa o servidor Flask
    app.run(debug=True, host='0.0.0.0')