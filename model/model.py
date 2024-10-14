import requests
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from model.config import Config
import os


class Model():
    
    def __init__(self, steps = []):
        
        self.categorical_vars = ['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal']
        self.continuous_vars = ['age', 'trestbps', 'thalach', 'oldpeak']
        self.target = ['target']
                
        mapper = DataFrameMapper(
              [([continuous_col], StandardScaler()) for continuous_col in self.continuous_vars] +
              [(categorical_col, LabelBinarizer()) for categorical_col in self.categorical_vars]
            )
        
        model_parameters = Config.model_parameters
        estimator = xgb.XGBClassifier( random_state=42, **model_parameters)
        
        
        self.pipeline = Pipeline(
                            [  ("mapper", mapper),
                               ("estimator", estimator)  ]
                            )
        return
    
    
    def fit_model(self, X_fit =[], y_fit=[], save_model = False):
        
        '''
            Fits the model using the provided data or data fetched from a database.
            
            Parameters:
            - X_fit: DataFrame containing the features for training the model.
            - y_fit: Series or DataFrame containing the target values for training the model.
            - save_model: Boolean flag indicating whether to save the trained model. If True, the model will be saved to the specified path.
            
            - The provided X_fit and y_fit will be used to train the model.
            - The model will be trained and evaluated using the provided data.
            
            The model's pipeline is fitted with the features from X_fit, and the selected features are stored in self.data_columns.
        '''
        
           
        if len(X_fit)==0 or len(y_fit)==0:
            print("X_fit or y_fit have no elements.")
        
        
        else:
            #Select only important features
            X_fit = X_fit[self.categorical_vars + self.continuous_vars]
            
            self.data_columns = X_fit.columns
            self.pipeline.fit(X_fit, y_fit)
            print("Model fitted with success!")
            
            
            if save_model == True:
                
                file_name = 'model.pkl'
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
                model_path = os.path.join(base_dir, file_name)
                
                self.save_model(model_path)
                print(f"Model saved in {model_path}")


    
    def evaluate_model(self, X_eval, y_eval): 
        #Select only important features
        X_eval = X_eval[self.categorical_vars + self.continuous_vars]

        cv_scores = cross_val_score(self.pipeline, X_eval, y_eval, cv=5, scoring='f1')
        print(f'F1 Score (Cross-Validation): {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')
    
        y_predict = self.pipeline.predict(X_eval)
        conf_matrix = confusion_matrix(y_eval, y_predict)
        # Plota a Matriz de Confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
                
                
    
    def predict_heart_disease(self, X_input):
        #Select only important features
        X_input = X_input[self.categorical_vars + self.continuous_vars]

        probs = self.pipeline.predict_proba(X_input)
        print(probs)
        return probs
    
    
    def save_model(self, filename):
        
        #os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

