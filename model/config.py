# model/config.py

class Config:
    
   
    model_parameters = {
                        'colsample_bytree': 0.3,
                        'learning_rate': 0.1,
                        'max_depth': 3,
                        'n_estimators': 20,
                        'subsample': 0.8, 
                        'eval_metric':'mlogloss',}
