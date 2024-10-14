# Heart Disease Prediction Web Application

This project is a web application that predicts the probability of heart disease based on patient data. The system is built using Python, Flask for the server, and a machine learning model for prediction.


## Project structure

```
├── app/
│   ├── __init__.py              # Initialize the aplication
│   ├── templates/
│   │   └── index.html           # The front-end page where users input patient data
│   ├── static           
|   │   └── script.html          # Front-end script for the webpage
|   |   └── css.html             # Front-end styles for the webpage
|   ├── database.py              # Module of functions for database management
|   ├── routes.py                # Definitions of the api routes
|   └── config.pkl               # Configurations for the app
├── data/                        # Contains datasets used for model training
├── model/
│   ├── model_class.py           # Class for model training, saving, loading, evaluating, and predicting
│   ├── model.pkl                # The saved trained model (created after training)
|   └── config.pkl               # The parameters of the model
├── notebook/
│   ├── EDA.ipynb                # Exploratory Data Analysis
│   └── model_selection.ipynb    # Notebooks for model selection and testing
├── venvHeatDisease/             # Virtual environment for project dependencies
├── database                     # A copy of my database used for this project (MySQL)
├── .env                         # Environment variables (database credentials, API keys, etc.)
├── requirements.txt             # List of dependencies
└── run                          # File to start the server and initialize the app
```
## How to Run the Project

1. **Activate Virtual Environment:**

   If you haven't already, activate the virtual environment:

   - On Mac/Linux:
     ```bash
     source venvHeatDisease/bin/activate
     ```

   - On Windows:
     ```bash
     venvHeatDisease\Scripts\activate
     ```

2. **Set Environment Variables:**

   Make sure your `.env` file contains the necessary environment variables, such as database credentials and any API keys.

3. **Database Setup:**

A copy of the MySQL database is provided in the `databases` folder as `database_backup.sql`. To restore the database, use the following command:

```bash
mysql -u root -p database_heart_disease < /databases/database_backup.sql
```
4. **Run the Server:**

   Start the server by executing the `run` file in the project root:

   ```bash
   python run
   ```
This will initialize the Flask app located in the app directory.

5. **Access the Application:**

   Open your browser and navigate to http://localhost:5000/templates/index.html. You will see the web interface where you can input patient data and receive a heart disease probability prediction.

## Model Information

The `model` folder contains the machine learning model used for prediction and the script to train the model. The model class has the following methods:

- **fit_model**: Used to train the model on new data.
- **save_model**: Saves the trained model to a file (`model.pkl`).
- **load_model**: Loads a saved model from a file.
- **evaluate_model**: Evaluates the model performance on a dataset.
- **predict_heart_disease**: Predicts the probability of heart disease based on patient data.

## Notebooks

In the `notebook` directory, there are two Jupyter notebooks:

- **EDA.ipynb**: Contains the exploratory data analysis used to understand the dataset and relationships between variables.
- **Model development.ipynb**: Contains the process of testing and selecting the best machine learning model for predicting heart disease.

## Future Steps

- Improve model accuracy by gathering more data and tuning hyperparameters.
- Refine the feature selection process to ensure the model only uses the most important variables.
- Deploy the application to a cloud platform for wider access.

## Conclusion

This project demonstrates a full pipeline from data exploration and model selection to building a web application that predicts the probability of heart disease based on patient input.
