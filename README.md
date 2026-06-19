# Egypt National Team Match Predictor

Welcome to the Egypt Match Prediction System. This project brings a bit of data science to football by predicting how the Egyptian national team will perform. Using a couple of machine learning models, you can tweak real-world match conditions and see what the stats say about the outcome.

## How It Works

This system uses two models working behind the scenes, alongside a web interface to make things easy to use:

* Goals Predictor (linear_reg.py): This script trains a Ridge Regression model to figure out exactly how many goals Egypt is likely to score based on things like rest days and the opponent's defensive stats.
* Outcome Predictor (log_reg.py): This trains a Ridge Classifier to look at the bigger picture and predict whether the match will end in a win, or a draw/loss.
* Web App (app.py): A clean, interactive Streamlit dashboard where you can slide around metrics like the opponent's FIFA rank, current team form, and whether Mohamed Salah is playing to get instant predictions.

## Getting Started

If you want to run this locally, follow these steps to get everything up and running.

1. Clone this repository to your local machine and navigate into the project directory.

2. Install the required Python packages:
   pip install numpy scikit-learn streamlit

3. Train the models. You need to run these scripts first so they can generate the required .pkl files:
   python linear_reg.py
   python log_reg.py

4. Launch the Streamlit interface:
   streamlit run app.py

## Project Structure

* linear_reg.py: The training script and dataset for the goal prediction model.
* log_reg.py: The training script and dataset for the match outcome classification model.
* app.py: The Streamlit frontend that loads the models and provides the UI.
* lin_model.pkl: The generated binary file holding the trained linear model (created after running the setup).
* log_model.pkl: The generated binary file holding the trained classification model (created after running the setup).
