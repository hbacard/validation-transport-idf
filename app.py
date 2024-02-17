import numpy as np
import pickle
import streamlit as st
from pathlib import Path

# Load necessary libraries and define constants
WORK_DIR = Path(__file__).resolve().parent
MODELS_DIR = "saved_models"

JOURNAL_WEEKDAY_NAMES = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

MODEL_NAME = "trained_model_OLYMPIADES.pickle"
MODEL_PATH = WORK_DIR.joinpath(MODELS_DIR, MODEL_NAME)

# Define a function to convert weekday number to name
def get_weekday_name(weekday):
    return JOURNAL_WEEKDAY_NAMES[weekday]

# Create a title for the Streamlit app
st.title("Prediction nombre de valiations OLYMPIADES")

# Define input features and their labels
features = [
    "JOUR DE LA SEMAINE",
    "MOIS",
    "NUMERO JOUR",
    "CATEGORIE TITRE AUTRE TITRE",
    "CATEGORIE TITRE FGT",
    "CATEGORIE TITRE IMAGINE R",
    "CATEGORIE TITRE INCONNU",
    "CATEGORIE TITRE NAVIGO",
    "CATEGORIE TITRE NAVIGO JOUR",
    "CATEGORIE TITRE NON DEFINI",
    "CATEGORIE TITRE TST"
]

# Load the model and create a main function for better organization
def load_and_predict():
    # Load the trained model
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)

    # Create input widgets for user inputs
    weekday = st.selectbox(label="JOUR DE LA SEMAINE", options=range(len(JOURNAL_WEEKDAY_NAMES)), format_func=get_weekday_name)
    month = st.number_input("MOIS", min_value=1, max_value=12, step=1)
    day_number = st.number_input("NUMERO JOUR", max_value=31, min_value=1, step=1)

    ticket_category = st.selectbox(label="Categorie titre", options=[
        "CATEGORIE TITRE AUTRE TITRE",
        "CATEGORIE TITRE FGT",
        "CATEGORIE TITRE IMAGINE R",
        "CATEGORIE TITRE INCONNU",
        "CATEGORIE TITRE NAVIGO",
        "CATEGORIE TITRE NAVIGO JOUR",
        "CATEGORIE TITRE NON DEFINI",
        "CATEGORIE TITRE TST"
    ])

    # Create an input vector based on user inputs and model features
    input_features = [weekday, month, day_number]
    for idx, feature in enumerate(features[3:]):
        if feature == ticket_category:
            input_features.append(1)
        else:
            input_features.append(0)

    # Convert the input vector to a numpy array
    input_vector = np.array(input_features)

    # Add a button for prediction and handle exceptions
    if st.button("Predict"):
        try:
            prediction = model.predict([input_vector])
            st.write(f"Number of ticket validations: {prediction[0]}")
        except Exception as e:
            st.write("An error occurred:", e)

# Call the main function when the script is run directly or from Streamlit
if __name__ == "__main__":
    load_and_predict()