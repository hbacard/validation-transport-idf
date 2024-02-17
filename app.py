import numpy as np
import pickle
import streamlit as st
from pathlib import Path

st.title("Prediction nombre de valiations OLYMPIADES")

features = [
    'JOUR DE LA SEMAINE',
    'MOIS',
    'NUMERO JOUR',
    'CATEGORIE TITRE AUTRE TITRE',
    'CATEGORIE TITRE FGT',
    'CATEGORIE TITRE IMAGINE R',
    'CATEGORIE TITRE INCONNU',
    'CATEGORIE TITRE NAVIGO',
    'CATEGORIE TITRE NAVIGO JOUR',
    'CATEGORIE TITRE NON DEFINI',
    'CATEGORIE TITRE TST'
]


# JOUR_MAPPING = {
#     "Lundi"
# }

WORK_DIR = Path(__file__).resolve().parent
MODELS_DIR = "saved_models"
MODEL_NAME = "trained_model_OLYMPIADES.pickle"
MODEL_PATH = WORK_DIR.joinpath(MODELS_DIR, MODEL_NAME)
def main():
    # load model
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    
    weekday = st.number_input(label='JOUR DE LA SEMAINE',step=1, min_value=0, max_value=6)
    month = st.number_input('MOIS', min_value=1, max_value=12, step=1)
    day_number = st.number_input('NUMERO JOUR', max_value=31, min_value=1, step=1)
    # other_ticket_type = st.number_input('CATEGORIE TITRE AUTRE TITRE')
    # fgt = st.number_input('CATEGORIE TITRE FGT')
    # image_r = st.number_input('CATEGORIE TITRE IMAGINE R')
    # unknown = st.number_input('CATEGORIE TITRE INCONNU')
    # navigo = st.number_input('CATEGORIE TITRE NAVIGO')
    # navigo_day = st.number_input('CATEGORIE TITRE NAVIGO JOUR')
    # non_defined = st.number_input('CATEGORIE TITRE NON DEFINI')
    # tst = st.number_input('CATEGORIE TITRE TST')
    ticket_category = st.selectbox(label="Categorie titre", options=['CATEGORIE TITRE AUTRE TITRE',
    'CATEGORIE TITRE FGT',
    'CATEGORIE TITRE IMAGINE R',
    'CATEGORIE TITRE INCONNU',
    'CATEGORIE TITRE NAVIGO',
    'CATEGORIE TITRE NAVIGO JOUR',
    'CATEGORIE TITRE NON DEFINI',
    'CATEGORIE TITRE TST'])

    input_features = [weekday, month, day_number]
    for _, feature in enumerate(features[3:]):
        if feature == ticket_category:
            input_features.append(1)
        else:
            input_features.append(0)
    

    
    input_vector = np.array(input_features)
    if st.button("Predict"):
        try:
            prediction = model.predict([input_vector])
            st.write(f"Number of ticket validations: {prediction}")
        except Exception as e:
            st.write("An erro occured:", e)





if __name__ == "__main__":
    main()