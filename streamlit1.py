import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import nbformat
from nbconvert import HTMLExporter
import requests

# Define the path to the image
image_path = 'intro/_77f47b66-9794-484f-807e-56df65a48d68.jfif'

# Check if the image file exists
if os.path.exists(image_path):
    # Load and convert image
    img = Image.open(image_path)
    img.save('intro/_77f47b66-9794-484f-807e-56df65a48d68.jpg', 'JPEG')
    st.image(img, use_column_width=True)
else:
    st.error(f"Image file not found: {image_path}")

st.title("Projet de prédiction de la gravité des accidents en France")
st.sidebar.title("Sommaire")
pages = [
    "Introduction",
    "Exploration",
    "DataVizualization",
    "Modélisation serie temporelle",
    "Modélisation classification binaire",
    "Modélisation classification multi-classes",
    "Conclusion"
]
page = st.sidebar.radio("Aller vers", pages)

if page == "Introduction" and os.path.exists(image_path):
    st.image(img, use_column_width=True)

# Information sur les auteurs
auteurs = {
    'LECLAIR Anne': 'https://www.linkedin.com/in/anne-leclair-11319258/',
    'HERMOSILLA Edu': 'https://www.linkedin.com/in/auteur2/',
    'PARA Olivier': 'https://www.linkedin.com/in/olivierpara'
}

# Ajout des auteurs dans la barre latérale
st.sidebar.title("Auteurs")
for auteur, lien in auteurs.items():
    st.sidebar.markdown(f"{auteur} - [LinkedIn]({lien})")

st.sidebar.markdown(f"\n\n")
st.sidebar.markdown(f"Formation continue Octobre 2023, [Datascientest](https://datascientest.com/)")
st.sidebar.markdown(f"\n\n")
st.sidebar.markdown(f"Données : [Kaggle](https://www.kaggle.com/datasets/ahmedlahlou/accidents-in-france-from-2005-to-2016/data), [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/)")

if page == "Exploration":
    st.write("### Exploration")
    st.write("4 bases de données regroupant des informations récoltées depuis 2005")
    db = st.selectbox(
        "Selectinner une base de données",
        ("Usagers", "Lieux", "Caracteristiques", "Vehicules"),
        index=None,
        placeholder="explorer une base de données...",
    )
    if db == "Usagers":
        st.image("exploration/usagers.png")
        st.image("exploration/head_usagers.png")
    elif db == "Lieux":
        st.image("exploration/lieux.png")
        st.image("exploration/head_lieux.png")
    elif db == "Caracteristiques":
        st.image("exploration/carac.png")
        st.image("exploration/head_carac.png")
    elif db == "Vehicules":
        st.image("exploration/vehicules.png")
        st.image("exploration/head_vehicules.png")

if page == "DataVizualization":
    st.write("### DataVizualisation")
    visu = st.selectbox(
        "Selectinner une visualisation",
        (
            "Nombre d'accidents par jour",
            "Nombre d'accidents par heure",
            "Nombre d'accidents par jour de la semaine",
            "Nombre d'accidents par mois",
            "gravité de l'accident en fonction du genre",
            "correlation des données caracteristiques",
            "correlation des données usagers"
        ),
        index=None,
        placeholder="explorer une base de données...",
    )
    if visu == "Nombre d'accidents par jour":
        st.image("Datavizualisation/accidents_par_jour.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    elif visu == "Nombre d'accidents par heure":
        st.image("Datavizualisation/accidents_par_heure.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    elif visu == "Nombre d'accidents par jour de la semaine":
        st.image("Datavizualisation/accidents_par_jour_de_la_semaine.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    elif visu == "Nombre d'accidents par mois":
        st.image("Datavizualisation/accidents_par_mois.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    elif visu == "gravité de l'accident en fonction du genre":
        st.image("Datavizualisation/accidents_gravite_sexe.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    elif visu == "correlation des données caracteristiques":
        st.image("Datavizualisation/correlation.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    elif visu == "correlation des données usagers":
        st.image("Datavizualisation/correlation_users.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

if page == "Modélisation serie temporelle":
    st.write("### Modélisation serie temporelle")
    
    # Menu déroulant pour sélectionner le modèle
    model_choice = st.selectbox(
        "Choisissez un modèle",
        ("Modèle Random Forest", "Modèle Dynamic Factor Model", "Modèle Exponential Smoothing"),
        index=0,
        placeholder="Sélectionnez un modèle..."
    )
    
    if model_choice == "Modèle Random Forest":
        st.write("Vous avez sélectionné le Modèle Random Forest.")
        st.markdown(""" # Analyse des Séries Temporelles et Modélisation avec RandomForest

Ce notebook présente une analyse des séries temporelles de deux jeux de données : les décès (`serie_morts`) et les accidents (`serie_accs`). L'objectif est de transformer ces séries pour les rendre stationnaires, puis d'utiliser un modèle de régression RandomForest pour prédire les valeurs futures.

## Étapes du Notebook

1. **Importation des Bibliothèques** : Chargement des bibliothèques nécessaires pour l'analyse et la modélisation.
2. **Chargement des Données** : Importation des séries temporelles depuis des fichiers CSV.
3. **Prétraitement des Données** : Transformation des séries pour les rendre stationnaires, y compris la différenciation et la transformation logarithmique.
4. **Visualisation des Données** : Affichage des séries temporelles et de leurs transformations.
5. **Tests de Stationnarité** : Utilisation du test de Dickey-Fuller augmenté pour vérifier la stationnarité des séries.
6. **Préparation des Données pour la Modélisation** : Division des données en ensembles d'entraînement et de test, et normalisation des données.
7. **Entraînement et Évaluation des Modèles** : Entraînement du modèle RandomForest et évaluation de ses performances à l'aide de métriques telles que RMSE, MAPE et MASE.
8. **Analyse des Résultats** : Affichage des résultats et des importances des variables pour les deux jeux de données.

## Objectif

L'objectif principal est de démontrer comment préparer des séries temporelles pour la modélisation, entraîner un modèle de régression RandomForest, et évaluer ses performances. Ce notebook peut servir de guide pour des analyses similaires sur d'autres séries temporelles.

## Prérequis

- Python 3.x
- Bibliothèques : pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn, sktime


---

""")
        # Ajout des images pour le modèle Random Forest
        st.image('intro/display_ser_morts.PNG')
        st.image('intro/display_serie_accs.PNG')
        st.markdown(f"\n\n")
        st.markdown(f"Serie morts")
        st.image('intro/serie_morts.PNG')
        st.image('intro/serie_morts_log.png')
        st.image('intro/serie_accs_log.png')
        st.image('intro/importance_accidents.png')
        st.image('intro/importance_morts.png')
    elif model_choice == "Modèle Dynamic Factor Model":
        st.write("Vous avez sélectionné le Modèle Dynamic Factor Model.")
        # Code pour le modèle Dynamic Factor Model (à ajouter)
    elif model_choice == "Modèle Exponential Smoothing":
        st.write("Vous avez sélectionné le Modèle Exponential Smoothing.")
        # Code pour le modèle Exponential Smoothing (à ajouter)
