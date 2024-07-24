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
        
        # Ajout du texte avant la première image
        st.markdown("""
        # Analyse des Séries Temporelles et Modélisation avec RandomForest

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
        st.markdown("### Serie morts")
        st.image('intro/serie_morts.PNG')
        st.markdown("### Serie morts log")
        st.image('intro/serie_morts_log.png')
        st.markdown("### Serie accidents log")
        st.image('intro/serie_accs_log.png')
        st.markdown("### Résultats serie morts")
        st.markdown("""
        Results for morts:
        {'Model Name': 'Random Forest Regressor', 'Model': RandomForestRegressor(), 'RMSE': 0.9896255226157039, 'MAPE': 1.6330739234680127, 'MASE': 0.4701224490257277, 'yhat': array([ 0.32542165, -0.71393091,  0.91215215, -0.82963514,  0.4194221 ,
                0.05302504, -0.75344515, -0.16856281, -0.56562013,  0.21426137,
               -0.20066631,  0.39048698]), 'resid': array([ 0.45679673, -0.18485571,  0.86876831,  0.10074539, -0.91839687,
                0.85454099,  2.78005394,  0.03925516,  0.73152128, -0.19090626,
                0.06091821,  0.93024063]), 'actual': array([ 0.78221838, -0.89878661,  1.78092046, -0.72888976, -0.49897477,
                0.90756602,  2.02660878, -0.12930764,  0.16590115,  0.02335511,
               -0.1397481 ,  1.32072761])}
        """)
        st.markdown("""
        **Analyse des Métriques de Performance**
        1. **RMSE (Root Mean Squared Error)** :
           - **Valeur : 0.9896255226157039**
           - Le RMSE mesure l'écart type des résidus (erreurs de prédiction). Une valeur plus faible indique un meilleur ajustement du modèle aux données. Ici, un RMSE de 0.99 est relativement faible, ce qui suggère que le modèle a une bonne précision.
        2. **MAPE (Mean Absolute Percentage Error)** :
           - **Valeur : 1.6330739234680127**
           - Le MAPE mesure l'erreur moyenne en pourcentage entre les valeurs prédites et les valeurs réelles. Un MAPE de 1.63% indique que, en moyenne, les prédictions du modèle sont à 1.63% près des valeurs réelles.
        3. **MASE (Mean Absolute Scaled Error)** :
           - **Valeur : 0.4701224490257277**
           - Le MASE est une mesure de l'erreur de prévision qui est indépendante de l'échelle des données. Une valeur de MASE inférieure à 1 indique que le modèle est meilleur que la moyenne des erreurs absolues d'un modèle de référence simple (comme la moyenne historique).
        """)
        st.markdown("""
        **Conclusion**
        Le modèle Random Forest Regressor montre une bonne performance avec un RMSE relativement faible et un MAPE de 1.63%, ce qui indique que les prédictions sont assez précises. Le MASE inférieur à 1 confirme que le modèle est performant par rapport à un modèle de référence simple. Les résidus montrent les écarts entre les valeurs réelles et prédites, et bien que certains écarts soient notables, la performance globale reste satisfaisante.
        Ces résultats suggèrent que le modèle est capable de capturer les tendances et les variations des séries temporelles des décès (serie_morts) de manière efficace.
        """)
        st.markdown("### Résultats serie accidents")
        st.markdown("""
        Results for accs:
        {'Model Name': 'Random Forest Regressor', 'Model': RandomForestRegressor(), 'RMSE': 0.7360697791883707, 'MAPE': 1.929022923086171, 'MASE': 0.4242266277968524, 'yhat': array([-0.32188171,  0.34778839, -0.00321162,  0.57383633, -0.86493909,
               -0.04776909, -0.11343434,  0.27701956, -0.6485534 ,  0.03669212,
               -0.57403235,  0.66788438]), 'resid': array([ 0.62141432, -0.08109798,  0.05140698,  0.12887384,  0.80512664,
               -0.23506058,  1.49880924,  1.04763099,  1.31944887,  0.38019014,
               -0.39342825, -0.04339357]), 'actual': array([ 0.29953261,  0.2666904 ,  0.04819536,  0.70271017, -0.05981246,
               -0.28282967,  1.3853749 ,  1.32465056,  0.67089546,  0.41688225,
               -0.9674606 ,  0.62449082])}
        """)
        st.markdown("""
        **Analyse des Métriques de Performance**
        1. **RMSE (Root Mean Squared Error)** :
           - **Valeur : 0.7360697791883707**
           - Le RMSE mesure l'écart type des résidus (erreurs de prédiction). Une valeur plus faible indique un meilleur ajustement du modèle aux données. Ici, un RMSE de 0.74 est relativement faible, ce qui suggère que le modèle a une bonne précision.
        2. **MAPE (Mean Absolute Percentage Error)** :
           - **Valeur : 1.929022923086171**
           - Le MAPE mesure l'erreur moyenne en pourcentage entre les valeurs prédites et les valeurs réelles. Un MAPE de 1.93% indique que, en moyenne, les prédictions du modèle sont à 1.93% près des valeurs réelles.
        3. **MASE (Mean Absolute Scaled Error)** :
           - **Valeur : 0.4242266277968524**
           - Le MASE est une mesure de l'erreur de prévision qui est indépendante de l'échelle des données. Une valeur de MASE inférieure à 1 indique que le modèle est meilleur que la moyenne des erreurs absolues d'un modèle de référence simple (comme la moyenne historique).
        """)
        st.markdown("""
        **Conclusion**
        Le modèle Random Forest Regressor montre une bonne performance avec un RMSE relativement faible et un MAPE de 1.93%, ce qui indique que les prédictions sont assez précises. Le MASE inférieur à 1 confirme que le modèle est performant par rapport à un modèle de référence simple. Les résidus montrent les écarts entre les valeurs réelles et prédites, et bien que certains écarts soient notables, la performance globale reste satisfaisante.
        Ces résultats suggèrent que le modèle est capable de capturer les tendances et les variations des séries temporelles des accidents (serie_accs) de manière efficace.
        """)
        st.image('intro/importance_accidents.png')
        st.markdown("""
        **Conclusion**
        L'analyse des importances des variables montre que les valeurs de la série temporelle 12 mois avant (x_12) et 1 mois avant (x_1) la période actuelle sont les plus critiques pour les prédictions du modèle RandomForestRegressor sur les données de décès. Cela suggère une forte saisonnalité annuelle ainsi qu'une dépendance à court terme dans les données. Les autres variables, bien qu'importantes, ont un impact moindre sur les prédictions.
        """)
        st.image('intro/importance_morts.png')
        st.markdown("""
        **Conclusion**
        L'analyse des importances des variables montre que la valeur de la série temporelle un mois avant la période actuelle (x_1) est la plus critique pour les prédictions du modèle RandomForestRegressor sur les données d'accidents. La valeur de la série temporelle 12 mois avant (x_12) est également très significative, indiquant une forte saisonnalité annuelle. Les autres variables, bien qu'importantes, ont un impact moindre en comparaison.
        """)
    elif model_choice == "Modèle Dynamic Factor Model":
        st.write("Vous avez sélectionné le Modèle Dynamic Factor Model.")
        # Code pour le modèle Dynamic Factor Model (à ajouter)
    elif model_choice == "Modèle Exponential Smoothing":
        st.write("Vous avez sélectionné le Modèle Exponential Smoothing.")
        # Code pour le modèle Exponential Smoothing (à ajouter)
