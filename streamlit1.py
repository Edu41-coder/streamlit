import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load and convert image
img = Image.open('intro/_77f47b66-9794-484f-807e-56df65a48d68.jfif')
img.save('intro/_77f47b66-9794-484f-807e-56df65a48d68.jpg', 'JPEG')

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

if page == "Introduction":
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