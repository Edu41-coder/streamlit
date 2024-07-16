import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

img = Image.open('_77f47b66-9794-484f-807e-56df65a48d68.jfif')

# Convertir l'image en JPG
img.save('_77f47b66-9794-484f-807e-56df65a48d68.jpg', 'JPEG')

# Maintenant, vous pouvez utiliser l'image convertie avec Streamlit
st.image(img, use_column_width=True)


st.title("Projet de prédiction de la gravité des accidents en France")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

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

if page == pages[0] : 
  st.write("### Introduction")