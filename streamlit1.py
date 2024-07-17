import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import nbformat
from nbconvert import HTMLExporter
import imgkit

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
    
    # Example: Upload a file
    uploaded_file = st.file_uploader("Choisissez un fichier", type=["csv", "xlsx", "json", "py", "jpg", "jpeg", "png", "ipynb"])
    data = None  # Initialize data to None
    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.write(file_details)
        
        # Ensure the uploads directory exists
        uploads_dir = "C:\\uploads"
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        # Save the uploaded file to disk
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File saved to {file_path}")
        
        # Process CSV file
        if uploaded_file.type == "text/csv":
            data = pd.read_csv(file_path)
            st.write("Aperçu des données CSV:")
            st.write(data.head())
        
        # Process Excel file
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(file_path)
            st.write("Aperçu des données Excel:")
            st.write(data.head())
        
        # Process JSON file
        elif uploaded_file.type == "application/json":
            data = pd.read_json(file_path)
            st.write("Aperçu des données JSON:")
            st.write(data.head())
        
        # Process Python file
        elif uploaded_file.type == "text/x-python":
            with open(file_path, "r") as f:
                code = f.read()
            st.write("Contenu du fichier Python:")
            st.code(code, language='python')
            
            # Execute the Python code safely
            try:
                exec(code, {'st': st, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns})
            except Exception as e:
                st.error(f"Erreur lors de l'exécution du code Python: {e}")
        
        # Process Image file
        elif uploaded_file.type in ["image/jpeg", "image/png"]:
            img = Image.open(file_path)
            st.image(img, caption=uploaded_file.name, use_column_width=True)
        
        # Process Jupyter Notebook file
        elif uploaded_file.type == "application/x-ipynb+json":
            with open(file_path, "r") as f:
                notebook_content = f.read()
            st.write("Contenu du fichier Jupyter Notebook:")
            st.code(notebook_content, language='json')
            
            # Add options for conversion
            conversion_option = st.selectbox(
                "Choisissez une option de conversion",
                ("Afficher en HTML", "Convertir en image")
            )
            
            if conversion_option == "Afficher en HTML":
                # Convert Jupyter Notebook to HTML
                try:
                    notebook = nbformat.reads(notebook_content, as_version=4)
                    html_exporter = HTMLExporter()
                    html_data, _ = html_exporter.from_notebook_node(notebook)
                    
                    # Display HTML
                    st.write("HTML du fichier Jupyter Notebook:")
                    st.components.v1.html(html_data, height=600, scrolling=True)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la conversion du fichier Jupyter Notebook en HTML: {e}")
            
            elif conversion_option == "Convertir en image":
                # Convert Jupyter Notebook to HTML and then to image
                try:
                    notebook = nbformat.reads(notebook_content, as_version=4)
                    html_exporter = HTMLExporter()
                    html_data, _ = html_exporter.from_notebook_node(notebook)
                    
                    # Save HTML to a temporary file
                    with open("temp_notebook.html", "w", encoding="utf-8") as f:
                        f.write(html_data)
                    
                    # Convert HTML to image
                    imgkit.from_file("temp_notebook.html", "notebook_image.jpg")
                    
                    # Display the image
                    img = Image.open("notebook_image.jpg")
                    st.image(img, caption="Notebook Image", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la conversion du fichier Jupyter Notebook en image: {e}")
        
        # Example: Plotting time series data
        if data is not None:
            st.write("### Visualisation des donn��es temporelles")
            if 'date' in data.columns and 'value' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
                st.line_chart(data['value'])
            else:
                st.error("Le fichier doit contenir des colonnes 'date' et 'value'.")

    # Example: Add more features as needed
    # You can add more widgets, visualizations, and model training code here
