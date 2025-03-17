import streamlit as st
import pandas as pd
import requests
import spacy
from sentence_transformers import SentenceTransformer, util
from geopy.geocoders import Nominatim
import plotly.graph_objects as go
import PyPDF2
import docx
import io
import base64
import os
os.system("python -m spacy download de_core_news_sm")

# Load NLP models
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("de_core_news_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

st.title("🚀 Headhunter AI – CV & Job Matching")

st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Menü", ["Startseite", "CV Hochladen", "Vakanz Hochladen", "Matching-Ergebnisse"])

if menu == "Startseite":
    st.header("Willkommen bei Headhunter AI")
    st.write("Laden Sie Lebensläufe und Jobbeschreibungen hoch, um automatisierte Analysen und Matching-Ergebnisse zu erhalten.")

elif menu == "CV Hochladen":
    uploaded_files = st.file_uploader("Lebensläufe hochladen", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if st.button("CVs analysieren"):
        for uploaded_file in uploaded_files:
            st.write("Datei analysiert:", uploaded_file.name)

            if uploaded_file.name.endswith('.pdf'):
                reader = PyPDF2.PdfReader(uploaded_file)
                text = "".join([page.extract_text() for page in reader.pages])
            elif uploaded_file.name.endswith('.docx'):
                doc = docx.Document(uploaded_file)
                text = "\n".join([p.text for p in doc.paragraphs])
            else:
                text = uploaded_file.getvalue().decode("utf-8")

            summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            st.subheader("Zusammenfassung:")
            st.write(summary)

            doc = nlp(text)
            adresse = None
            for ent in doc.ents:
                if ent.label_ == "LOC":
                    adresse = ent.text
                    break

            if adresse:
                geocode = requests.get(f'https://nominatim.openstreetmap.org/search?format=json&q={adresse}').json()
                if geocode:
                    lat, lon = float(geocode[0]['lat']), float(geocode[0]['lon'])
                    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))

elif menu == "Matching-Ergebnisse":
    st.header("Kandidaten Matching")
    uploaded_job = st.file_uploader("Vakanz hochladen", type=["pdf", "docx", "txt", "png"])
    uploaded_cvs = st.file_uploader("Lebensläufe hochladen", accept_multiple_files=True)

    if st.button("Matching starten"):
        if uploaded_vakanz:
            if uploaded_vakanz.name.endswith('.pdf'):
                reader = PyPDF2.PdfReader(uploaded_vakanz)
                vakanz_text = "".join([page.extract_text() for page in reader.pages])
            summary_vakanz = summarizer(vakanz_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

            vakanz_emb = sentence_model.encode(vakanz_text)

            results = []
            for cv in uploaded_cvs:
                if cv.name.endswith('.pdf'):
                    reader = PyPDF2.PdfReader(cv)
                    cv_text = "".join([page.extract_text() for page in reader.pages])

                cv_embedding = sentence_model.encode(cv_text)
                score = util.cos_sim(cv_embedding, vakanz_embedding)[0][0].item()
                category = "gut" if score >= 0.8 else "mittel" if score >= 0.6 else "schlecht"
                results.append({"Kandidat": cv.name, "Score": round(score.item()*100,2), "Kategorie": category})

            results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
            st.table(results_df)

if st.sidebar.button("Alle Filter löschen"):
    st.experimental_rerun()
