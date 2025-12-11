import os
import sys

# --- 1. CORRECTIF POUR LE CLOUD (Obligatoire pour Linux/Streamlit Cloud) ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import streamlit as st
import google.generativeai as genai
import chromadb
import time

# --- 2. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Comprendre Ma Paie", page_icon="💡", layout="centered")
st.title("Comprendre Ma Paie 💡")
st.caption("L'assistant expert pour décrypter votre bulletin de salaire ©2025 Sylvain Attal")

# --- 3. SÉCURITÉ & CONNEXION ---
with st.sidebar:
    st.header("🔐 Configuration")
    api_key = None
    
    # Tentative de récupération automatique depuis les secrets
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ Clé API connectée (Mode Illimité)")
    except:
        pass

    # Champ manuel si les secrets ne fonctionnent pas en local
    if not api_key:
        api_key = st.text_input("Entrez votre clé API Google", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)

if not api_key:
    st.warning("⬅️ Veuillez configurer votre clé API pour commencer.")
    st.stop()

# --- 4. LE CERVEAU (Base de données vectorielle) ---
@st.cache_resource(show_spinner=False)
def charger_cerveau():
    client = chromadb.Client()
    nom_collection = "paie_expert_v4" # Nouvelle version pour forcer la mise à jour des données

    try:
        client.delete_collection(nom_collection)
    except:
        pass
    
    collection = client.create_collection(nom_collection)

    # Récupération des fichiers .txt
    tous_les_fichiers = [f for f in os.listdir('.') if f.endswith('.txt') and f != 'requirements.txt']
    
    if not tous_les_fichiers:
        return None

    docs_globaux = []
    ids_globaux = []
    compteur = 0
    
    # Lecture et découpage
    for fichier in tous_les_fichiers:
        with open(fichier, "r", encoding="utf-8") as f:
            contenu = f.read()
        
        taille_bloc = 1000
        chevauchement = 100
        
        for i in range(0, len(contenu), taille_bloc - chevauchement):
            morceau = contenu[i : i + taille_bloc]
            if len(morceau.strip()) > 10:
                docs_globaux.append(f"Source [{fichier}] : {morceau}")
                ids_globaux.append(f"doc_{compteur}")
                compteur += 1

    if not docs_globaux:
        return None

    # Vectorisation (Embedding)
    embeddings = []
    total = len(docs_globaux)
    barre = st.progress(0, text=f"Lecture des documents de référence ({total} extraits)...")
    
    # Modèle d'embedding (gratuit et performant)
    modele_embedding = "models/text-embedding-004"

    try:
        # Test rapide de connexion
        genai.embed_content(model=modele_embedding, content="Test", task_type="retrieval_document")
    except Exception as e:
        barre.empty()
        st.error(f"⛔️ Erreur de connexion API : {e}")
        return None

    for i, doc in enumerate(docs_globaux):
        try:
            res = genai.embed_content(model=modele_embedding, content=doc, task_type="retrieval_document")
            embeddings.append(res['embedding'])
            time.sleep(0.1) # Très rapide car quota illimité maintenant
        except:
            pass
        barre.progress(min((i + 1) / total, 1.0))
    
    barre.empty()
    
    if len(embeddings) > 0:
        collection.add(documents=docs_globaux, ids=ids_globaux, embeddings=embeddings)
        return collection
    return None

# --- 5. INTERFACE DE CHAT ---
with st.spinner("Initialisation de l'expert..."):
    db = charger_cerveau()

if db:
    st.success("✅ Assistant prêt à répondre !")
else:
    st.error("❌ Aucun document trouvé. Veuillez vérifier la présence des fichiers .txt.")

# Historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis connecté aux barèmes officiels 2025. Quelle ligne de votre bulletin de