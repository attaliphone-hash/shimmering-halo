import os
import sys

# Correctif Spécial pour le Cloud (Linux)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # On est sur Mac/Windows, on ne fait rien, tout va bien.

import streamlit as st
import google.generativeai as genai
import chromadb
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Comprendre Ma Paie", page_icon="💡")
st.title("Comprendre Ma Paie 💡")
st.caption("L'expert pour tout comprendre de votre bulletin de salaire ©2025-2026 Sylvain Attal")

# --- 1. SÉCURITÉ & CONNEXION ---
with st.sidebar:
    st.header("🔐 Connexion")
    
    api_key = None
    
    # TENTATIVE D'OUVERTURE DU COFFRE (SECRETS)
    try:
        # On vérifie si le coffre existe sans faire planter l'app
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ Clé API intégrée")
    except FileNotFoundError:
        pass # Pas de fichier secrets sur le Mac, on ignore
    except Exception:
        pass # Autre erreur de coffre, on ignore

    # Si pas de clé trouvée (ou coffre absent), on la demande
    if not api_key:
        api_key = st.text_input("Clé API Google", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
        
        # --- 🕵️‍♂️ DÉBUT DU CODE ESPION ---
        st.write("---")
        st.warning("🕵️‍♂️ MODE DIAGNOSTIC ACTIVÉ")
        st.write("**Voici la liste EXACTE des modèles disponibles pour votre clé :**")
        try:
            liste_modeles = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    st.code(m.name) # Affiche le nom exact
                    liste_modeles.append(m.name)
            
            if not liste_modeles:
                st.error("Aucun modèle trouvé ! Vérifiez votre clé API.")
            
            st.stop() # 🛑 ON ARRÊTE TOUT ICI POUR LIRE LA LISTE
        except Exception as e:
            st.error(f"Erreur lors du scan : {e}")
            st.stop()
        # --- 🕵️‍♂️ FIN DU CODE ESPION ---

if not api_key:
    st.warning("⬅️ Veuillez entrer une clé API pour commencer.")
    st.stop()

# --- 2. FONCTION D'INDEXATION ---
@st.cache_resource(show_spinner=False)
def charger_cerveau():
    client = chromadb.Client()
    try:
        client.delete_collection("paie_explainer_v2")
    except:
        pass
    collection = client.create_collection("paie_explainer_v2")

    # Recherche de tous les fichiers explicatifs
    tous_les_fichiers = [f for f in os.listdir('.') if f.endswith('.txt') and f != 'requirements.txt']
    
    if not tous_les_fichiers:
        st.error("❌ Je ne trouve pas de documents explicatifs (.txt).")
        return None

    docs_globaux = []
    ids_globaux = []
    compteur = 0
    
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
        st.error("❌ Les fichiers sont vides.")
        return None

    embeddings = []
    total = len(docs_globaux)
    barre = st.progress(0, text=f"Lecture des guides pédagogiques ({total} extraits)...")
    
    modele_embedding = "models/text-embedding-004"

    try:
        genai.embed_content(model=modele_embedding, content="Test", task_type="retrieval_document")
    except Exception as e:
        barre.empty()