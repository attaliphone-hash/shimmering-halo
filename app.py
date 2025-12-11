import os
import sys

# --- CORRECTIF POUR LE CLOUD (Incontournable) ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import streamlit as st
import google.generativeai as genai
import chromadb
import time

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Comprendre Ma Paie", page_icon="💡")
st.title("Comprendre Ma Paie 💡")
st.caption("L'expert pour tout comprendre de votre bulletin de salaire ©2025-2026 Sylvain Attal")

# --- 1. SÉCURITÉ & CONNEXION ---
with st.sidebar:
    st.header("🔐 Connexion")
    
    api_key = None
    
    # Récupération de la clé depuis les secrets (Cloud) ou saisie manuelle
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ Clé API intégrée")
    except:
        pass

    if not api_key:
        api_key = st.text_input("Clé API Google", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)

if not api_key:
    st.warning("⬅️ Veuillez entrer une clé API pour commencer.")
    st.stop()

# --- 2. LE CERVEAU (Base de données vectorielle) ---
@st.cache_resource(show_spinner=False)
def charger_cerveau():
    client = chromadb.Client()
    try:
        client.delete_collection("paie_explainer_v3") # On change de version pour forcer le nettoyage
    except:
        pass
    collection = client.create_collection("paie_explainer_v3")

    # Recherche des fichiers explicatifs (.txt)
    tous_les_fichiers = [f for f in os.listdir('.') if f.endswith('.txt') and f != 'requirements.txt']
    
    if not tous_les_fichiers:
        return None

    docs_globaux = []
    ids_globaux = []
    compteur = 0
    
    # Lecture et découpage des fichiers
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
    barre = st.progress(0, text=f"Lecture des guides pédagogiques ({total} extraits)...")
    
    modele_embedding = "models/text-embedding-004"

    # Petit test de connexion avant de lancer la boucle
    try:
        genai.embed_content(model=modele_embedding, content="Test", task_type="retrieval_document")
    except Exception as e:
        barre.empty()
        st.error(f"⛔️ Erreur API (Embedding) : {e}")
        return None

    for i, doc in enumerate(docs_globaux):
        try:
            res = genai.embed_content(model=modele_embedding, content=doc, task_type="retrieval_document")
            embeddings.append(res['embedding'])
            time.sleep(0.5) # Petite pause pour ménager le quota
        except:
            break
        barre.progress(min((i + 1) / total, 1.0))
    
    barre.empty()
    
    if len(embeddings) > 0:
        collection.add(documents=docs_globaux, ids=ids_globaux, embeddings=embeddings)
        return collection
    return None

# --- 3. DÉMARRAGE DE L'INTERFACE ---
with st.spinner("Initialisation de l'assistant..."):
    db = charger_cerveau()

if db:
    st.success("✅ Assistant prêt à expliquer !")
else:
    st.error("❌ Impossible de charger les documents. Vérifiez qu'il y a bien des fichiers .txt dans le dossier.")

# Gestion de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis l'expert de votre bulletin de paie. Posez-moi une question (ex: 'C'est quoi la CSG ?')."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Zone de saisie
if question := st.chat_input("Votre question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    if db:
        try:
            # 1. Recherche des infos pertinentes
            q_vec = genai.embed_content(model="models/text-embedding-004", content=question, task_type="retrieval_query")
            res = db.query(query_embeddings=[q_vec['embedding']], n_results=5)
            
            if res['documents'] and res['documents'][0]:
                contexte = "\n\n".join(res['documents'][0])
                
                # 2. Construction du Prompt Pédagogique
                prompt = f"""Tu es un assistant pédagogique expert en paie.
                Ta mission : Expliquer des termes complexes à un salarié novice.
                
                Consignes :
                - Utilise un ton clair, bienveillant et rassurant.
                - Évite le jargon comptable inutile.
                - Base-toi UNIQUEMENT sur le contexte ci-dessous.
                
                CONTEXTE : {contexte}
                
                QUESTION : {question}"""
                
                # --- LE CHOIX DU MODÈLE (C'est ici que ça se joue) ---
                # On utilise la version LITE PREVIEW de votre liste pour le quota max
                model = genai.GenerativeModel('models/gemini-2.0-flash-lite-preview-02-05') 
                
                # Génération
                reponse = model.generate_content(prompt)
                
                # Affichage
                st.chat_message("assistant").write(reponse.text)
                st.session_state.messages.append({"role": "assistant", "content": reponse.text})
            else:
                st.warning("Je n'ai pas trouvé l'information dans mes documents de référence.")
        except Exception as e:
            st.error(f"Oups, une erreur technique : {e}")