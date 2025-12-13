import os
import sys

# --- 1. CORRECTIF CLOUD ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import streamlit as st
import google.generativeai as genai
import chromadb
import time

# --- 2. CONFIG PAGE ---
st.set_page_config(page_title="Comprendre Ma Paie", page_icon="💡", layout="centered")
st.title("Comprendre Ma Paie 💡")
st.caption("L'assistant expert pour décrypter votre bulletin de salaire ©2025 Sylvain Attal")

# --- 3. SÉCURITÉ ---
with st.sidebar:
    st.header("🔐 Configuration")
    api_key = None
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ Clé API connectée (Illimité)")
    except:
        pass

    if not api_key:
        api_key = st.text_input("Clé API Google", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)

if not api_key:
    st.warning("⬅️ Veuillez configurer votre clé API.")
    st.stop()

# --- 4. LE CERVEAU (GPS INTÉGRÉ) ---
@st.cache_resource(show_spinner=False)
def charger_cerveau():
    client = chromadb.Client()
    nom_collection = "paie_expert_v6_fix" # Nouvelle version

    try:
        client.delete_collection(nom_collection)
    except:
        pass
    
    collection = client.create_collection(nom_collection)

    # --- LE CORRECTIF GPS (Cherche DANS le dossier comprendre-paie) ---
    dossier_actuel = os.path.dirname(os.path.abspath(__file__))
    
    try:
        tous_les_fichiers = [f for f in os.listdir(dossier_actuel) if f.endswith('.txt')]
    except FileNotFoundError:
        return None

    if not tous_les_fichiers:
        return None

    docs_globaux = []
    ids_globaux = []
    compteur = 0
    
    # Lecture des fichiers
    for fichier in tous_les_fichiers:
        chemin_complet = os.path.join(dossier_actuel, fichier)
        with open(chemin_complet, "r", encoding="utf-8") as f:
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

    # Vectorisation
    embeddings = []
    total = len(docs_globaux)
    barre = st.progress(0, text=f"Analyse des règles de paie ({total} extraits)...")
    modele_embedding = "models/text-embedding-004"

    for i, doc in enumerate(docs_globaux):
        try:
            res = genai.embed_content(model=modele_embedding, content=doc, task_type="retrieval_document")
            embeddings.append(res['embedding'])
            time.sleep(0.05)
        except:
            pass
        barre.progress(min((i + 1) / total, 1.0))
    
    barre.empty()
    
    if len(embeddings) > 0:
        collection.add(documents=docs_globaux, ids=ids_globaux, embeddings=embeddings)
        return collection
    return None

# --- 5. CHAT ---
with st.spinner("Initialisation de l'expert paie..."):
    db = charger_cerveau()

if db:
    st.success("✅ Assistant Paie opérationnel !")
else:
    st.error("❌ Erreur : Fichiers .txt introuvables dans le dossier 'comprendre-paie'.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis l'expert Paie. Une ligne de votre bulletin vous intrigue ?"}]

for msg in st.session_state.messages:
    icone = "👔" if msg["role"] == "assistant" else "👤"
    st.chat_message(msg["role"], avatar=icone).write(msg["content"])

if question := st.chat_input("Votre question sur la paie..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user", avatar="👤").write(question)

    if db:
        try:
            # RAG
            q_vec = genai.embed_content(model="models/text-embedding-004", content=question, task_type="retrieval_query")
            res = db.query(query_embeddings=[q_vec['embedding']], n_results=5)
            
            if res['documents'] and res['documents'][0]:
                contexte = "\n\n".join(res['documents'][0])
                
                # Prompt Expert
                prompt = f"""Tu es un Expert Paie Pédagogue.
                Réponds à la question en utilisant les barèmes officiels ci-dessous.
                Sois précis sur les chiffres (Taux 2025) et clair dans l'explication.
                
                CONTEXTE :
                {contexte}
                
                QUESTION : {question}"""
                
                # Moteur Stable
                model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
                reponse = model.generate_content(prompt)
                
                st.chat_message("assistant", avatar="👔").write(reponse.text)
                st.session_state.messages.append({"role": "assistant", "content": reponse.text})
            else:
                st.warning("Je n'ai pas l'info dans mes fiches.")
        except Exception as e:
            st.error(f"Erreur : {e}")