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

import streamlit as st
import google.generativeai as genai
import chromadb
import os
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
        st.error(f"⛔️ Erreur API : {e}")
        return None

    for i, doc in enumerate(docs_globaux):
        try:
            res = genai.embed_content(model=modele_embedding, content=doc, task_type="retrieval_document")
            embeddings.append(res['embedding'])
            time.sleep(1.0)
        except:
            break
        barre.progress(min((i + 1) / total, 1.0))
    
    barre.empty()
    
    if len(embeddings) > 0:
        collection.add(documents=docs_globaux, ids=ids_globaux, embeddings=embeddings)
        return collection
    return None

# --- 3. DÉMARRAGE ---
with st.spinner("Préparation de l'assistant..."):
    db = charger_cerveau()

if db:
    st.success("✅ Assistant prêt à expliquer !")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Une ligne de votre bulletin de paie vous semble obscure ? Donnez-moi son nom, je vous l'explique simplement. 😊"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("Ex: C'est quoi la CSG ?"):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        try:
            q_vec = genai.embed_content(model="models/text-embedding-004", content=question, task_type="retrieval_query")
            res = db.query(query_embeddings=[q_vec['embedding']], n_results=5)
            
            if res['documents'] and res['documents'][0]:
                contexte = "\n\n".join(res['documents'][0])
                
                # --- PROMPT PÉDAGOGIQUE ---
                prompt = f"""Tu es un assistant pédagogique expert en paie, bienveillant et clair.
                Ta mission : Expliquer des termes complexes de paie à un salarié novice.
                Consignes :
                1. Utilise des mots simples, évite le jargon technique froid.
                2. Sois rassurant.
                3. Base-toi UNIQUEMENT sur le contexte fourni ci-dessous.
                
                CONTEXTE : {contexte}
                
                QUESTION DU SALARIÉ : {question}"""
                
                model = genai.GenerativeModel('gemini-1.5-flash-002') 
                reponse = model.generate_content(prompt)
                
                st.chat_message("assistant").write(reponse.text)
                st.session_state.messages.append({"role": "assistant", "content": reponse.text})
            else:
                st.warning("Je n'ai pas trouvé d'explication dans mes guides pour ce terme.")
        except Exception as e:
            st.error(f"Erreur : {e}")