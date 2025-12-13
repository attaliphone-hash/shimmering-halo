import os
import sys

# --- 1. LE FIX MAGIQUE POUR LE CLOUD (SQLite) ---
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# --- 2. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Mon Logement & Mes Droits",
    page_icon="🏠",
    layout="wide"
)

# --- 3. GESTION DE LA CLÉ API ---
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.warning("⚠️ Clé API non détectée. Configurez les secrets (.streamlit/secrets.toml).")
    st.stop()

genai.configure(api_key=api_key)

# --- 4. LE CERVEAU (RAG) ---
@st.cache_resource
def initialize_knowledge_base():
    try:
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(
            name="logement_knowledge", 
            embedding_function=sentence_transformer_ef
        )
        
        # Liste des fichiers Logement
        files = [
            "logement_loi_89_generale.txt",
            "logement_qui_paye_quoi.txt",
            "logement_depot_garantie.txt",
            "logement_preavis_depart.txt",
            "logement_expulsion_et_impayes.txt",
            "logement_encadrement_loyers_2025.txt"
        ]
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, filename in enumerate(files):
            file_path = os.path.join(os.getcwd(), filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    metadatas.append({"source": filename})
                    ids.append(f"doc_{idx}")
        
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            
        return collection
    except Exception as e:
        st.error(f"Erreur init base : {e}")
        return None

collection = initialize_knowledge_base()

# --- 5. INTERFACE ---
st.title("Mon Logement (Locataire & Proprio) 🏠")
st.markdown("_L'assistant expert en droit du logement (Loi de 89, loyers, travaux, expulsion)._")

if collection:
    st.success("✅ Assistant connecté aux lois Logement 2025")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Un problème de caution, de travaux ou de loyer ? Je peux vous aider à comprendre vos droits."}
    ]

# Affichage historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. LOGIQUE DE RÉPONSE ---
if prompt := st.chat_input("Ex: Mon propriétaire ne rend pas la caution, que faire ?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recherche RAG
    results = collection.query(query_texts=[prompt], n_results=3)
    context_text = "\n\n".join(results['documents'][0])

    # Prompt Système Spécifique Logement
    system_prompt = f"""
    Tu es un juriste expert en droit du logement français.
    Ta mission est d'informer l'utilisateur sur ses droits et devoirs (Locataire ou Propriétaire).

    RÈGLES IMPORTANTES :
    1. Base tes réponses UNIQUEMENT sur le CONTEXTE fourni.
    2. Cite systématiquement les sources (ex: "Selon la Loi de 89...", "D'après le décret de 87...").
    3. Si la question concerne un conflit (caution, travaux), propose une approche amiable d'abord, puis les recours légaux.
    4. Ne donne JAMAIS de conseil illégal (ex: "arrêtez de payer le loyer").
    5. Sois clair et structuré (listes à puces).

    CONTEXTE JURIDIQUE :
    {context_text}
    
    QUESTION :
    {prompt}
    """

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(system_prompt, stream=True)
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")

# --- 7. FEEDBACK ---
if len(st.session_state.messages) > 1:
    st.write("---")
    st.caption("Cette réponse vous a-t-elle aidé ?")
    feedback_key = f"feedback_{len(st.session_state.messages)}"
    feedback = st.feedback("thumbs", key=feedback_key)
    if feedback is not None:
        if feedback == 1:
            st.toast("Merci ! 👍")
        elif feedback == 0:
            st.toast("Merci, nous allons améliorer ça. 👎")