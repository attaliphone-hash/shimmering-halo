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
    page_title="Comprendre Mes Aides CAF",
    page_icon="🏦",
    layout="wide"
)

# --- 3. GESTION DE LA CLÉ API GOOGLE ---
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    with st.sidebar:
        st.header("🔐 Configuration")
        api_key = st.text_input("Votre clé API Google :", type="password")
        if not api_key:
            st.warning("Veuillez entrer une clé API pour commencer.")
            st.stop()

# Configuration de Google Gemini
genai.configure(api_key=api_key)

# --- 4. LE CERVEAU (RAG & CHROMADB) ---
@st.cache_resource
def initialize_knowledge_base():
    try:
        # On garde le modèle d'embedding local par défaut (gratuit et efficace)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(
            name="caf_knowledge", 
            embedding_function=sentence_transformer_ef
        )
        
        files = [
            "caf_rsa_socle.txt", 
            "caf_prime_activite.txt", 
            "caf_aides_logement_apl.txt", 
            "caf_ressources_a_declarer.txt"
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
st.title("Comprendre Mes Aides (CAF) 🏦")
st.markdown("_L'assistant expert pour décrypter le RSA, la Prime d'Activité et les APL selon les barèmes 2025._")

if collection:
    st.success("✅ Assistant connecté aux barèmes CAF 2025 (Mode Google Gemini)")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je connais les règles 2025 pour le RSA, la Prime d'Activité et les APL. Une question ?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. LOGIQUE DE RÉPONSE (GEMINI) ---
if prompt := st.chat_input("Ex: Quel est le montant du RSA pour une personne seule ?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recherche RAG
    results = collection.query(query_texts=[prompt], n_results=3)
    context_text = "\n\n".join(results['documents'][0])

    # Prompt Système adapté pour Gemini
    system_prompt = f"""
    Tu es un assistant expert CAF (RSA, Prime d'Activité, APL).
    Règles :
    1. Base tes réponses UNIQUEMENT sur le CONTEXTE ci-dessous.
    2. Si tu ne sais pas, dis-le.
    3. Pas de morale.
    4. Utilise des LISTES à puces, pas de tableaux.
    5. Rappelle que c'est une estimation.
    
    CONTEXTE :
    {context_text}
    
    QUESTION UTILISATEUR :
    {prompt}
    """

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # Appel à Google Gemini (Flash est rapide et gratuit)
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(system_prompt, stream=True)
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Erreur Gemini : {e}")