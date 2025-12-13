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
    page_title="Comprendre Mon Chômage",
    page_icon="💼",
    layout="wide"
)

# --- 3. GESTION DE LA CLÉ API GOOGLE ---
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    # Fallback pour le local si pas de secrets
    st.warning("⚠️ Clé API non détectée dans les secrets. L'application ne pourra pas répondre.")
    st.stop()

# Configuration de Google Gemini
genai.configure(api_key=api_key)

# --- 4. LE CERVEAU (RAG & CHROMADB) ---
@st.cache_resource
def initialize_knowledge_base():
    try:
        # Modèle d'embedding local
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(
            name="chomage_knowledge", 
            embedding_function=sentence_transformer_ef
        )
        
        # Liste des fichiers à charger
        files = [
            "chomage_conditions_eligibilite.txt",
            "chomage_calcul_montant.txt",
            "chomage_duree_indemnisation.txt",
            "chomage_carence_et_differe.txt",
            "chomage_intermittents_spectacle.txt"
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
st.title("Comprendre Mon Chômage (France Travail) 💼")
st.markdown("_L'assistant expert pour comprendre vos droits (ARE), la réforme 2025 et le régime des intermittents._")

if collection:
    st.success("✅ Assistant connecté aux règles France Travail 2025")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je connais les règles d'indemnisation chômage, y compris pour les intermittents. Une question sur vos droits ?"}
    ]

# Affichage de l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. LOGIQUE DE RÉPONSE (GEMINI) ---
if prompt := st.chat_input("Ex: Combien de temps vais-je être indemnisé ?"):
    
    # 1. Affiche le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Recherche RAG (Contexte)
    results = collection.query(query_texts=[prompt], n_results=3)
    context_text = "\n\n".join(results['documents'][0])

    # 3. Prompt Système Expert
    system_prompt = f"""
    Tu es un assistant expert en assurance chômage (France Travail / ex-Pôle Emploi).
    Ta mission est d'aider l'utilisateur à comprendre ses droits (ARE) avec empathie et précision.

    RÈGLES IMPORTANTES :
    1. Base tes réponses UNIQUEMENT sur le CONTEXTE fourni ci-dessous.
    2. Si l'information n'est pas dans le contexte, dis que tu ne sais pas et conseille de contacter France Travail.
    3. Ne fais JAMAIS de morale (ex: sur la démission ou la recherche d'emploi). Reste factuel.
    4. PRÉSENTATION : Utilise systématiquement des LISTES à puces. Évite les tableaux.
    5. AVERTISSEMENT : Si la réponse contient des montants financiers (euros), précise bien que ce sont des estimations.
    6. INTERMITTENTS : Si la question concerne les artistes ou techniciens (annexes 8/10), base-toi priorité sur le fichier "chomage_intermittents_spectacle".

    CONTEXTE (Sources Officielles) :
    {context_text}
    
    QUESTION UTILISATEUR :
    {prompt}
    """

    # 4. Génération de la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # Utilisation du modèle rapide Gemini 2.0 Flash Exp
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

# --- 7. SYSTÈME DE FEEDBACK ---
# S'affiche en bas de page dès qu'il y a eu un échange
if len(st.session_state.messages) > 1:
    st.write("---")
    st.caption("Cette réponse vous a-t-elle aidé ?")
    
    # Création d'une clé unique pour chaque échange afin de ne pas mélanger les votes
    feedback_key = f"feedback_{len(st.session_state.messages)}"
    
    feedback = st.feedback("thumbs", key=feedback_key)

    if feedback is not None:
        if feedback == 1:
            st.toast("Merci pour votre retour positif ! 👍")
        elif feedback == 0:
            st.toast("Merci. Nous allons travailler à améliorer cette réponse. 👎")