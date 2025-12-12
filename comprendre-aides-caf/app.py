import os
import sys

# --- 1. LE FIX MAGIQUE POUR LE CLOUD (SQLite) ---
# Indispensable pour que ChromaDB fonctionne sur les serveurs Linux de Streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import openai

# --- 2. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Comprendre Mes Aides CAF",
    page_icon="🏦",
    layout="wide"
)

# --- 3. GESTION DE LA CLÉ API (SECRETS) ---
# On tente de récupérer la clé dans les "Secrets" de Streamlit (pour le Cloud)
# Sinon, on demande à l'utilisateur de la rentrer (pour le test local)
api_key = st.secrets.get("OPENAI_API_KEY")

if not api_key:
    with st.sidebar:
        st.header("🔐 Configuration")
        api_key = st.text_input("Votre clé API OpenAI :", type="password")
        if not api_key:
            st.warning("Veuillez entrer une clé API pour commencer.")
            st.stop()

# Configuration du client OpenAI
client = openai.Client(api_key=api_key)

# --- 4. LE CERVEAU (RAG & CHROMADB) ---
@st.cache_resource
def initialize_knowledge_base():
    """Charge les textes officiels CAF et prépare le moteur de recherche."""
    try:
        # On utilise le modèle d'embedding par défaut (léger et efficace)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # Création de la base de données en mémoire (volatile)
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(
            name="caf_knowledge", 
            embedding_function=sentence_transformer_ef
        )
        
        # Liste des documents à charger
        files = [
            "caf_rsa_socle.txt", 
            "caf_prime_activite.txt", 
            "caf_aides_logement_apl.txt", 
            "caf_ressources_a_declarer.txt"
        ]
        
        documents = []
        metadatas = []
        ids = []
        
        # Lecture et chargement des fichiers
        for idx, filename in enumerate(files):
            file_path = os.path.join(os.getcwd(), filename) # Chemin relatif
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append(content)
                    metadatas.append({"source": filename})
                    ids.append(f"doc_{idx}")
            else:
                st.error(f"Fichier introuvable : {filename}")

        # Indexation dans ChromaDB
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            
        return collection
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de la base de connaissances : {e}")
        return None

# Chargement unique au démarrage
collection = initialize_knowledge_base()

# --- 5. INTERFACE UTILISATEUR ---

# Titre et Sous-titre
st.title("Comprendre Mes Aides (CAF) 🏦")
st.markdown("""
_L'assistant expert pour décrypter le RSA, la Prime d'Activité et les APL selon les barèmes 2025._
""")

# Indicateur de statut
if collection:
    st.success("✅ Assistant connecté aux barèmes CAF 2025 (RSA, Prime d'Activité, APL)")
else:
    st.error("❌ Erreur : Impossible de charger les connaissances.")

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je connais les règles 2025 pour le RSA, la Prime d'Activité et les APL. Une question sur vos droits ou vos déclarations ?"}
    ]

# Affichage des messages précédents
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. LOGIQUE DE RÉPONSE (CHAT) ---
if prompt := st.chat_input("Ex: Quel est le montant du RSA pour une personne seule ?"):
    
    # 1. Afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Rechercher les infos pertinentes (RAG)
    results = collection.query(query_texts=[prompt], n_results=3)
    context_text = "\n\n".join(results['documents'][0])

    # 3. Construire le Prompt Système (La personnalité de l'IA)
    system_prompt = f"""
    Tu es un assistant expert en droits sociaux français (CAF), spécialisé dans le RSA, la Prime d'Activité et les APL.
    Ta mission est d'aider l'utilisateur à comprendre ses droits avec empathie, clarté et précision.

    RÈGLES IMPORTANTES :
    1. Base tes réponses UNIQUEMENT sur le CONTEXTE fourni ci-dessous (documents officiels 2025).
    2. Si l'information n'est pas dans le contexte, dis honnêtement que tu ne sais pas et conseille de contacter la CAF.
    3. Ne fais JAMAIS de morale. Reste factuel et bienveillant.
    4. PRÉSENTATION : Utilise systématiquement des LISTES à puces pour énumérer des conditions ou des montants. Évite les tableaux (difficiles à lire sur mobile).
    5. AVERTISSEMENT : Rappelle souvent que tu donnes des estimations basées sur les barèmes, mais que seule la simulation sur caf.fr fait foi.
    6. SUJET SENSIBLE : Si on parle d'épargne pour le RSA, explique bien la règle des 3% (ou 0.25% mensuel) mentionnée dans le contexte.

    CONTEXTE (Sources Officielles) :
    {context_text}
    """

    # 4. Appeler l'IA (OpenAI)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo", # Ou gpt-4o-mini si disponible
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                temperature=0.3 # Température basse pour éviter les hallucinations sur les chiffres
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Sauvegarder la réponse
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")