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
    nom_collection = "paie_expert_v5" # Nouvelle version pour forcer la lecture de tous les fichiers

    try:
        client.delete_collection(nom_collection)
    except:
        pass
    
    collection = client.create_collection(nom_collection)

    # Récupération de TOUS les fichiers .txt (Taux + Explications)
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
            time.sleep(0.05) # Très rapide car quota illimité maintenant
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
    # La phrase ci-dessous est bien sur une seule ligne pour éviter le bug
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis connecté aux barèmes officiels 2025. Quelle ligne de votre bulletin de paie voulez-vous comprendre ?"}]

for msg in st.session_state.messages:
    # Avatar personnalisé : Cravate pour l'assistant, Bonhomme pour l'utilisateur
    icone = "👔" if msg["role"] == "assistant" else "👤"
    st.chat_message(msg["role"], avatar=icone).write(msg["content"])

# Zone de saisie
if question := st.chat_input("Votre question (ex: C'est quoi la CSG ? Mon brut est de 3000€...)"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user", avatar="👤").write(question)

    if db:
        try:
            # 1. Recherche RAG
            q_vec = genai.embed_content(model="models/text-embedding-004", content=question, task_type="retrieval_query")
            res = db.query(query_embeddings=[q_vec['embedding']], n_results=5)
            
            if res['documents'] and res['documents'][0]:
                contexte = "\n\n".join(res['documents'][0])
                
                # 2. Prompt Expert & Pédagogue
                prompt = f"""Tu es un Expert Paie et Pédagogue.
                Ta mission : Répondre à la question du salarié en utilisant les barèmes officiels fournis ci-dessous.
                
                Règles d'or :
                - Ton : Bienveillant, clair, rassurant.
                - Précision : Utilise les chiffres du contexte (Taux 2025).
                - Si on te demande un calcul, fais-le étape par étape.
                - Cite tes sources implicitement ("Selon les barèmes officiels...").
                
                DOCUMENTS DE RÉFÉRENCE (CONTEXTE) :
                {contexte}
                
                QUESTION DU SALARIÉ : {question}"""
                
                # --- LE MOTEUR (Maintenant débridé grâce à la facturation) ---
                # On utilise le modèle 2.0 Flash standard
                model = genai.GenerativeModel('models/gemini-2.0-flash')
                
                reponse = model.generate_content(prompt)
                
                st.chat_message("assistant", avatar="👔").write(reponse.text)
                st.session_state.messages.append({"role": "assistant", "content": reponse.text})
            else:
                st.warning("Je n'ai pas trouvé cette information dans mes documents de référence.")
        except Exception as e:
            st.error(f"Une erreur technique est survenue : {e}")