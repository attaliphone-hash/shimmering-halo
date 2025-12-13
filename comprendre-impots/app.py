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
st.set_page_config(page_title="Comprendre Mes Impôts", page_icon="🏛️", layout="centered")
st.title("Comprendre Mes Impôts 🏛️")
st.caption("L'assistant expert pour décrypter votre avis d'imposition (Salariés & Indépendants) ©2025 Sylvain Attal")

# --- 3. SÉCURITÉ & CONNEXION ---
with st.sidebar:
    st.header("🔐 Configuration")
    api_key = None
    
    # Tentative de récupération automatique depuis les secrets
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ Clé API connectée (Illimité)")
    except:
        pass

    # Champ manuel si les secrets ne fonctionnent pas en local
    if not api_key:
        api_key = st.text_input("Entrez votre clé API Google", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)

    st.info("ℹ️ **Note :** Cet assistant couvre les régimes courants (Salariés, Retraités, Micro-Entrepreneurs). Pour les montages complexes (Sociétés IS, Holding), consultez un expert-comptable.")

if not api_key:
    st.warning("⬅️ Veuillez configurer votre clé API pour commencer.")
    st.stop()

# --- 4. LE CERVEAU (Base de données vectorielle) ---
@st.cache_resource(show_spinner=False)
def charger_cerveau():
    client = chromadb.Client()
    # On change de version pour forcer l'intégration du fichier Micro-Entrepreneur
    nom_collection = "impots_expert_v5_final" 

    try:
        client.delete_collection(nom_collection)
    except:
        pass
    
    collection = client.create_collection(nom_collection)

    # --- LE CORRECTIF GPS ---
    # Permet de trouver les fichiers .txt même si l'app est lancée depuis ailleurs
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
    
    # Lecture et découpage
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

    # Vectorisation (Embedding)
    embeddings = []
    total = len(docs_globaux)
    barre = st.progress(0, text=f"Analyse des règles fiscales ({total} extraits)...")
    
    modele_embedding = "models/text-embedding-004"

    for i, doc in enumerate(docs_globaux):
        try:
            res = genai.embed_content(model=modele_embedding, content=doc, task_type="retrieval_document")
            embeddings.append(res['embedding'])
            time.sleep(0.05) # Rapide car quota illimité
        except:
            pass
        barre.progress(min((i + 1) / total, 1.0))
    
    barre.empty()
    
    if len(embeddings) > 0:
        collection.add(documents=docs_globaux, ids=ids_globaux, embeddings=embeddings)
        return collection
    return None

# --- 5. INTERFACE DE CHAT ---
with st.spinner("Initialisation de l'expert fiscal..."):
    db = charger_cerveau()

if db:
    st.success("✅ Assistant prêt à répondre (Salariés & Micro-Entrepreneurs) !")
else:
    st.error("❌ Aucun document trouvé. Vérifiez la présence des fichiers .txt dans le dossier 'comprendre-impots'.")

# Historique de conversation
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis à jour des barèmes 2025. Une question sur votre avis ou votre statut d'indépendant ?"}]

for msg in st.session_state.messages:
    icone = "🏛️" if msg["role"] == "assistant" else "👤"
    st.chat_message(msg["role"], avatar=icone).write(msg["content"])

# Zone de saisie
if question := st.chat_input("Votre question (ex: Je suis auto-entrepreneur, comment déclarer ?..."):
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
                prompt = f"""Tu es un Expert Fiscaliste Pédagogue (Assistant DGFiP).
                Ta mission : Aider le contribuable à comprendre son impôt 2025 (sur revenus 2024).

                RÈGLES D'OR :
                1. Base tes réponses UNIQUEMENT sur le contexte fourni.
                2. Si on te demande un calcul, utilise le barème 2025 du contexte.
                3. Pour les Micro-Entrepreneurs : sois très vigilant à distinguer le régime "Classique" (Abattement forfaitaire) du "Versement Libératoire".
                4. Sois clair, pédagogique et rassurant.
                5. Rappelle toujours que tu donnes une estimation informative.
                
                CONTEXTE DOCUMENTAIRE :
                {contexte}
                
                QUESTION DU CONTRIBUABLE : {question}"""
                
                # --- LE MOTEUR ---
                model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
                
                reponse = model.generate_content(prompt)
                
                st.chat_message("assistant", avatar="🏛️").write(reponse.text)
                st.session_state.messages.append({"role": "assistant", "content": reponse.text})
            else:
                st.warning("Je n'ai pas trouvé cette information précise dans ma base documentaire (Fichiers textes).")
        
        # C'est cette partie qui manquait probablement :
        except Exception as e:
            st.error(f"Une erreur technique est survenue : {e}")