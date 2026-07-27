import streamlit as st
import requests
import chromadb
from sentence_transformers import SentenceTransformer
 
st.set_page_config(
    page_title="Assistant RH Intelligent",
    page_icon="🤖",
    layout="wide"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #f3f0ff 0%, #fdf4ff 50%, #fff7ed 100%); }
 
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2d1b69 0%, #4c1d95 100%);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #ddd6fe !important; font-size: 0.85rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: white !important; }
[data-testid="stSidebar"] hr { border-color: #5b21b6 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    width: 100%; box-shadow: 0 4px 12px rgba(249,115,22,0.3) !important;
}
 
/* Messages chat */
.msg-user {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%;
    box-shadow: 0 4px 12px rgba(124,58,237,0.25);
    font-size: 0.92rem;
    line-height: 1.6;
}
.msg-bot {
    background: white;
    color: #1e1b4b;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 8px 0;
    box-shadow: 0 2px 8px rgba(109,40,217,0.1);
    border-left: 3px solid #7c3aed;
    font-size: 0.92rem;
    line-height: 1.6;
}
.msg-avatar-bot {
    font-size: 1.4rem;
    margin-right: 8px;
    vertical-align: middle;
}
 
/* Header */
.chat-header {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(109,40,217,0.08);
    border: 1px solid #ede9fe;
    display: flex;
    align-items: center;
    gap: 16px;
}
 
/* Input */
.stTextInput > div > div > input {
    border-radius: 25px !important;
    border: 2px solid #ede9fe !important;
    padding: 12px 20px !important;
    font-size: 0.92rem !important;
    background: white !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;
}
 
h1 { color: #2d1b69 !important; font-weight: 800 !important; }
h2, h3 { color: #4c1d95 !important; }
p, li { color: #374151 !important; }
</style>
""", unsafe_allow_html=True)
 
 
# ── INITIALISATION ──────────────────────────────────────────
@st.cache_resource
def charger_systeme():
    """Charge le modèle et la base de données une seule fois"""
    modele = SentenceTransformer('all-MiniLM-L6-v2')
    client_db = chromadb.Client()
 
    collection = client_db.create_collection("rh_docs")
 
    # Base de connaissances RH — à personnaliser selon le client !
    documents = [
        "Le poste de Data Scientist requiert minimum 2 ans d'expérience en Python et Machine Learning.",
        "La politique salariale pour un Data Scientist junior est entre 38 000€ et 45 000€ par an.",
        "Les candidats sans diplôme peuvent être acceptés si leur portfolio GitHub est solide.",
        "Un entretien technique de 2 heures est obligatoire pour tout poste tech avant embauche.",
        "La période d'essai est de 3 mois renouvelable une fois pour les cadres.",
        "Le télétravail est possible jusqu'à 3 jours par semaine après la période d'essai.",
        "Les congés payés sont de 25 jours ouvrés par an, conformément à la convention collective.",
        "Le poste de Chef de Projet requiert 5 ans d'expérience minimum et une certification PMP.",
        "Les entretiens annuels d'évaluation ont lieu en janvier de chaque année.",
        "La mutuelle entreprise couvre 60% des cotisations santé pour le salarié et sa famille.",
        "Un plan de formation est proposé à chaque salarié avec un budget de 1 500€ par an.",
        "Le recrutement pour les postes seniors nécessite l'accord de la direction générale.",
    ]
 
    embeddings = modele.encode(documents).tolist()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    return modele, collection
 
modele_embed, collection = charger_systeme()
 
 
# ── FONCTION RAG ─────────────────────────────────────────────
def repondre(question, cle_api, historique):
    # 1. Chercher les documents pertinents
    q_embed = modele_embed.encode([question]).tolist()
    resultats = collection.query(query_embeddings=q_embed, n_results=3)
    contexte = "\n- ".join(resultats['documents'][0])
 
    # 2. Construire l'historique pour le contexte
    messages = [
        {"role": "system", "content": f"""Tu es Alex, un assistant RH intelligent et bienveillant.
        Tu aides les équipes RH à prendre de meilleures décisions.
        Tu réponds UNIQUEMENT en te basant sur les documents fournis.
        Si une information est absente, dis-le clairement et poliment.
        Tu réponds toujours en français, de façon professionnelle mais accessible.
 
        Documents de référence disponibles :
        - {contexte}
        """}
    ]
 
    # Ajouter l'historique de conversation
    for msg in historique[-6:]:  # Garder les 6 derniers messages
        messages.append(msg)
 
    messages.append({"role": "user", "content": question})
 
    # 3. Appel Mistral
    rep = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {cle_api}", "Content-Type": "application/json"},
        json={"model": "mistral-small-latest", "messages": messages, "temperature": 0.7}
    )
    return rep.json()["choices"][0]["message"]["content"]
 
 
# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Assistant RH")
    st.markdown("*Powered by Mistral AI + RAG*")
    st.markdown("---")
 
    cle_api = st.text_input("🔑 Clé API Mistral", type="password",
                             placeholder="Colle ta clé ici...")
    st.markdown("---")
 
    st.markdown("**📚 Base de connaissances**")
    st.markdown("*12 politiques RH chargées*")
    st.markdown("""
    - ✅ Salaires & grilles
    - ✅ Politique recrutement
    - ✅ Congés & avantages
    - ✅ Télétravail
    - ✅ Formations
    """)
    st.markdown("---")
 
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
 
    st.markdown("---")
    st.markdown("""
    <div style='color:#a78bfa; font-size:0.78rem; text-align:center;'>
    RH Predict · Assistant IA<br>Fetta Adjedour Consulting
    </div>
    """, unsafe_allow_html=True)
 
 
# ── INTERFACE PRINCIPALE ─────────────────────────────────────
st.markdown("""
<div class="chat-header">
  <div style="background:linear-gradient(135deg,#7c3aed,#a855f7); color:white;
              width:52px; height:52px; border-radius:14px; display:flex;
              align-items:center; justify-content:center; font-size:1.6rem;
              box-shadow:0 4px 14px rgba(124,58,237,0.35); flex-shrink:0;">🤖</div>
  <div>
    <div style="font-size:1.3rem; font-weight:800; color:#2d1b69;">Alex — Assistant RH</div>
    <div style="font-size:0.82rem; color:#7c3aed; font-weight:500;">
      🟢 En ligne · Répond instantanément · Basé sur vos politiques RH
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
 
# Initialiser l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Message de bienvenue
if not st.session_state.messages:
    st.markdown("""
    <div class="msg-bot">
    <span class="msg-avatar-bot">🤖</span>
    Bonjour ! Je suis <strong>Alex</strong>, votre assistant RH intelligent.<br><br>
    Je peux vous aider sur :<br>
    • 💼 Les conditions de recrutement<br>
    • 💰 Les grilles salariales<br>
    • 📅 Les congés et avantages<br>
    • 🏠 La politique télétravail<br>
    • 📋 Les procédures RH internes<br><br>
    <em>Posez-moi votre première question !</em>
    </div>
    """, unsafe_allow_html=True)
 
# Afficher l'historique
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">🙋 {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-bot"><span class="msg-avatar-bot">🤖</span>{msg["content"]}</div>',
                    unsafe_allow_html=True)
 
# Zone de saisie
st.markdown("<br>", unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])
 
with col_input:
    question = st.text_input("", placeholder="Posez votre question RH ici...",
                              label_visibility="collapsed", key="input_question")
with col_btn:
    envoyer = st.button("Envoyer ➤", use_container_width=True)
 
# Traitement de la question
if (envoyer or question) and question:
    if not cle_api:
        st.warning("⚠️ Entre ta clé API Mistral dans la sidebar !")
    else:
        # Afficher la question
        st.markdown(f'<div class="msg-user">🙋 {question}</div>',
                    unsafe_allow_html=True)
 
        # Obtenir la réponse
        with st.spinner("Alex réfléchit..."):
            reponse = repondre(question, cle_api, st.session_state.messages)
 
        # Afficher la réponse
        st.markdown(f'<div class="msg-bot"><span class="msg-avatar-bot">🤖</span>{reponse}</div>',
                    unsafe_allow_html=True)
 
        # Sauvegarder dans l'historique
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": reponse})
 
