import streamlit as st
import requests
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Alex — Assistant RH",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

/* === FOND === */
.stApp { background-color: #f1f5f9; }
.main .block-container { padding: 1.5rem 2rem; max-width: 900px; margin: 0 auto; }

/* === SIDEBAR === */
[data-testid="stSidebar"] { background-color: #1e293b; }
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.8rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stSidebar"] p { color: #cbd5e1 !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-size: 1rem !important; font-weight: 700 !important; }
[data-testid="stSidebar"] hr { border-color: #334155 !important; }
[data-testid="stSidebar"] li { color: #94a3b8 !important; font-size: 0.83rem !important; }
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 0.85rem !important; width: 100%;
    padding: 0.6rem !important;
    box-shadow: 0 4px 12px rgba(249,115,22,0.3) !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #0f172a !important; color: #f1f5f9 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
}

/* === TEXTE PRINCIPAL === */
h1, h2, h3, h4 { color: #0f172a !important; }
p, li, span { color: #334155 !important; }
strong { color: #0f172a !important; }

/* === HEADER CHAT === */
.chat-header {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    display: flex; align-items: center; gap: 14px;
    border-bottom: 3px solid #0ea5e9;
}

/* === MESSAGES === */
.msg-user {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white !important;
    padding: 11px 16px;
    border-radius: 16px 16px 4px 16px;
    margin: 6px 0 6px 25%;
    font-size: 0.9rem; line-height: 1.6;
    box-shadow: 0 3px 10px rgba(14,165,233,0.25);
}
.msg-user p, .msg-user span { color: white !important; }
.msg-bot {
    background: white;
    color: #1e293b !important;
    padding: 11px 16px;
    border-radius: 16px 16px 16px 4px;
    margin: 6px 25% 6px 0;
    font-size: 0.9rem; line-height: 1.6;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    border-left: 3px solid #0ea5e9;
}
.msg-bot p, .msg-bot span, .msg-bot li { color: #1e293b !important; }

/* === UPLOAD CARD === */
.upload-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 2px dashed #e2e8f0;
}
.upload-card:hover { border-color: #0ea5e9; }

/* === INPUT === */
.stTextInput input {
    border-radius: 25px !important;
    border: 2px solid #e2e8f0 !important;
    padding: 10px 18px !important;
    font-size: 0.9rem !important;
    background: white !important;
    color: #1e293b !important;
}
.stTextInput input:focus { border-color: #0ea5e9 !important; }
.stTextInput input::placeholder { color: #94a3b8 !important; }

/* === BOUTON ENVOYER === */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important; border: none !important;
    border-radius: 25px !important; font-weight: 700 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 4px 12px rgba(14,165,233,0.3) !important;
}

/* === MÉTRIQUES === */
[data-testid="stMetric"] {
    background: white; border-radius: 10px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    border-top: 2px solid #0ea5e9;
}
[data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }

hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── MODÈLE & BASE DE DONNÉES ────────────────────────────────
@st.cache_resource
def charger_systeme():
    modele = SentenceTransformer('all-MiniLM-L6-v2')
    client_db = chromadb.Client()
    collection = client_db.get_or_create_collection("rh_docs")
    documents = [
        "Le poste de Data Scientist requiert minimum 2 ans d'expérience en Python et Machine Learning.",
        "La politique salariale pour un Data Scientist junior est entre 38 000€ et 45 000€ par an.",
        "Les candidats sans diplôme peuvent être acceptés si leur portfolio GitHub est solide.",
        "Un entretien technique de 2 heures est obligatoire pour tout poste tech avant embauche.",
        "La période d'essai est de 3 mois renouvelable une fois pour les cadres.",
        "Le télétravail est possible jusqu'à 3 jours par semaine après la période d'essai.",
        "Les congés payés sont de 25 jours ouvrés par an.",
        "Le poste de Chef de Projet requiert 5 ans d'expérience minimum.",
        "Les entretiens annuels d'évaluation ont lieu en janvier de chaque année.",
        "La mutuelle entreprise couvre 60% des cotisations santé.",
        "Un plan de formation est proposé avec un budget de 1 500€ par an.",
        "Le recrutement pour les postes seniors nécessite l'accord de la direction.",
    ]
    emb = modele.encode(documents).tolist()
    collection.add(documents=documents, embeddings=emb,
                   ids=[f"doc_{i}" for i in range(len(documents))])
    return modele, collection

modele_embed, collection = charger_systeme()


# ── FONCTION RAG ─────────────────────────────────────────────
def repondre(question, cle_api, historique, cv_texte=""):
    q_emb = modele_embed.encode([question]).tolist()
    resultats = collection.query(query_embeddings=q_emb, n_results=3)
    contexte = "\n- ".join(resultats['documents'][0])

    cv_section = f"\n\nCV DU CANDIDAT ANALYSÉ :\n{cv_texte}" if cv_texte else ""

    messages = [{"role": "system", "content": f"""Tu es Alex, assistant RH expert et bienveillant.
Tu aides les équipes RH à prendre de meilleures décisions de recrutement.
Tu réponds UNIQUEMENT en français, de façon professionnelle et claire.
Tu te bases sur les documents RH fournis. Si une info est absente, dis-le poliment.

Documents RH de référence :
- {contexte}{cv_section}
"""}]

    for msg in historique[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    rep = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {cle_api}", "Content-Type": "application/json"},
        json={"model": "mistral-small-latest", "messages": messages, "temperature": 0.7}
    )
    return rep.json()["choices"][0]["message"]["content"]


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Alex — Assistant RH")
    st.markdown("*Mistral AI · RAG · Analyse CV*")
    st.markdown("---")

    cle_api = st.text_input("🔑 Clé API Mistral", type="password",
                             placeholder="Colle ta clé ici...")
    st.markdown("---")

    st.markdown("### 📄 Charger un CV")
    cv_file = st.file_uploader("Dépose un fichier CV (.txt)", type=["txt"],
                                label_visibility="collapsed")
    cv_texte = ""
    if cv_file:
        cv_texte = cv_file.read().decode("utf-8")
        st.success(f"✅ CV chargé : {cv_file.name}")
        st.markdown(f"*{len(cv_texte.split())} mots analysés*")

    st.markdown("---")
    st.markdown("### 📚 Base de connaissances")
    st.markdown("12 politiques RH chargées")
    for item in ["Salaires & grilles", "Recrutement", "Congés", "Télétravail", "Formations"]:
        st.markdown(f"✅ {item}")

    st.markdown("---")
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── HEADER ───────────────────────────────────────────────────
col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    st.metric("Statut", "🟢 En ligne")
with col_h2:
    st.metric("Documents RH", "12 politiques")
with col_h3:
    st.metric("CV chargé", "✅ Oui" if cv_texte else "❌ Non")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
  <div style="background:linear-gradient(135deg,#0ea5e9,#0284c7); color:white;
              width:46px; height:46px; border-radius:12px; display:flex;
              align-items:center; justify-content:center; font-size:1.4rem;
              box-shadow:0 4px 12px rgba(14,165,233,0.35); flex-shrink:0;">🤖</div>
  <div>
    <div style="font-size:1.15rem; font-weight:800; color:#0f172a;">Alex — Assistant RH Intelligent</div>
    <div style="font-size:0.8rem; color:#0ea5e9; font-weight:600; margin-top:2px;">
      Analyse de CV · Politiques RH · Recommandations de recrutement
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── CHAT ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div class="msg-bot">
    🤖 <strong>Bonjour !</strong> Je suis <strong>Alex</strong>, votre assistant RH.<br><br>
    Je peux vous aider à :<br>
    • 📄 <strong>Analyser un CV</strong> — chargez-le dans la sidebar<br>
    • 💰 <strong>Vérifier les grilles salariales</strong><br>
    • 📋 <strong>Répondre sur vos politiques RH</strong><br>
    • ✅ <strong>Recommander ou non un candidat</strong><br><br>
    <em>Quelle est votre question ?</em>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">🙋 {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-bot">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_i, col_b = st.columns([5, 1])
with col_i:
    question = st.text_input("", placeholder="Ex: Analyse ce CV pour un poste de Data Scientist...",
                              label_visibility="collapsed", key="question_input")
with col_b:
    envoyer = st.button("Envoyer ➤", use_container_width=True)

if (envoyer or question) and question:
    if not cle_api:
        st.warning("⚠️ Entre ta clé API Mistral dans la sidebar !")
    else:
        st.markdown(f'<div class="msg-user">🙋 {question}</div>', unsafe_allow_html=True)
        with st.spinner("Alex analyse..."):
            reponse = repondre(question, cle_api, st.session_state.messages, cv_texte)
        st.markdown(f'<div class="msg-bot">🤖 {reponse}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": reponse})

st.markdown("""
<hr>
<div style="text-align:center; color:#94a3b8; font-size:0.75rem; padding:0.5rem 0;">
Alex · Assistant RH · Fetta Adjedour Consulting · Powered by Mistral AI
</div>
""", unsafe_allow_html=True)
