import streamlit as st
import requests
import chromadb
import io
from sentence_transformers import SentenceTransformer

try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

import streamlit.components.v1 as components

st.set_page_config(
    page_title="Alex — Assistant RH",
    page_icon="✨",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── FOND PRINCIPAL ── */
.stApp { background: linear-gradient(160deg, #faf5ff 0%, #f5f0fe 50%, #fefce8 100%); }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 920px; margin: 0 auto; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e0a3c 0%, #2d1657 60%, #1e0a3c 100%);
    border-right: 1px solid #4c2a85;
}
[data-testid="stSidebar"] * { color: #f0e6ff !important; }
[data-testid="stSidebar"] h3 { color: #fbbf24 !important; font-size: 0.95rem !important; font-weight: 700 !important; letter-spacing: 0.03em; }
[data-testid="stSidebar"] label { color: #e2d9f3 !important; font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.07em; }
[data-testid="stSidebar"] p { color: #f0e6ff !important; font-size: 0.85rem !important; font-weight: 500 !important; }
[data-testid="stSidebar"] li { color: #f0e6ff !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] span { color: #f0e6ff !important; }
[data-testid="stSidebar"] small { color: #d8ccf0 !important; }
[data-testid="stSidebar"] hr { border-color: #4c2a85 !important; margin: 0.8rem 0 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: #0f051f !important; color: #f0e6ff !important;
    border: 1px solid #6d28d9 !important; border-radius: 8px !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder { color: #9f7aea !important; }
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.82rem !important; width: 100% !important;
    padding: 0.55rem 1rem !important;
    box-shadow: 0 3px 12px rgba(124,58,237,0.35) !important;
}
[data-testid="stSidebar"] .stFileUploader {
    background: #0f051f !important; border-radius: 10px !important;
    border: 2px dashed #6d28d9 !important; padding: 0.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * { color: #d8ccf0 !important; }
[data-testid="stSidebar"] .stSuccess { background: rgba(34,197,94,0.15) !important; border-radius: 8px !important; }
[data-testid="stSidebar"] .stSuccess * { color: #86efac !important; }

/* ── TEXTE PRINCIPAL ── */
h1, h2, h3, h4 { color: #1e0a3c !important; }
p, li { color: #3b0764 !important; }
strong { color: #1e0a3c !important; }
.stMarkdown p { color: #3b0764 !important; }

/* ── MÉTRIQUES ── */
[data-testid="stMetric"] {
    background: white; border-radius: 12px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 2px 12px rgba(109,40,217,0.08);
    border-top: 3px solid #c4a84f;
}
[data-testid="stMetricValue"] { color: #1e0a3c !important; font-weight: 800 !important; font-size: 1.2rem !important; }
[data-testid="stMetricLabel"] { color: #6d28d9 !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── INPUT QUESTION ── */
.stTextInput input {
    border-radius: 30px !important;
    border: 2px solid #ddd6fe !important;
    padding: 12px 20px !important;
    font-size: 0.9rem !important;
    background: white !important;
    color: #1e0a3c !important;
    box-shadow: 0 2px 8px rgba(109,40,217,0.06) !important;
}
.stTextInput input:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important; }
.stTextInput input::placeholder { color: #a78bfa !important; }

/* ── BOUTON ENVOYER ── */
.stButton > button {
    background: linear-gradient(135deg, #c4a84f, #a88730) !important;
    color: white !important; border: none !important;
    border-radius: 30px !important; font-weight: 700 !important;
    font-size: 0.88rem !important; padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(196,168,79,0.4) !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(196,168,79,0.5) !important; }

/* ── MESSAGES ── */
.msg-user {
    background: linear-gradient(135deg, #7c3aed, #5b21b6);
    color: white !important;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%;
    font-size: 0.88rem; line-height: 1.65;
    box-shadow: 0 4px 14px rgba(124,58,237,0.3);
}
.msg-user * { color: white !important; }
.msg-bot {
    background: white;
    color: #1e0a3c !important;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 8px 0;
    font-size: 0.88rem; line-height: 1.65;
    box-shadow: 0 2px 10px rgba(109,40,217,0.08);
    border-left: 4px solid #c4a84f;
}
.msg-bot * { color: #1e0a3c !important; }
.msg-bot strong { color: #1e0a3c !important; font-weight: 700; }

/* ── ALERTES ── */
.stWarning { background: #fef3c7 !important; border-left: 3px solid #f59e0b !important; color: #78350f !important; border-radius: 8px !important; }
.stSuccess { background: #f0fdf4 !important; border-left: 3px solid #22c55e !important; color: #14532d !important; border-radius: 8px !important; }

hr { border-color: #ddd6fe !important; }

/* ── SIDEBAR TEXTE FORCE BLANC ── */
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .element-container p,
section[data-testid="stSidebar"] .element-container span {
    color: #f0e6ff !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── CHAT INPUT BARRE DU BAS ── */
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div,
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: linear-gradient(160deg, #faf5ff, #f5f0fe) !important;
    border-top: 1px solid #ede9fe !important;
}
[data-testid="stChatInput"] textarea,
textarea[data-testid="stChatInputTextArea"],
[data-testid="stChatInputTextArea"] {
    background: white !important;
    color: #1e0a3c !important;
    border: 2px solid #ddd6fe !important;
    border-radius: 30px !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #c4a84f, #a88730) !important;
    border-radius: 50% !important; border: none !important;
    box-shadow: 0 3px 10px rgba(196,168,79,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ── MODÈLE & BASE RAG ────────────────────────────────────────
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
    if collection.count() == 0:
        emb = modele.encode(documents).tolist()
        collection.add(documents=documents, embeddings=emb,
                       ids=[f"doc_{i}" for i in range(len(documents))])
    return modele, collection

modele_embed, collection = charger_systeme()


def extraire_texte_pdf(fichier):
    """Extrait le texte d'un fichier PDF."""
    if not PDF_OK:
        return "⚠️ Module pdfplumber non installé."
    texte = ""
    with pdfplumber.open(io.BytesIO(fichier.read())) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texte += t + "\n"
    return texte.strip()


def repondre(question, cle_api, historique, cv_texte=""):
    q_emb = modele_embed.encode([question]).tolist()
    resultats = collection.query(query_embeddings=q_emb, n_results=3)
    contexte = "\n- ".join(resultats['documents'][0])
    cv_section = f"\n\nCV DU CANDIDAT :\n{cv_texte[:3000]}" if cv_texte else ""

    messages = [{"role": "system", "content": f"""Tu es Alex, assistant RH expert, bienveillant et professionnel.
Tu aides les équipes RH à prendre les meilleures décisions de recrutement.
Tu réponds UNIQUEMENT en français, de façon claire et structurée.
Appuie-toi sur les documents RH fournis. Si une info manque, dis-le poliment.

Politiques RH de référence :
- {contexte}{cv_section}
"""}]

    for msg in historique[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    rep = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {cle_api}", "Content-Type": "application/json"},
        json={"model": "mistral-small-latest", "messages": messages, "temperature": 0.65}
    )
    return rep.json()["choices"][0]["message"]["content"]


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 0.2rem;">
      <div style="font-size:2rem;">✨</div>
      <div style="font-size:1.05rem; font-weight:800; color:#fbbf24;">Alex</div>
      <div style="font-size:0.75rem; color:#c4b5fd; margin-top:2px;">Assistant RH · Mistral AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🔑 Clé API Mistral")
    cle_api = st.text_input("", type="password", placeholder="Colle ta clé ici...",
                             label_visibility="collapsed")
    st.markdown("---")

    st.markdown("### 📄 Charger un CV")
    st.markdown("<p style='font-size:0.78rem; margin-bottom:6px;'>Formats acceptés : PDF ou TXT</p>",
                unsafe_allow_html=True)
    cv_file = st.file_uploader("", type=["pdf", "txt"], label_visibility="collapsed")

    cv_texte = ""
    if cv_file:
        if cv_file.name.endswith(".pdf"):
            cv_texte = extraire_texte_pdf(cv_file)
        else:
            cv_texte = cv_file.read().decode("utf-8")

        if cv_texte:
            st.success(f"✅ {cv_file.name}")
            mots = len(cv_texte.split())
            st.markdown(f"<p style='font-size:0.78rem;'>📊 {mots} mots extraits</p>",
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Base de connaissances")
    for item in ["✅ Salaires & grilles", "✅ Recrutement", "✅ Congés", "✅ Télétravail", "✅ Formations"]:
        st.markdown(f"<p>{item}</p>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── MÉTRIQUES ────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Statut", "🟢 En ligne")
with c2:
    st.metric("Base RH", "12 politiques")
with c3:
    st.metric("CV", "✅ Chargé" if cv_texte else "— Non chargé")

st.markdown("<br>", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style="background:white; border-radius:14px; padding:1rem 1.4rem;
            box-shadow:0 3px 16px rgba(109,40,217,0.1);
            display:flex; align-items:center; gap:14px; margin-bottom:1rem;
            border-bottom:3px solid #c4a84f;">
  <div style="background:linear-gradient(135deg,#7c3aed,#5b21b6); color:white;
              width:48px; height:48px; border-radius:12px; display:flex;
              align-items:center; justify-content:center; font-size:1.5rem;
              box-shadow:0 4px 14px rgba(124,58,237,0.4); flex-shrink:0;">✨</div>
  <div>
    <div style="font-size:1.1rem; font-weight:800; color:#1e0a3c;">Alex — Assistant RH Intelligent</div>
    <div style="font-size:0.78rem; color:#c4a84f; font-weight:600; margin-top:3px;">
      Analyse de CV (PDF) &nbsp;·&nbsp; Politiques RH &nbsp;·&nbsp; Aide au recrutement
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MESSAGES ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div class="msg-bot">
      <strong>Bonjour 👋</strong> Je suis <strong>Alex</strong>, votre assistant RH alimenté par l'IA.<br><br>
      Je peux vous aider à :<br>
      &nbsp;&nbsp;📄 <strong>Analyser un CV</strong> — chargez-le en PDF dans la sidebar<br>
      &nbsp;&nbsp;💰 <strong>Vérifier les grilles salariales</strong><br>
      &nbsp;&nbsp;📋 <strong>Répondre à vos questions RH</strong><br>
      &nbsp;&nbsp;✅ <strong>Recommander ou écarter un profil</strong><br><br>
      <em>Par quoi souhaitez-vous commencer ?</em>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    css_class = "msg-user" if msg["role"] == "user" else "msg-bot"
    icone = "🙋" if msg["role"] == "user" else "✨"
    st.markdown(f'<div class="{css_class}">{icone} {msg["content"]}</div>',
                unsafe_allow_html=True)

# ── FORCER COULEUR BARRE DU BAS (JS) ─────────────────────────
components.html("""
<script>
function fixChatBar() {
    const doc = window.parent.document;
    const targets = doc.querySelectorAll(
        '[data-testid="stBottom"], .stChatFloatingInputContainer'
    );
    targets.forEach(el => {
        el.style.background = 'linear-gradient(160deg, #faf5ff, #f5f0fe)';
        el.style.borderTop = '1px solid #ede9fe';
    });
}
fixChatBar();
setTimeout(fixChatBar, 500);
setTimeout(fixChatBar, 1500);
</script>
""", height=0)

# ── INPUT FIXÉ EN BAS ─────────────────────────────────────────
question = st.chat_input("Posez votre question à Alex...")

if question:
    if not cle_api:
        st.warning("⚠️ Entrez votre clé API Mistral dans la sidebar pour commencer.")
    else:
        st.markdown(f'<div class="msg-user">🙋 {question}</div>', unsafe_allow_html=True)
        with st.spinner("Alex réfléchit..."):
            reponse = repondre(question, cle_api, st.session_state.messages, cv_texte)
        st.markdown(f'<div class="msg-bot">✨ {reponse}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": reponse})
        st.rerun()

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center; color:#a78bfa; font-size:0.72rem; padding:0.4rem 0;">
  Alex · Assistant RH · <strong style="color:#c4a84f;">Fetta Adjedour Consulting</strong> · Powered by Mistral AI &amp; RAG
</div>
""", unsafe_allow_html=True)
