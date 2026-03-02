import io
import json
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from groq import Groq

st.set_page_config(page_title="TitanicBot 🚢", page_icon="🚢", layout="centered")

st.markdown("""
<style>
    .stApp { background: #0a1628; }
    .main-title { font-size:2.2rem; font-weight:800; color:#fff; text-align:center; margin-bottom:0.2rem; }
    .sub-title  { text-align:center; color:#8ba3c7; font-size:0.9rem; margin-bottom:1rem; }
    .user-bubble {
        background:#1e4d8c; border-radius:18px 18px 4px 18px;
        padding:12px 16px; color:#fff; max-width:78%;
        margin-left:auto; margin-bottom:10px; font-size:0.9rem;
    }
    .bot-bubble {
        background:#1a2744; border:1px solid #2a3f6f;
        border-radius:18px 18px 18px 4px;
        padding:12px 16px; color:#d6e4f7;
        max-width:85%; margin-bottom:10px; font-size:0.9rem;
    }
    .caption-text { color:#8ba3c7; font-size:0.8rem; text-align:center; margin-top:4px; }
    section[data-testid="stSidebar"] { background:#0d1e38 !important; }
    section[data-testid="stSidebar"] * { color:#8ba3c7 !important; }
    .stTextInput > div > div > input {
        background:#1a2744 !important; color:#fff !important;
        border:1px solid #2a3f6f !important; border-radius:24px !important;
        padding:12px 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Dataset ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()

# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚢 TitanicBot")
    groq_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("---")
    st.markdown("**Dataset:** 891 passengers")
    st.markdown("**Columns:** Survived, Pclass, Sex, Age, Fare, Embarked")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Fig to base64 ──────────────────────────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return encoded

# ── Get Stats (no LangChain, no tokens wasted) ────────────────────────────────
def get_stats(q):
    q = q.lower()
    if "male" in q or "female" in q or "gender" in q or "sex" in q:
        counts = df["Sex"].value_counts()
        total = len(df)
        return (f"Gender breakdown:\n"
                f"- Male: {counts.get('male',0)} ({counts.get('male',0)/total*100:.1f}%)\n"
                f"- Female: {counts.get('female',0)} ({counts.get('female',0)/total*100:.1f}%)")
    if "surviv" in q:
        rate = df["Survived"].mean() * 100
        return (f"Survival stats:\n"
                f"- Survived: {df['Survived'].sum()} ({rate:.1f}%)\n"
                f"- Perished: {len(df)-df['Survived'].sum()} ({100-rate:.1f}%)")
    if "fare" in q:
        return (f"Ticket fare stats:\n"
                f"- Average: ${df['Fare'].mean():.2f}\n"
                f"- Median: ${df['Fare'].median():.2f}\n"
                f"- Max: ${df['Fare'].max():.2f}")
    if "age" in q:
        age = df["Age"].dropna()
        return (f"Age stats:\n"
                f"- Average: {age.mean():.1f} yrs\n"
                f"- Median: {age.median():.1f} yrs\n"
                f"- Range: {age.min():.0f}–{age.max():.0f}")
    if "embark" in q or "port" in q:
        port_map = {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}
        counts = df["Embarked"].value_counts()
        lines = [f"- {name}: {counts.get(code,0)}" for code,name in port_map.items()]
        return "Embarkation ports:\n" + "\n".join(lines)
    if "class" in q or "pclass" in q:
        counts = df["Pclass"].value_counts().sort_index()
        lines = [f"- Class {c}: {n}" for c,n in counts.items()]
        return "Passengers by class:\n" + "\n".join(lines)
    return f"Total passengers: {len(df)}"

# ── Create Chart ───────────────────────────────────────────────────────────────
def create_chart(q):
    req = q.lower()
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(7, 4))
    caption = ""

    if "age" in req:
        data = df["Age"].dropna()
        ax.hist(data, bins=25, color="#4C72B0", edgecolor="white")
        ax.set_title("Passenger Ages"); ax.set_xlabel("Age"); ax.set_ylabel("Count")
        caption = f"Avg: {data.mean():.1f} yrs"
    elif "fare" in req:
        data = df["Fare"].dropna()
        ax.hist(data, bins=30, color="#55A868", edgecolor="white")
        ax.set_title("Ticket Fares"); ax.set_xlabel("Fare (£)"); ax.set_ylabel("Count")
        caption = f"Avg: £{data.mean():.2f}"
    elif "surviv" in req and "class" in req:
        s = df.groupby("Pclass")["Survived"].mean() * 100
        bars = ax.bar(["1st","2nd","3rd"], s.values, color=["#4C72B0","#55A868","#C44E52"])
        ax.set_title("Survival Rate by Class"); ax.set_ylabel("%"); ax.set_ylim(0,100)
        for b, v in zip(bars, s.values):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontweight="bold")
        caption = "Survival by class"
    elif "surviv" in req and ("gender" in req or "sex" in req or "male" in req or "female" in req):
        s = df.groupby("Sex")["Survived"].mean() * 100
        bars = ax.bar(s.index.str.capitalize(), s.values, color=["#C44E52","#4C72B0"])
        ax.set_title("Survival Rate by Gender"); ax.set_ylabel("%"); ax.set_ylim(0,100)
        for b, v in zip(bars, s.values):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontweight="bold")
        caption = "Survival by gender"
    elif "surviv" in req:
        counts = df["Survived"].value_counts()
        ax.pie([counts.get(0,0), counts.get(1,0)], labels=["Perished","Survived"],
               colors=["#C44E52","#55A868"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Survival Distribution")
        caption = f"{df['Survived'].sum()} survived of {len(df)}"
    elif "embark" in req or "port" in req:
        port_map = {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}
        counts = df["Embarked"].map(port_map).value_counts()
        bars = ax.bar(counts.index, counts.values, color=["#4C72B0","#55A868","#C44E52"])
        ax.set_title("Passengers by Port"); ax.set_ylabel("Count")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x()+b.get_width()/2, v+2, str(v), ha="center", fontweight="bold")
        caption = "By embarkation port"
    elif "class" in req or "pclass" in req:
        counts = df["Pclass"].value_counts().sort_index()
        bars = ax.bar(["1st","2nd","3rd"], counts.values, color=["#4C72B0","#55A868","#C44E52"])
        ax.set_title("Passengers by Class"); ax.set_ylabel("Count")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x()+b.get_width()/2, v+2, str(v), ha="center", fontweight="bold")
        caption = "By ticket class"
    elif "gender" in req or "sex" in req:
        counts = df["Sex"].value_counts()
        bars = ax.bar(counts.index.str.capitalize(), counts.values, color=["#4C72B0","#C44E52"])
        ax.set_title("Passengers by Gender"); ax.set_ylabel("Count")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x()+b.get_width()/2, v+2, str(v), ha="center", fontweight="bold")
        caption = "Gender distribution"
    else:
        counts = df["Survived"].value_counts()
        ax.pie([counts.get(0,0), counts.get(1,0)], labels=["Perished","Survived"],
               colors=["#C44E52","#55A868"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Survival Distribution")
        caption = "Overall survival"

    return fig_to_base64(fig), caption

# ── Route question (no LangChain) ─────────────────────────────────────────────
def route_and_answer(question, api_key):
    q = question.lower()
    wants_chart = any(w in q for w in ["show","plot","chart","histogram","visuali","graph","display"])

    if wants_chart:
        image_b64, caption = create_chart(q)
        # Use Groq only for a short text reply (~50 tokens)
        client = Groq(api_key=api_key)
        stats = get_stats(q)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=80,
            messages=[{"role":"user","content":f"In 1-2 sentences summarize: {stats}"}]
        )
        text = resp.choices[0].message.content
        return text, image_b64, caption
    else:
        stats = get_stats(q)
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=100,
            messages=[
                {"role":"system","content":"You are TitanicBot. Answer briefly using the data provided."},
                {"role":"user","content":f"Data: {stats}\nQuestion: {question}"}
            ]
        )
        text = resp.choices[0].message.content
        return text, None, None

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚢 TitanicBot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask anything about Titanic passengers — stats & charts!</div>', unsafe_allow_html=True)

st.markdown("**Quick questions:**")
cols = st.columns(3)
quick = [
    "What % of passengers were male?",
    "Show histogram of passenger ages",
    "What was the average ticket fare?",
    "Passengers by embarkation port",
    "Show survival rate by class",
    "Show survival rate by gender",
]
for i, q in enumerate(quick):
    if cols[i % 3].button(q, key=f"q{i}"):
        st.session_state.pending = q

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">🚢 {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("image_b64"):
            img = Image.open(io.BytesIO(base64.b64decode(msg["image_b64"])))
            st.image(img, use_column_width=True)
            if msg.get("caption"):
                st.markdown(f'<div class="caption-text">📊 {msg["caption"]}</div>', unsafe_allow_html=True)

st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("Ask a question…", value=st.session_state.pending,
                                key="input", label_visibility="collapsed",
                                placeholder="e.g. Show survival rate by class")
with col2:
    send = st.button("Send ➤", use_container_width=True)

def handle_send(question: str):
    if not question.strip():
        return
    if not groq_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar!")
        return

    st.session_state.pending = ""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🚢 Analyzing..."):
        try:
            text, image_b64, caption = route_and_answer(question, groq_key)
            st.session_state.messages.append({
                "role": "assistant",
                "content": text,
                "image_b64": image_b64,
                "caption": caption,
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {str(e)}",
            })

    st.rerun()

if send and user_input.strip():
    handle_send(user_input)
elif st.session_state.pending:
    handle_send(st.session_state.pending)