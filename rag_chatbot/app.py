import streamlit as st
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from rag_pipeline import TekMarkRAG

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TekBot")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TekMark AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Hide avatars */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"],
.stChatMessage > div:first-child,
[data-testid="stChatMessageAvatarContainer"] {
    display: none !important;
}

/* Outer Chat Window */
.chat-window {
    border: 2px solid #D1D5DB;
    border-radius: 22px;
    padding: 25px;
    background: white;
    box-shadow: 0 10px 28px rgba(0,0,0,0.06);
    margin-top: 20px;
}

/* Header Box */
.bot-header-box {
    border: 2px solid #E5E7EB;
    border-radius: 16px;
    padding: 20px 24px;
    background-color: #F9FAFB;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}

.bot-header-title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 6px;
}

.bot-header-subtitle {
    font-size: 14px;
    font-weight: 700;
    color: #374151;
}

/* User Bubble LEFT */
.user-bubble {
    background: #1E73E8;
    color: white;
    padding: 14px 18px;
    border-radius: 16px;
    width: fit-content;
    max-width: 75%;
    margin-right: auto;
    margin-bottom: 15px;
    font-weight: 500;
    font-size: 17px;
    box-shadow: 0 4px 10px rgba(30,115,232,0.25);
}

/* AI Bubble RIGHT */
.ai-bubble {
    background: #F3F4F6;
    padding: 14px 18px;
    border-radius: 16px;
    width: fit-content;
    max-width: 75%;
    margin-left: auto;
    margin-bottom: 15px;
    font-weight: 500;
    font-size: 17px;
    color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag" not in st.session_state:
    st.session_state.rag = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ---------------- INIT RAG ----------------
if not st.session_state.indexed:
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    pinecone_key = os.getenv("PINECONE_API_KEY", "")

    if gemini_key and pinecone_key:
        try:
            rag = TekMarkRAG(
                gemini_api_key=gemini_key,
                pinecone_api_key=pinecone_key,
                pinecone_index=os.getenv("PINECONE_INDEX", "tekmark-rag"),
            )

            stats = rag.index.describe_index_stats()
            vector_count = stats.total_vector_count

            if vector_count == 0:
                pdf_path = os.path.join(
                    os.path.dirname(__file__),
                    "TekMark Website Content.pdf"
                )

                if Path(pdf_path).exists():
                    rag.index_document(pdf_path)

            st.session_state.rag = rag
            st.session_state.indexed = True

        except Exception as e:
            log.error(f"RAG Init Failed: {e}")

# ---------------- MAIN UI ----------------
logo_path = os.path.join(os.path.dirname(__file__), "..", "TekMark---Logo-01.png")

st.markdown('<div class="chat-window">', unsafe_allow_html=True)

if Path(logo_path).exists():
    st.image(logo_path, width=220, )

st.markdown("""
<div class="bot-header-box">
    <div class="bot-header-title">
        AI<span style="color:black;">Tek</span>BOT
    </div>
    <div class="bot-header-subtitle">
        Intelligent Conversations. Exceptional Customer Experiences.
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(
    "**Hi! I’m an AI-powered Chatbot Assistant. I can help answer any questions "
    "you have about TekMark’s QA services and AI solutions. How can I assist you today?**"
)

st.caption("*Example: 'Can TekMark support API testing?'*")

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="ai-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

# ---------------- GENERATE RESPONSE IF THINKING ----------------
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
    and st.session_state.messages[-1]["content"] == "Thinking..."
):
    try:
        last_user_msg = next(
            msg["content"]
            for msg in reversed(st.session_state.messages)
            if msg["role"] == "user"
        )

        result = st.session_state.rag.query(
            last_user_msg,
            chat_history=st.session_state.messages[:-2]
        )

        email_footer = (
            "\n\n If you want more information about our services, "
            "please email us at: info@tekmarksolutions.com"
        )

        st.session_state.messages[-1]["content"] = result["answer"] + email_footer

        st.rerun()

    except Exception as e:
        st.session_state.messages[-1]["content"] = f"Error: {e}"
        st.rerun()

# ---------------- CHAT INPUT ----------------
if question := st.chat_input("Ask about TekMark services..."):

    if not st.session_state.indexed or st.session_state.rag is None:
        st.error("Could not connect to services. Check API keys.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": "Thinking..."
        })

        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
