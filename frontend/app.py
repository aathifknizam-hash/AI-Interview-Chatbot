"""
app.py — Streamlit Chat Frontend
==================================
2026 stack: st.chat_message + st.chat_input native chat UI,
session-state message history, sidebar health check and topic browser.

Run with:
    streamlit run app.py
"""

import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────
API_BASE_URL    = "http://localhost:8000"
REQUEST_TIMEOUT = 10          # seconds before giving up on the backend
MAX_HISTORY     = 100         # cap message history to avoid huge session state

WELCOME_MESSAGE = (
    "Hi! I'm your **AI Interview Coach**. "
    "Ask me anything about Machine Learning, Deep Learning, NLP, Statistics, "
    "MLOps, or career strategy — or just say **'mock interview'** to get started!"
)

TOPIC_LIST = [
    "Bias-Variance Tradeoff",
    "Precision, Recall, F1",
    "Overfitting & Regularisation",
    "Gradient Descent",
    "Backpropagation",
    "CNNs / RNNs / Transformers",
    "BERT & LLMs",
    "ML in Production (MLOps)",
    "Feature Engineering",
    "Statistics & Probability",
    "SQL & Data Engineering",
    "Resume & Portfolio Tips",
    "Mock Interview Questions",
    "Behavioural Questions",
]


# ── Page setup (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ── Session state initialisation ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]


# ── Helpers ────────────────────────────────────────────────────────────────
def _call_api(user_text: str) -> str:
    """POST to /chat and return the bot response string."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": user_text},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "No response received.")
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Cannot reach the backend. "
            "Make sure the FastAPI server is running on port 8000:\n\n"
            "```\npython main.py\n```"
        )
    except requests.exceptions.Timeout:
        return "⚠️ The backend took too long to respond. Please try again."
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response else "?"
        if code == 422:
            return "⚠️ Message was rejected by the server (too long or empty)."
        if code == 503:
            return (
                "⚠️ The model index is not loaded. "
                "Run `train.py` first then restart the server."
            )
        return f"⚠️ Server error {code}. Please try again."
    except Exception as exc:
        return f"⚠️ Unexpected error: {exc}"


def _check_health() -> tuple[bool, str]:
    """Return (is_healthy, version_string)."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get("version", "?")
        return False, ""
    except Exception:
        return False, ""


def _add_message(role: str, content: str) -> None:
    msgs: list = st.session_state["messages"]
    msgs.append({"role": role, "content": content})
    # Trim history to avoid unbounded growth (keep welcome message at index 0)
    if len(msgs) > MAX_HISTORY:
        st.session_state["messages"] = [msgs[0]] + msgs[-(MAX_HISTORY - 1):]


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 AI Interview Coach")
    st.caption("Powered by sentence-transformers")

    # Health indicator
    is_healthy, version = _check_health()
    if is_healthy:
        st.success(f"Backend online — v{version}")
    else:
        st.error("Backend offline — start `python main.py`")

    st.divider()

    # Quick-topic buttons (each sends a predefined query)
    st.markdown("**Quick topics**")
    st.caption("Click any topic to ask about it.")
    for topic in TOPIC_LIST:
        if st.button(topic, use_container_width=True, key=f"topic_{topic}"):
            query = f"Explain {topic}"
            _add_message("user", query)
            with st.spinner("Thinking …"):
                answer = _call_api(query)
            _add_message("assistant", answer)
            st.rerun()

    st.divider()

    # Clear conversation
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────
st.markdown("## 🎯 AI Interview Coach")
st.caption("Your personal ML & AI interview preparation assistant")

# Render all past messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept new input
if prompt := st.chat_input("Ask me anything about AI interviews …"):
    # Show user bubble immediately
    _add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            answer = _call_api(prompt)
        st.markdown(answer)

    _add_message("assistant", answer)