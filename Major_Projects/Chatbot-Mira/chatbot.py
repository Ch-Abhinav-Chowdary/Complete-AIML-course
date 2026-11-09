"""A polished, session-based Streamlit chatbot powered by Google Gemini."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


APP_TITLE = "Mira"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_SYSTEM_PROMPT = (
    "You are an original, upbeat AI assistant with bold, determined, "
    "never-give-up energy. Be warmly enthusiastic, optimistic, and genuinely "
    "motivating. Help people see what they can do next, celebrate their effort "
    "and progress, and encourage them to keep going with practical, specific "
    "support. Be sincere, not relentlessly cheerful: listen carefully, "
    "acknowledge difficult feelings, and never minimize someone's problems "
    "or promise that everything will work out. Stay respectful, truthful, "
    "and clear; ask a clarifying question when needed."
)
MODEL_NAME = DEFAULT_MODEL
TEMPERATURE = 0.7
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
LOGGER = logging.getLogger(__name__)

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

st.set_page_config(
    page_title=f"{APP_TITLE} · AI Chat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #e9edf7;
        --muted: #9aa6be;
        --line: rgba(170, 190, 225, .13);
        --panel: rgba(19, 27, 46, .74);
        --accent: #9b8cff;
        --accent-2: #67dfd0;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(101, 83, 184, .20), transparent 34rem),
            radial-gradient(ellipse at 94% 24%, rgba(42, 153, 152, .12), transparent 30rem),
            #0b1020;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: rgba(12, 17, 33, .92);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    .brand {
        display: flex; align-items: center; gap: .75rem;
        padding: .35rem .2rem 1.7rem;
    }
    .brand-mark {
        display: grid; place-items: center; width: 2.45rem; height: 2.45rem;
        border-radius: .9rem; color: #14142a; font-size: 1.35rem;
        background: linear-gradient(140deg, #c0b5ff, #78e7d7);
        box-shadow: 0 0 28px rgba(147, 132, 255, .26);
    }
    .brand-name {
        color: var(--ink); font: 800 1.16rem 'Manrope', sans-serif;
        letter-spacing: -.04em;
    }
    .brand-caption { color: var(--muted); font-size: .73rem; margin-top: .1rem; }
    .eyebrow {
        color: var(--accent-2); font-size: .72rem; font-weight: 700;
        letter-spacing: .14em; text-transform: uppercase;
    }
    .hero {
        max-width: 760px; margin: 4.2rem auto 2.4rem; text-align: center;
    }
    .hero h1 {
        margin: .55rem 0 .65rem; color: #f0f2fb;
        font: 800 clamp(2.4rem, 6vw, 4.2rem)/1.08 'Manrope', sans-serif;
        letter-spacing: -.065em;
    }
    .hero p { color: var(--muted); font-size: 1.03rem; line-height: 1.7; }
    .hero-glow {
        width: 5rem; height: .22rem; margin: 0 auto 1.55rem; border-radius: 2rem;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        box-shadow: 0 0 24px rgba(151, 137, 255, .48);
    }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--line); border-radius: 1.15rem;
        background: var(--panel); padding: 1rem 1.15rem;
        margin: .75rem auto; max-width: 820px;
        box-shadow: 0 12px 34px rgba(0, 0, 0, .12);
    }
    [data-testid="stChatInput"] {
        max-width: 820px; margin: 1.25rem auto 1rem;
    }
    [data-testid="stChatInput"] > div {
        border: 1px solid rgba(164, 151, 255, .38);
        border-radius: 1rem; background: rgba(19, 27, 46, .92);
        box-shadow: 0 8px 34px rgba(0, 0, 0, .22);
    }
    [data-testid="stChatInput"] textarea { color: var(--ink); }
    [data-testid="stChatInput"] textarea::placeholder { color: #8390a9; }
    [data-testid="stChatMessage"] p, [data-testid="stMarkdownContainer"] p {
        line-height: 1.72;
    }
    div[data-testid="stAlert"] { border-radius: .85rem; }
    .sidebar-note {
        color: var(--muted); font-size: .78rem; line-height: 1.55;
        padding: .8rem .85rem; border: 1px solid var(--line);
        border-radius: .8rem; background: rgba(255,255,255,.025);
    }
    .footer {
        color: #78849c; text-align: center; font-size: .75rem;
        padding: .6rem 0 1.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_text(content: object) -> str:
    """Convert Gemini's supported text content shapes into displayable text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text")
        return text.strip() if isinstance(text, str) else ""
    if isinstance(content, (list, tuple)):
        parts = [extract_text(part) for part in content]
        return "\n".join(part for part in parts if part)
    return ""


def history_messages(history: list[dict[str, str]], system_prompt: str) -> list[BaseMessage]:
    """Build the provider message list from validated chat history."""
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if not content.strip():
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def current_history() -> list[dict[str, str]]:
    """Return well-formed chat turns from Streamlit's per-session state."""
    stored: Any = st.session_state.get("messages", [])
    if not isinstance(stored, list):
        st.session_state["messages"] = []
        return st.session_state["messages"]

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in stored
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]
    st.session_state["messages"] = history
    return history


def request_reply(
    history: list[dict[str, str]],
    api_key: str,
    model_name: str,
    temperature: float,
    system_prompt: str,
) -> str | None:
    """Call Gemini and keep provider failures visible and recoverable."""
    try:
        model = ChatGoogleGenerativeAI(
            model=model_name.strip(),
            temperature=temperature,
            max_tokens=1024,
            api_key=api_key,
            request_timeout=60,
            retries=2,
        )
        response = model.invoke(history_messages(history, system_prompt))
        answer = extract_text(response.content)
        if not answer:
            raise ValueError("The model returned an empty response.")
    except Exception as error:
        LOGGER.exception("Gemini request failed")
        st.session_state["last_error"] = (
            f"Gemini could not complete that request ({type(error).__name__}). "
            "Check the app configuration, internet connection, and service quota, then retry."
        )
        st.session_state["last_failed"] = True
        return None

    history.append({"role": "assistant", "content": answer})
    st.session_state["last_error"] = ""
    st.session_state["last_failed"] = False
    return answer


def clear_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["last_error"] = ""
    st.session_state["last_failed"] = False


history = current_history()
st.session_state.setdefault("last_error", "")
st.session_state.setdefault("last_failed", False)

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">✦</div>
            <div>
                <div class="brand-name">Mira</div>
                <div class="brand-caption">Your thoughtful AI companion</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("＋  New conversation", use_container_width=True):
        clear_conversation()
        st.rerun()

    if history:
        transcript = "\n\n".join(
            f"{'You' if item['role'] == 'user' else APP_TITLE}:\n{item['content']}"
            for item in history
        )
        st.download_button(
            "Download conversation",
            data=transcript,
            file_name="mira-conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.markdown(
    """
    <div class="hero">
        <div class="hero-glow"></div>
        <div class="eyebrow">A little clarity, one conversation at a time</div>
        <h1>What’s on your mind?</h1>
        <p>Ask a question, explore an idea, or get help with something you’re working on.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not history:
    st.markdown(
        '<div class="footer">Start with a question below. Your conversation stays in this browser session.</div>',
        unsafe_allow_html=True,
    )

for turn in history:
    with st.chat_message(turn["role"], avatar="🧑‍💻" if turn["role"] == "user" else "✨"):
        st.markdown(turn["content"])

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])
    if st.session_state["last_failed"] and history and st.button("Retry last response"):
        with st.spinner("Mira is thinking…"):
            answer = request_reply(
                history,
                API_KEY,
                MODEL_NAME,
                TEMPERATURE,
                SYSTEM_PROMPT,
            )
        if answer:
            st.rerun()
        else:
            st.error(st.session_state["last_error"])

prompt = st.chat_input(
    "Message Mira…",
    disabled=not API_KEY,
    max_chars=20_000,
)

if not API_KEY:
    st.error("Chat is unavailable because the app has no configured API credentials.")

if prompt and prompt.strip():
    history.append({"role": "user", "content": prompt.strip()})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt.strip())
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Mira is thinking…"):
            answer = request_reply(
                history,
                API_KEY,
                MODEL_NAME,
                TEMPERATURE,
                SYSTEM_PROMPT,
            )
        if answer:
            st.markdown(answer)
        else:
            st.error(st.session_state["last_error"])
