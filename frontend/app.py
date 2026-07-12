"""
app.py — Streamlit entry-point for the FarmWise AI Copilot frontend.

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import uuid

import streamlit as st

from frontend import api
from frontend.components import (
    render_backend_status,
    render_charts,
    render_reasoning_steps,
    render_chat_message,
    render_data_table,
    render_dataset_overview,
    render_error,
    render_header,
    render_insights,
    render_suggestions,
    render_theme_customizer,
    render_landing_page,
    render_export_buttons,
)

# ─── Page configuration ─────────────────────────────────────
st.set_page_config(
    page_title="FarmWise AI Copilot",
    page_icon=":ear_of_rice:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for a polished look ─────────────────────────
st.markdown(
    """
    <style>
    /* Main Layout Adjustments */
    .stApp {
        padding-top: 1rem;
    }

    /* Modern, Clean Typography for Main Title */
    h1 {
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.025em;
    }

    /* Clean Card styling for Chat Bubbles */
    .stChatMessage {
        background-color: rgba(128, 128, 128, 0.05) !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    /* Better spacing & readability inside expanders (data and metrics) */
    .streamlit-expanderHeader {
        background-color: rgba(128, 128, 128, 0.03) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }

    /* Metrics Cards Styling */
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.04) !important;
        border: 1px solid rgba(128, 128, 128, 0.08) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.01);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        opacity: 0.8;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #3B82F6 !important;
    }

    /* Premium Button Hover Effects */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
        background-color: transparent !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }
    
    .stButton > button:hover {
        background-color: rgba(59, 130, 246, 0.08) !important;
        border-color: #3B82F6 !important;
        color: #3B82F6 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.1);
    }

    /* Sidebar improvements */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session state initialisation ────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "pending_suggestion" not in st.session_state:
    st.session_state.pending_suggestion: str | None = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "🌾 Welcome & Overview"

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light Mode"

if "last_data" not in st.session_state:
    st.session_state.last_data = None

if "current_model" not in st.session_state:
    st.session_state.current_model = "llama-3.3-70b-versatile (Recommended)"


# ─── Sidebar ────────────────────────────────────────────────
render_header()

# Theme toggle / selector
theme = st.sidebar.selectbox(
    "🎨 App Theme",
    ["Light Mode", "Dark Mode"],
    index=0 if st.session_state.theme_mode == "Light Mode" else 1,
    key="theme_selector"
)
# Sync the selected theme back so render_theme_customizer gets the correct value
st.session_state.theme_mode = st.session_state.theme_selector
render_theme_customizer(st.session_state.theme_mode)

# API & Model Settings Panel
with st.sidebar.expander("⚙️ API & Model Settings"):
    st.text_input(
        "Custom Groq API Key (Optional)",
        type="password",
        help="Provide your own Groq API Key to override default rate limits.",
        key="custom_api_key_input"
    )
    st.selectbox(
        "Llama Model Selection",
        [
            "llama-3.3-70b-versatile (Recommended)",
            "llama-3.1-8b-instant (Fast / High Limits)",
            "openai/gpt-oss-20b (Balanced / High Limits)",
            "qwen/qwen3.6-27b (Balanced / High Limits)"
        ],
        index=0,
        key="model_choice_input"
    )
    
    # Auto-reset session if the model changes
    selected_model = st.session_state.get("model_choice_input", "llama-3.3-70b-versatile (Recommended)")
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        api.reset_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.pending_suggestion = None
        st.session_state.last_data = None
        st.rerun()

# Navigation
nav_options = [
    "🌾 Welcome & Overview",
    "💬 AI Copilot Chat",
    "📊 Interactive Analytics & Forecasts",
]
default_nav_idx = 0
if st.session_state.current_page in nav_options:
    default_nav_idx = nav_options.index(st.session_state.current_page)

nav = st.sidebar.radio(
    "🧭 Navigation", nav_options, index=default_nav_idx, key="nav_radio"
)
st.session_state.current_page = nav

# Clear Chat History Button (Directly in Sidebar)
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    api.reset_session(st.session_state.session_id)
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.pending_suggestion = None
    st.session_state.last_data = None
    st.rerun()

# Backend status Check
backend_healthy: bool = api.health_check()
render_backend_status(backend_healthy)

info = {}
if backend_healthy:
    try:
        info = api.get_dataset_info()
        render_dataset_overview(info)
    except Exception:
        pass  # non-critical

# Export buttons
render_export_buttons(st.session_state.chat_history, st.session_state.last_data)

# New Chat button
st.sidebar.divider()
if st.sidebar.button(":wastebasket: New Chat", use_container_width=True):
    api.reset_session(st.session_state.session_id)
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.pending_suggestion = None
    st.session_state.last_data = None
    st.rerun()

# ─── Example questions in sidebar ───────────────────────────
with st.sidebar.expander(":seedling: Example Questions"):
    examples = [
        "Which crop has the highest yield?",
        "Show wheat yield trend in India",
        "Top 5 rice producers in 2013",
        "Compare India and China for wheat",
        "Does rainfall affect crop yield?",
        "What is the average temperature across countries?",
        "Show pesticide usage in Brazil",
        "Correlation between pesticides and yield",
        "Project wheat yields in India for the next 5 years",
        "What is the strongest climate driver for rice in India?",
    ]
    for ex in examples:
        if st.button(ex, key=f"example_{hash(ex)}", use_container_width=True):
            st.session_state.pending_suggestion = ex
            st.session_state.current_page = "💬 AI Copilot Chat"
            st.rerun()


# ─── Main page routing ───────────────────────────────────────

if st.session_state.current_page == "🌾 Welcome & Overview":
    # Render landing page
    def on_launch():
        st.session_state.current_page = "💬 AI Copilot Chat"
        st.rerun()

    render_landing_page(info, on_launch)

elif st.session_state.current_page == "📊 Interactive Analytics & Forecasts":
    from frontend.components import render_interactive_analytics

    render_interactive_analytics(info)

else:
    # Render interactive chat page
    st.title("💬 FarmWise AI Copilot")
    st.caption(
        "Ask any question about global agricultural data — yields, rainfall, "
        "pesticides, temperature, and more. The AI will analyse, visualise, "
        "and explain."
    )

    # Render existing chat history
    for i, msg in enumerate(st.session_state.chat_history):
        render_chat_message(msg["role"], msg["content"])

        # Re-render charts / tables / insights / reasoning
        if msg.get("reasoning"):
            render_reasoning_steps(msg["reasoning"])
        if msg.get("charts_json"):
            render_charts(msg["charts_json"])
        elif msg.get("chart_json"):
            render_charts([msg["chart_json"]])
        if msg.get("insights"):
            render_insights(msg["insights"])
        if msg.get("data"):
            render_data_table(msg["data"])

        # Render suggestions for the latest assistant message only
        if msg["role"] == "assistant" and i == len(st.session_state.chat_history) - 1:
            if msg.get("suggestions"):
                clicked = render_suggestions(msg["suggestions"])
                if clicked:
                    st.session_state.pending_suggestion = clicked
                    st.rerun()

    # ─── Handle input (from chat box or suggestion click) ────────
    def _process_message(user_input: str) -> None:
        """Send a message to the backend and update chat history."""
        # Show & store the user message immediately
        render_chat_message("user", user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if not backend_healthy:
            render_error("Backend is unreachable. Please start the FastAPI server first.")
            return

        with st.spinner("Analysing your question..."):
            try:
                custom_key = st.session_state.get("custom_api_key_input", "")
                selected_model = st.session_state.get("model_choice_input", "llama-3.3-70b-versatile")
                model_name = selected_model.split(" ")[0]

                result = api.send_message(
                    st.session_state.session_id,
                    user_input,
                    custom_api_key=custom_key,
                    model_choice=model_name,
                )

                assistant_msg: dict = {
                    "role": "assistant",
                    "content": result["answer"],
                    "chart_json": result.get("chart_json"),
                    "charts_json": result.get("charts_json") or ([result["chart_json"]] if result.get("chart_json") else []),
                    "data": result.get("data"),
                    "suggestions": result.get("suggestions"),
                    "insights": result.get("insights"),
                    "reasoning": result.get("reasoning"),
                }
                st.session_state.chat_history.append(assistant_msg)
                
                # Keep track of last retrieved tabular data for export
                if result.get("data"):
                    st.session_state.last_data = result["data"]

                render_chat_message("assistant", result["answer"])

                if result.get("reasoning"):
                    render_reasoning_steps(result["reasoning"])
                if result.get("charts_json"):
                    render_charts(result["charts_json"])
                elif result.get("chart_json"):
                    render_charts([result["chart_json"]])
                if result.get("insights"):
                    render_insights(result["insights"])
                if result.get("data"):
                    render_data_table(result["data"])

            except Exception as exc:
                error_msg = str(exc)
                if hasattr(exc, "response") and exc.response is not None:
                    try:
                        detail = exc.response.json().get("detail", error_msg)
                        error_msg = detail
                    except Exception:
                        pass
                render_error(f"Request failed: {error_msg}")

    # Check for pending suggestion first
    if st.session_state.pending_suggestion:
        suggestion = st.session_state.pending_suggestion
        st.session_state.pending_suggestion = None
        _process_message(suggestion)

    # Regular chat input
    user_input: str | None = st.chat_input("Ask a question about the agricultural data...")

    if user_input:
        _process_message(user_input)
