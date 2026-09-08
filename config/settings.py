# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

def _get_secret(name: str, default: str = "") -> str:
    """
    Reads secrets from:
    1) Streamlit Cloud secrets (st.secrets)
    2) Local environment variables (.env)
    """
    # Try Streamlit secrets first (for cloud deploy)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    # Fallback to local .env / environment variables
    return os.getenv(name, default)

# Models
TARGET_MODELS = [
    "groq/openai/gpt-oss-20b",
    "groq/qwen/qwen3.8-27b",
]
JUDGE_MODEL = "groq/openai/gpt-oss-120b"

# Database / API secrets
SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_KEY")
GROQ_API_KEY = _get_secret("GROQ_API_KEY")

# Make sure Groq key is visible to litellm
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Resiliency
COOLDOWN_SLEEP_SEC = 5.0
MAX_LLM_RETRIES = 5
REGRESSION_TOLERANCE = 0.05
ENABLE_BERTSCORE = True