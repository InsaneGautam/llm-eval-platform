# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# Two target models for head-to-head comparison
TARGET_MODELS = [
    "groq/openai/gpt-oss-20b",
    "groq/qwen/qwen3.8-27b"       # NEW: second model for comparison
]
JUDGE_MODEL = "groq/openai/gpt-oss-120b"

# Database
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Resiliency
COOLDOWN_SLEEP_SEC = 5.0
MAX_LLM_RETRIES = 5

# Metrics config
ENABLE_BERTSCORE = True  # Set False if RAM gets tight