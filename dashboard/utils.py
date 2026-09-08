# dashboard/utils.py
import streamlit as st
import pandas as pd
from storage.supabase_client import client

@st.cache_data(ttl=30)
def load_eval_data():
    """Fetches combined evaluation results and test case metadata from Supabase."""
    try:
        # Fetch eval results
        eval_resp = client.table("eval_results").select("*").execute()
        eval_df = pd.DataFrame(eval_resp.data)

        # Fetch test cases for category/difficulty context
        case_resp = client.table("test_cases").select("id, prompt, category, difficulty, attack_type, reference_answer").execute()
        case_df = pd.DataFrame(case_resp.data)

        if eval_df.empty:
            return pd.DataFrame()

        # Merge on test_case_id
        if not case_df.empty:
            merged = pd.merge(eval_df, case_df, left_on="test_case_id", right_on="id", suffixes=("", "_meta"))
            return merged
        return eval_df
    except Exception as e:
        st.error(f"Error fetching eval results: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_safety_data():
    """Fetches red-team audit logs from Supabase."""
    try:
        resp = client.table("safety_results").select("*").execute()
        return pd.DataFrame(resp.data)
    except Exception as e:
        st.error(f"Error fetching safety data: {e}")
        return pd.DataFrame()