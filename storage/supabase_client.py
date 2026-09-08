# storage/supabase_client.py
from supabase import create_client
from config.settings import SUPABASE_URL, SUPABASE_KEY

client = create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_test_case(prompt, ref_answer, context, category, attack_type="none"):
    data = {
        "prompt": prompt,
        "reference_answer": ref_answer,
        "source_context": context,
        "category": category,
        "attack_type": attack_type,
        "version": "v2.0"  # New version tag for synthetic data
    }
    client.table("test_cases").insert(data).execute()

def get_test_cases(version=None):
    query = client.table("test_cases").select("*")
    if version:
        query = query.eq("version", version)
    response = query.execute()
    return response.data

def insert_eval_result(result):
    client.table("eval_results").insert(result).execute()

def get_eval_results(version=None):
    query = client.table("eval_results").select("*")
    if version:
        query = query.eq("eval_version", version)
    response = query.execute()
    return response.data