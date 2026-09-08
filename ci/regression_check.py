# ci/regression_check.py
import sys
import pandas as pd
from storage.supabase_client import client
from config.settings import REGRESSION_TOLERANCE

def run_regression_check():
    print("🔍 RUNNING AUTOMATED LLM QUALITY REGRESSION CHECK...")
    print("=" * 65)

    # 1. Fetch evaluation history from Supabase
    res = client.table("eval_results").select("*").execute()
    data = res.data

    if not data or len(data) < 2:
        print("⚠️ Not enough evaluation runs in Supabase to perform a regression comparison.")
        print("✅ Passing check as initial baseline setup.")
        sys.exit(0)

    df = pd.DataFrame(data)

    # Convert timestamp to datetime if present, otherwise sort by ID
    df["id"] = df["id"].astype(int)
    df = df.sort_values("id")

    # Get distinct versions or run batches
    versions = df["eval_version"].unique()
    
    if len(versions) < 2:
        print("ℹ️ Only 1 eval version found. Setting current run as baseline.")
        sys.exit(0)

    baseline_version = versions[-2]
    current_version = versions[-1]

    print(f"📊 Comparing Current Run [{current_version}] vs Baseline Run [{baseline_version}]")

    # Calculate average scores
    baseline_df = df[df["eval_version"] == baseline_version]
    current_df = df[df["eval_version"] == current_version]

    metrics = ["judge_faithfulness", "bertscore_f1", "rouge_l_f1", "latency_ms"]

    b_faith = baseline_df["judge_faithfulness"].mean()
    c_faith = current_df["judge_faithfulness"].mean()

    b_bert = baseline_df["bertscore_f1"].mean()
    c_bert = current_df["bertscore_f1"].mean()

    print("\n--- BENCHMARK COMPARISON SUMMARY ---")
    print(f"Faithfulness : Baseline = {b_faith:.2f} | Current = {c_faith:.2f}")
    print(f"BERTScore    : Baseline = {b_bert:.3f} | Current = {c_bert:.3f}")

    # Calculate percentage changes
    faith_change = (c_faith - b_faith) / b_faith if b_faith > 0 else 0
    bert_change = (c_bert - b_bert) / b_bert if b_bert > 0 else 0

    has_regression = False
    reasons = []

    # Check against tolerance threshold (e.g., 0.05 = 5% drop)
    if faith_change < -REGRESSION_TOLERANCE:
        has_regression = True
        reasons.append(f"Faithfulness dropped by {abs(faith_change)*100:.1f}% (Threshold: {REGRESSION_TOLERANCE*100}%)")

    if bert_change < -REGRESSION_TOLERANCE:
        has_regression = True
        reasons.append(f"BERTScore dropped by {abs(bert_change)*100:.1f}% (Threshold: {REGRESSION_TOLERANCE*100}%)")

    print("=" * 65)
    if has_regression:
        print("❌ QUALITY REGRESSION DETECTED! BUILD FAILED.")
        for r in reasons:
            print(f"   • {r}")
        print("=" * 65)
        sys.exit(1)  # Exits with non-zero code to fail GitHub Actions workflow
    else:
        print("✅ QUALITY CHECKS PASSED. NO SIGNIFICANT REGRESSION DETECTED.")
        print("=" * 65)
        sys.exit(0)

if __name__ == "__main__":
    run_regression_check()