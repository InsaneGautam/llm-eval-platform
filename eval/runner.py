# eval/runner.py
import os
import logging
os.environ["LITELLM_LOG"] = "ERROR"           # Suppress LiteLLM info spam
logging.getLogger("transformers").setLevel(logging.ERROR)  # Suppress Roberta warnings
import time
from litellm import completion
from litellm.exceptions import RateLimitError
from eval.llm_judge import judge_response
from eval.metrics import compute_all_programmatic
from storage.supabase_client import get_test_cases, insert_eval_result
from config.settings import TARGET_MODELS, COOLDOWN_SLEEP_SEC, MAX_LLM_RETRIES
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=3, max=30),
    stop=stop_after_attempt(MAX_LLM_RETRIES),
    reraise=True
)
def _generate_target_response(model, prompt):
    return completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

def run_eval_suite():
    test_cases = get_test_cases()
    if not test_cases:
        print("No test cases found. Exiting.")
        return

    print(f"Loaded {len(test_cases)} test cases from Supabase.")
    print(f"Target models: {[m.split('/')[-1] for m in TARGET_MODELS]}")
    print("Beginning evaluation run...\n")

    for idx, tc in enumerate(test_cases, 1):
        print(f"[{idx}/{len(test_cases)}] '{tc['prompt'][:50]}...'")

        for model in TARGET_MODELS:
            try:
                # 1. Get model response
                start = time.time()
                response = _generate_target_response(model, tc["prompt"])
                latency = int((time.time() - start) * 1000)
                model_answer = response.choices[0].message.content
                time.sleep(COOLDOWN_SLEEP_SEC)

                # 2. LLM Judge scoring (API call)
                judge_scores = judge_response(
                    tc["prompt"],
                    tc.get("reference_answer", ""),
                    model_answer
                )
                time.sleep(COOLDOWN_SLEEP_SEC)

                # 3. Programmatic metrics (local, free)
                prog_scores = compute_all_programmatic(
                    tc.get("reference_answer", ""),
                    model_answer
                )

                # 4. Store everything
                result = {
                    "test_case_id": tc["id"],
                    "model_name": model,
                    "model_response": model_answer,
                    "judge_faithfulness": judge_scores["faithfulness"],
                    "judge_relevance": judge_scores["relevance"],
                    "judge_coherence": judge_scores["coherence"],
                    "rouge_l_f1": prog_scores["rouge_l_f1"],
                    "bertscore_f1": prog_scores["bertscore_f1"],
                    "latency_ms": latency,
                    "tokens_used": response.usage.total_tokens,
                    "eval_version": "v1.1"
                }
                insert_eval_result(result)

                # Pretty print
                model_short = model.split("/")[-1]
                print(f"  ✓ {model_short}")
                print(f"    Judge  → Faith:{judge_scores['faithfulness']} Rel:{judge_scores['relevance']} Coh:{judge_scores['coherence']}")
                print(f"    ROUGE  → {prog_scores['rouge_l_f1']}")
                print(f"    BERT   → {prog_scores['bertscore_f1']}")
                print(f"    Speed  → {latency}ms\n")

                time.sleep(COOLDOWN_SLEEP_SEC)

            except Exception as e:
                print(f"  ❌ {model}: {str(e)}\n")
                time.sleep(COOLDOWN_SLEEP_SEC * 2)

    print("Evaluation suite complete!")

if __name__ == "__main__":
    run_eval_suite()