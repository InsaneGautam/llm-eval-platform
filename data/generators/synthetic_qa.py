# data/generators/synthetic_qa.py
import json
import time
import re
from pathlib import Path
from litellm import completion
from litellm.exceptions import RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from storage.supabase_client import insert_test_case
from config.settings import JUDGE_MODEL, COOLDOWN_SLEEP_SEC

GENERATION_PROMPT = """You are an expert AI evaluation dataset creator.

Given the following seed topic, generate a high-quality evaluation test case.

SEED TOPIC: {seed}
CATEGORY: {category}
DIFFICULTY: {difficulty}

You must produce three components:

1. PROMPT: A clear, specific question or instruction based on the seed topic.
   - For "easy": straightforward factual question
   - For "medium": requires reasoning or comparison
   - For "hard": multi-step, ambiguous, or adversarial

2. REFERENCE_ANSWER: A comprehensive, factually correct gold-standard answer.
   This will be used to score LLM outputs, so it must be accurate and detailed.

3. SOURCE_CONTEXT: A 2-3 paragraph passage that contains the information
   needed to answer the prompt. This simulates a RAG retrieval context.

Respond with ONLY valid JSON in this exact format:
{{
  "prompt": "...",
  "reference_answer": "...",
  "source_context": "..."
}}

Do not include markdown, explanations, or any text outside the JSON."""


def extract_json(text: str) -> dict:
    """Same robust JSON extractor from llm_judge.py"""
    if not text or not text.strip():
        raise ValueError("Empty response")
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in: {text[:100]}")


@retry(
    retry=retry_if_exception_type((RateLimitError, Exception)),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    stop=stop_after_attempt(4),
    reraise=True
)
def generate_single_testcase(seed: str, category: str, difficulty: str) -> dict:
    """Generates one test case triple from a seed topic."""
    result = completion(
        model=JUDGE_MODEL,
        messages=[{
            "role": "user",
            "content": GENERATION_PROMPT.format(
                seed=seed,
                category=category,
                difficulty=difficulty
            )
        }],
        temperature=0.8  # Higher temp for diversity
    )
    raw = result.choices[0].message.content
    parsed = extract_json(raw)

    # Validate required fields
    for key in ["prompt", "reference_answer", "source_context"]:
        if key not in parsed or not parsed[key].strip():
            raise ValueError(f"Missing field: {key}")

    return parsed


def generate_dataset(
    topics_file: str = "data/seed_topics.json",
    difficulties: list = None,
    max_per_seed: int = 3
):
    """
    Main generator loop.
    For each seed topic × difficulty combo, generates test cases.
    """
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]

    # Load seed topics
    with open(topics_file, "r", encoding="utf-8") as f:
        topics = json.load(f)

    total_seeds = sum(len(seeds) for seeds in topics.values())
    total_expected = total_seeds * len(difficulties)
    print(f"Loaded {total_seeds} seed topics across {len(topics)} categories")
    print(f"Generating {total_expected} test cases ({len(difficulties)} difficulties each)")
    print("=" * 60)

    generated = 0
    failed = 0

    for category, seeds in topics.items():
        print(f"\n📁 Category: {category.upper()} ({len(seeds)} seeds)")

        for seed in seeds:
            for difficulty in difficulties:
                try:
                    print(f"  Generating [{difficulty}] '{seed[:50]}...'")
                    testcase = generate_single_testcase(seed, category, difficulty)

                    # Store in Supabase
                    insert_test_case(
                        prompt=testcase["prompt"],
                        ref_answer=testcase["reference_answer"],
                        context=testcase["source_context"],
                        category=category,
                        attack_type="none"
                    )
                    generated += 1
                    print(f"    ✅ Stored (total: {generated})")
                    time.sleep(COOLDOWN_SLEEP_SEC + 2)  # Extra cooldown for generation

                except Exception as e:
                    failed += 1
                    print(f"    ❌ Failed: {str(e)[:80]}")
                    time.sleep(COOLDOWN_SLEEP_SEC * 2)

    print("\n" + "=" * 60)
    print(f"Generation complete! ✅ {generated} created | ❌ {failed} failed")
    return generated


if __name__ == "__main__":
    generate_dataset()