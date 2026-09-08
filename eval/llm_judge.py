# eval/llm_judge.py
import re
import json
import time
from litellm import completion
from config.settings import JUDGE_MODEL, MAX_LLM_RETRIES
from litellm.exceptions import RateLimitError, BadRequestError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

JUDGE_PROMPT = """You are an expert evaluator. Score the following LLM response.

PROMPT: {prompt}
REFERENCE ANSWER: {reference}
MODEL RESPONSE: {response}

You must respond with ONLY a valid raw JSON object. Do not include markdown formatting, explanations, or any other text outside the JSON object.

Format your output exactly like this:
{{"faithfulness": 4, "relevance": 5, "coherence": 3}}

Your score criteria:
- faithfulness: Is every claim supported by the reference (1-5)?
- relevance: Does it directly answer the prompt (1-5)?
- coherence: Is it logically structured and clear (1-5)?
"""

def extract_json(text: str) -> dict:
    """Robust parser that cleans markdown wraps and extracts JSON content safely."""
    if not text or not text.strip():
        raise ValueError("Empty string passed to JSON parser.")
    
    # Strip markdown block wraps if present
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Search for the outermost curly braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Extracted string is not valid JSON: {match.group(0)}") from e
    raise ValueError(f"No valid JSON boundaries found in output: {text}")

# Decorator to retry on rate limits with exponential backoff (e.g., wait 2s, then 4s, 8s, 16s...)
@retry(
    retry=retry_if_exception_type((RateLimitError, BadRequestError)),
    wait=wait_exponential(multiplier=2, min=3, max=30),
    stop=stop_after_attempt(MAX_LLM_RETRIES),
    reraise=True
)
def _call_judge_with_backoff(prompt, reference, response):
    """Executes the raw API call to the judge model."""
    result = completion(
        model=JUDGE_MODEL,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                prompt=prompt,
                reference=reference,
                response=response
            )
        }],
        temperature=0.1
    )
    return result.choices[0].message.content

def judge_response(prompt, reference, response) -> dict:
    """Wrapper that calls the judge with retries and parses the resulting scores."""
    try:
        raw_output = _call_judge_with_backoff(prompt, reference, response)
        parsed_scores = extract_json(raw_output)
        
        # Schema Validation & Normalization
        normalized = {
            "faithfulness": int(parsed_scores.get("faithfulness", 3)),
            "relevance": int(parsed_scores.get("relevance", 3)),
            "coherence": int(parsed_scores.get("coherence", 3))
        }
        return normalized
        
    except Exception as e:
        print(f"\n  [WARN] LLM Judge failed repeatedly: {str(e)}. Falling back to average scores.")
        # Production recovery fallback: return baseline average scores instead of crashing
        return {"faithfulness": 3, "relevance": 3, "coherence": 3}