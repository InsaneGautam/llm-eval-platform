# eval/metrics.py
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc

# Initialize ROUGE scorer once (reused across all calls)
_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_rouge(reference: str, prediction: str) -> float:
    """
    ROUGE-L F1 score: measures longest common subsequence overlap
    between reference and prediction.
    Range: 0.0 (no overlap) to 1.0 (perfect match)
    Runs locally in <5ms. Zero API cost.
    """
    if not reference.strip() or not prediction.strip():
        return 0.0
    scores = _rouge_scorer.score(reference, prediction)
    return round(scores["rougeL"].fmeasure, 4)

def compute_bertscore(reference: str, prediction: str) -> float:
    """
    BERTScore F1: measures semantic similarity using contextual embeddings.
    Unlike ROUGE (word overlap), BERTScore understands synonyms and
    paraphrasing. e.g., "car" and "automobile" score high.
    Range: 0.0 to 1.0
    Uses roberta-base model (~500MB RAM, runs on CPU).
    """
    if not reference.strip() or not prediction.strip():
        return 0.0
    try:
        P, R, F1 = bert_score_calc(
            [prediction],
            [reference],
            model_type="roberta-base",  # Lightweight model for 8GB RAM
            lang="en",
            verbose=False
        )
        return round(F1.item(), 4)
    except Exception as e:
        print(f"  [WARN] BERTScore failed: {e}. Returning 0.0")
        return 0.0

def compute_all_programmatic(reference: str, prediction: str) -> dict:
    """Runs all programmatic metrics and returns a unified dict."""
    return {
        "rouge_l_f1": compute_rouge(reference, prediction),
        "bertscore_f1": compute_bertscore(reference, prediction)
    }