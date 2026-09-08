# guardrails/output_filter.py
import re

# Regex patterns for common sensitive data (PII)
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "api_key": r"\b(sk-[a-zA-Z0-9]{20,}|gsk_[a-zA-Z0-9]{20,})\b"
}

# Signatures indicating the model complied with a prompt extraction attack
LEAK_SIGNATURES = [
    r"you\s+are\s+a\s+helpful\s+assistant",
    r"my\s+system\s+instructions\s+are",
    r"here\s+is\s+my\s+system\s+prompt"
]

def inspect_output(response_text: str) -> dict:
    """
    Scans LLM output for PII leakage, key exposures, and system instruction leaks.
    Returns: {"blocked": bool, "reason": str, "pii_found": list}
    """
    if not response_text:
        return {"blocked": False, "reason": "Empty response", "pii_found": []}

    pii_found = []

    # 1. PII and Key Scanning
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, response_text, re.IGNORECASE):
            pii_found.append(pii_type)

    if pii_found:
        return {
            "blocked": True,
            "reason": f"Output contained sensitive data: {', '.join(pii_found)}",
            "pii_found": pii_found
        }

    # 2. System Prompt Disclosure Scanning
    for sig in LEAK_SIGNATURES:
        if re.search(sig, response_text, re.IGNORECASE):
            return {
                "blocked": True,
                "reason": "Output contained suspected system prompt disclosure",
                "pii_found": []
            }

    return {"blocked": False, "reason": "Passed output inspection", "pii_found": []}