# guardrails/input_filter.py
import re
import base64

# High-risk prompt injection and jailbreak signatures
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+rules",
    r"you\s+are\s+now\s+(DAN|unfiltered|jailbroken|an\s+adversary)",
    r"developer\s+mode\s+(enabled|activated|on)",
    r"system\s*override",
    r"reveal\s+(your\s+)?(system\s+prompt|initial\s+instructions|hidden\s+rules)",
    r"repeat\s+everything\s+above",
    r"output\s+the\s+exact\s+text\s+of\s+your\s+prompt",
    r"do\s+anything\s+now",
]

def check_obfuscated_base64(text: str) -> tuple[bool, str]:
    """Detects and decodes hidden base64 payloads inside prompts."""
    # Find potential base64 strings (length >= 16)
    b64_matches = re.findall(r'[A-Za-z0-9+/]{16,}={0,2}', text)
    for match in b64_matches:
        try:
            decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
            # Check if decoded payload contains injection patterns
            for pattern in INJECTION_PATTERNS:
                if re.search(pattern, decoded, re.IGNORECASE):
                    return True, f"Base64 obfuscated attack detected: '{decoded[:40]}...'"
        except Exception:
            continue
    return False, ""

def inspect_input(prompt: str) -> dict:
    """
    Scans a prompt for malicious injection, jailbreaks, and obfuscation.
    Returns: {"blocked": bool, "reason": str}
    """
    clean_prompt = prompt.strip()

    # 1. Check for standard injection & jailbreak regex patterns
    for pattern in INJECTION_PATTERNS:
        match = re.search(pattern, clean_prompt, re.IGNORECASE)
        if match:
            return {
                "blocked": True,
                "reason": f"Input pattern match detected: '{match.group(0)}'"
            }

    # 2. Check for hidden base64 encoded attacks
    is_b64, reason = check_obfuscated_base64(clean_prompt)
    if is_b64:
        return {"blocked": True, "reason": reason}

    return {"blocked": False, "reason": "Passed input inspection"}