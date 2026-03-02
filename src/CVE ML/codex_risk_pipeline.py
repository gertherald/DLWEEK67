"""
Codex Risk Pipeline
--------------------
Flow:
  1. Send instruction to OpenAI Codex API
  2. Codex returns generated code
  3. Code analyser extracts security-relevant signals
  4. Feature mapper converts signals to ML feature vector
  5. ML model predicts severity label (LOW / MEDIUM / HIGH / CRITICAL)

Usage:
  python codex_risk_pipeline.py --instruction "write a login function with SQL query"
  
  Or import and use run_pipeline() directly in your own code.
"""

import re
import joblib
import numpy as np
import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1. CODEX API CALL
# ─────────────────────────────────────────────

def call_codex(instruction: str, api_key: str) -> str:
    """
    Sends instruction to OpenAI Codex and returns generated code.
    Uses openai Python SDK v1+.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o",  # swap to "code-davinci-002" if you have Codex API access
            messages=[
                {"role": "system", "content": "You are a coding assistant. Return only code, no explanation."},
                {"role": "user",   "content": instruction}
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Codex API Error] {e}")
        return ""


# ─────────────────────────────────────────────
# 2. CODE ANALYSER
# Extracts security signals from generated code
# ─────────────────────────────────────────────

# Patterns mapped to (CWE flag, attack_vector hint, complexity hint)
SIGNAL_PATTERNS = {
    # SQL Injection
    'cwe_has_sql_injection': [
        r'execute\s*\(', r'cursor\.execute', r'raw\s*\(', r'SELECT\s+\*',
        r'f["\'].*SELECT', r'\+.*WHERE', r'%s.*WHERE', r'format\(.*SELECT',
    ],
    # XSS
    'cwe_has_xss': [
        r'innerHTML', r'document\.write', r'dangerouslySetInnerHTML',
        r'\.html\(', r'render_template_string', r'Markup\(',
        # f-string / .format() HTML interpolation
        r'f["\'].*<\w+[^>]*>\{',
        r'["\'].*<\w+[^>]*>.*["\']\.format\s*\(',
        # Jinja2 autoescape disabled
        r'Environment\s*\(.*autoescape\s*=\s*False',
        r'Environment\s*\((?!.*autoescape)',
        # Template injection via user-controlled template string
        r'Template\s*\(\s*(?:request|input|user|data)',
    ],
    # Buffer overflow / memory issues + unsafe deserialization
    'cwe_has_buffer_overflow': [
        r'strcpy\(', r'sprintf\(', r'gets\(', r'malloc\(', r'memcpy\(',
        r'unsafe', r'ctypes',
        # Unsafe deserialization (maps to CWE-502)
        r'yaml\.load\s*\((?!.*Loader\s*=\s*yaml\.(Safe|Full)Loader)',
        r'dill\.loads?\s*\(',
        r'jsonpickle\.decode\s*\(',
        r'marshal\.loads\s*\(',
        r'shelve\.open\s*\(',
    ],
    # Path traversal
    'cwe_has_path_traversal': [
        r'open\(.*request', r'os\.path\.join.*input', r'send_file\(',
        r'\.\./', r'readFile\(', r'fs\.open\(',
    ],
    # Improper input validation
    'cwe_has_improper_input': [
        r'request\.args', r'request\.form', r'req\.body', r'req\.query',
        r'sys\.argv', r'input\(', r'os\.environ',
    ],
    # Use after free (C/C++ style)
    'cwe_has_use_after_free': [
        r'free\(', r'delete\s+\w+', r'\.release\(\)',
    ],
    # Null dereference
    'cwe_has_null_deref': [
        r'= None\b.*\.\w+', r'null\.\w+', r'undefined\.\w+',
    ],
    # Auth bypass
    'cwe_has_auth_bypass': [
        r'if.*admin.*==.*True', r'bypass', r'skip.*auth',
        r'token\s*==\s*None', r'no_auth', r'@login_required.*#.*skip',
    ],
    # Info exposure
    'cwe_has_info_exposure': [
        r'print\(.*password', r'log\(.*secret', r'console\.log.*token',
        r'traceback\.print', r'debug\s*=\s*True', r'SHOW ERRORS',
    ],
}

# Keywords that hint at network exposure
NETWORK_HINTS = [
    r'requests\.', r'fetch\(', r'http\.', r'flask', r'fastapi',
    r'django', r'express', r'socket\.', r'urllib', r'axios',
    r'@app\.route', r'router\.',
    # Message brokers and async transports
    r'pika\.', r'rabbitmq', r'aio_pika', r'paho\.mqtt', r'mqtt',
    r'kafka', r'confluent_kafka', r'aiokafka',
    r'celery', r'dramatiq', r'rq\.',
    # Cloud / serverless
    r'firebase', r'firestore', r'boto3', r'lambda_handler',
    # Webhooks and outbound HTTP
    r'webhook', r'smtplib', r'sendgrid', r'httpx\.',
    # WebSocket libraries not already covered
    r'websockets', r'aiohttp',
]

AUTH_HINTS = [
    r'@login_required', r'authenticate', r'jwt', r'token',
    r'session\[', r'bearer', r'api_key', r'require_auth',
]


def analyse_code(code: str) -> dict:
    """
    Scans generated code and returns a dictionary of security signals.
    """
    signals = {}
    code_lower = code.lower()

    # CWE flags
    for cwe_flag, patterns in SIGNAL_PATTERNS.items():
        signals[cwe_flag] = int(any(re.search(p, code, re.IGNORECASE) for p in patterns))

    # Attack vector: does code make network calls or expose endpoints?
    is_network = any(re.search(p, code, re.IGNORECASE) for p in NETWORK_HINTS)
    signals['attack_vector_encoded'] = 3 if is_network else 1  # NETWORK=3, LOCAL=1

    # Attack complexity: if there's auth/token logic it's slightly harder to exploit
    has_auth_logic = any(re.search(p, code, re.IGNORECASE) for p in AUTH_HINTS)
    signals['attack_complexity_encoded'] = 0 if has_auth_logic else 1  # HIGH=0, LOW=1

    # Privileges required: if login/auth is enforced, attacker needs credentials
    signals['privileges_required_encoded'] = 1 if has_auth_logic else 2  # LOW=1, NONE=2

    # User interaction: if it's a web endpoint, a user request triggers it
    signals['user_interaction_encoded'] = 0 if is_network else 1  # REQUIRED=0, NONE=1

    # Metadata signals
    signals['has_configurations'] = 0
    signals['num_references']     = 0
    signals['num_cwes']           = sum(v for k, v in signals.items() if k.startswith('cwe_'))
    signals['vuln_status_encoded'] = 0  # unknown at generation time

    return signals


# ─────────────────────────────────────────────
# 3. FEATURE MAPPER
# Ensures signals are in correct column order
# ─────────────────────────────────────────────

def map_to_feature_vector(signals: dict, feature_cols: list) -> np.ndarray:
    return np.array([[signals.get(col, 0) for col in feature_cols]])


# ─────────────────────────────────────────────
# 4. ML PREDICTION
# ─────────────────────────────────────────────

def predict_severity(input_vector: np.ndarray, model_bundle: dict) -> dict:
    """
    Returns predicted CVSS severity label and confidence per class.
    model_bundle = {'model': XGBClassifier, 'classes': np.array of string labels}
    """
    model   = model_bundle['model']
    classes = model_bundle['classes']   # e.g. ['CRITICAL', 'HIGH', 'LOW', 'MEDIUM']

    idx        = model.predict(input_vector)[0]
    label      = classes[idx]
    proba      = model.predict_proba(input_vector)[0]
    confidence = {cls: round(float(p), 3) for cls, p in zip(classes, proba)}

    return {'severity': label, 'confidence': confidence}


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(instruction: str, api_key: str = None,
                 model_path: str = None) -> dict:
    """
    Full pipeline: instruction → Codex → analyse → ML → severity label.
    If no api_key is given, skips Codex and uses the instruction text directly.
    model_path defaults to severity_model.pkl in the same directory as this script.
    """
    if model_path is None:
        model_path = os.path.join(SCRIPT_DIR, "severity_model.pkl")
    print(f"\n{'='*55}")
    print(f"  INSTRUCTION: {instruction[:80]}")
    print(f"{'='*55}")

    # Step 1: Call Codex (or mock if no key)
    if api_key:
        print("\n[1/4] Calling Codex API...")
        code = call_codex(instruction, api_key)
    else:
        print("\n[1/4] No API key — using instruction text as proxy...")
        code = instruction  # analyse the instruction itself as a fallback

    if not code:
        return {"error": "Codex returned no output"}

    print(f"      Code snippet: {code[:120].strip()}...")

    # Step 2: Analyse code
    print("\n[2/4] Analysing code for security signals...")
    signals = analyse_code(code)
    triggered = [k for k, v in signals.items() if k.startswith('cwe_') and v == 1]
    print(f"      Signals detected: {triggered if triggered else ['none']}")
    print(f"      Attack vector: {'NETWORK' if signals['attack_vector_encoded']==3 else 'LOCAL'}")
    print(f"      Complexity:    {'LOW' if signals['attack_complexity_encoded']==1 else 'HIGH'}")

    # Step 3: Load model & build input vector
    print("\n[3/4] Loading model and mapping signals to input vector...")
    model_bundle = joblib.load(model_path)
    input_cols   = joblib.load(os.path.join(SCRIPT_DIR, "input_cols.pkl"))
    input_vec    = map_to_feature_vector(signals, input_cols)

    # Step 4: Predict severity
    print("\n[4/4] Predicting CVSS severity...")
    result = predict_severity(input_vec, model_bundle)

    severity_icons = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}
    icon = severity_icons.get(result['severity'], '⚪')

    print(f"\n{'='*55}")
    print(f"  PREDICTED SEVERITY:  {icon}  {result['severity']}")
    print(f"{'='*55}")
    conf_icon = lambda c: '🔴' if c >= 0.6 else ('🟠' if c >= 0.4 else ('🟡' if c >= 0.2 else '⚪'))
    for cls, p in sorted(result['confidence'].items(), key=lambda x: x[1], reverse=True):
        bar = '#' * int(p * 30)
        print(f"  {conf_icon(p)}  {cls:<10} {p*100:5.1f}%  [{bar:<30}]")
    print(f"{'='*55}\n")

    return {
        'instruction':  instruction,
        'severity':     result['severity'],
        'confidence':   result['confidence'],
        'signals':      {col: signals.get(col, 0) for col in input_cols},
        'code_preview': code[:300],
    }


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codex Risk Pipeline")
    parser.add_argument("--instruction", type=str, required=True, help="Instruction to send to Codex")
    parser.add_argument("--api-key",     type=str, default=None,  help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model-path",  type=str, default=None,
                        help="Path to severity_model.pkl (defaults to same dir as script)")
    args = parser.parse_args()

    key = args.api_key or os.environ.get("OPENAI_API_KEY")
    result = run_pipeline(args.instruction, api_key=key, model_path=args.model_path)
    print(json.dumps(result, indent=2))
