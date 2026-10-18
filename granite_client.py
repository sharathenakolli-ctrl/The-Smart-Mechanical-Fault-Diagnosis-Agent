"""
IBM watsonx.ai Granite LLM client
Handles authentication and text generation via the watsonx.ai REST API.
"""
import os
import requests

# Load from environment or fall back to config.env
def _load_config():
    cfg = {}
    try:
        with open("config.env") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return cfg

_cfg = _load_config()

WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY",    _cfg.get("WATSONX_API_KEY", ""))
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", _cfg.get("WATSONX_PROJECT_ID", ""))
WATSONX_URL        = os.environ.get("WATSONX_URL",        _cfg.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"))
GRANITE_MODEL      = os.environ.get("IBM_GRANITE_MODEL",  _cfg.get("IBM_GRANITE_MODEL", "ibm/granite-13b-instruct-v2"))

_iam_token_cache = {"token": None, "expires_at": 0}


def _get_iam_token() -> str:
    """Fetch / cache an IAM bearer token."""
    import time
    now = time.time()
    if _iam_token_cache["token"] and now < _iam_token_cache["expires_at"] - 60:
        return _iam_token_cache["token"]

    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WATSONX_API_KEY,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _iam_token_cache["token"] = data["access_token"]
    _iam_token_cache["expires_at"] = now + data.get("expires_in", 3600)
    return _iam_token_cache["token"]


def granite_generate(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Send a prompt to IBM Granite via watsonx.ai and return the generated text.
    """
    token = _get_iam_token()
    url = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"

    payload = {
        "model_id": GRANITE_MODEL,
        "project_id": WATSONX_PROJECT_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": 1.1,
        },
    }

    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if results:
        return results[0].get("generated_text", "").strip()
    return ""
