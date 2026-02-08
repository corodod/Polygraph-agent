#src/openrouter_client.py
from __future__ import annotations
import os
import json
import time
import requests
from typing import Any, Dict, Optional

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("openrouter")


OR_MODEL = os.getenv("OR_MODEL", "mistralai/mistral-7b-instruct")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def extract_json_object(text: str) -> Optional[dict]:
    """
    Finds the first valid JSON object in text.
    More robust than slicing from first '{' to last '}'.
    """
    if not text:
        return None

    # Быстрый путь: если весь текст — чистый JSON
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Поиск первого валидного {...}
    # Идем по всем позициям '{' и пробуем декодировать JSON с помощью JSONDecoder.raw_decode
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None

def _normalize_llm_obj(obj: Any) -> Optional[dict]:
    """
    Fixes common LLM JSON issues:
    - keys like '\"route\"' -> 'route'
    - values that contain JSON string -> parse it
    """
    if obj is None:
        return None

    # Если пришла строка, попробуем распарсить как JSON
    if isinstance(obj, str):
        s = obj.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                obj = parsed
        except Exception:
            return None

    if not isinstance(obj, dict):
        return None

    # Если внутри dict есть единственное строковое поле с JSON — распарсить его
    if len(obj) == 1:
        only_val = next(iter(obj.values()))
        if isinstance(only_val, str) and "{" in only_val and "}" in only_val:
            try:
                parsed = json.loads(only_val)
                if isinstance(parsed, dict):
                    obj = parsed
            except Exception:
                pass

    # Нормализуем ключи типа '"route"' или '\"route\"'
    fixed = {}
    for k, v in obj.items():
        nk = k
        if isinstance(k, str):
            nk = k.strip()
            # убираем внешние кавычки если ключ сам в кавычках
            if (nk.startswith('"') and nk.endswith('"')) or (nk.startswith("'") and nk.endswith("'")):
                nk = nk[1:-1]
            nk = nk.replace('\\"', '"')
            # если после replace снова в кавычках — убрать
            nk = nk.strip()
            if (nk.startswith('"') and nk.endswith('"')) or (nk.startswith("'") and nk.endswith("'")):
                nk = nk[1:-1]
        fixed[nk] = v

    return fixed



def _headers() -> Dict[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it into .env or export it in your shell."
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OR_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OR_APP_TITLE", "polygraph-agent"),
    }

def or_chat_completion(system_text: str, user_text: str, max_tokens: int = 400) -> str:
    print("OR: calling model =", OR_MODEL)
    payload = {
        "model": OR_MODEL,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }

    log.info("OpenRouter call: model=%s, user_preview=%s", OR_MODEL, user_text[:120].replace("\n", " "))
    r = requests.post(API_URL, headers=_headers(), json=payload, timeout=120)
    log.info("OpenRouter status=%s", r.status_code)

    if r.status_code != 200:
        log.error("OpenRouter error body=%s", r.text[:800])
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:500]}")

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    log.info("OpenRouter content preview=%s", content[:200].replace("\n", " "))
    print("OR: content preview =", content[:200])
    return content

def llm_json(system_text: str, user_text: str, *, max_tokens: int = 400, max_retries: int = 6) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            out = or_chat_completion(system_text, user_text, max_tokens=max_tokens)
            obj = extract_json_object(out)
            obj = _normalize_llm_obj(obj)
            print("OR: normalized obj =", obj, flush=True)

            print("OR: parsed obj =", obj)
            if obj is None:
                raise ValueError(f"Could not parse JSON. Model output: {out[:300]}")
            return obj
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"LLM failed after retries: {last_err}")

