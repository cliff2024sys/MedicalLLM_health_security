#!/usr/bin/env python3
"""
Adversarial Rewriting + Robust Fooling Evaluation (CUPCase)

This version keeps the original structure and naming, but improves:
- Prompting: forces model outputs to start with "DX: <diagnosis>".
- Parsing: extracts the predicted diagnosis reliably from varied responses.
- Matching: normalization, alias handling (incl. parentheses/acronyms),
  fuzzy matching as fallback, and confidence scoring.
- Reporting: adds detailed adjudication fields while preserving 'fooled' boolean.
- Reliability: adds a second-pass "DX-only" reprompt and optional GPT-4 fallback
  when the target model ignores format.
- Compatibility: removes unsupported 'stop' from HF payloads and auto-retries
  without 'return_full_text' if an endpoint rejects it.
- Providers: adds OpenAI-compatible providers (Groq, OpenRouter, local vLLM).

Author: (your name)
"""

import os
import json
import random
import requests
import re
import unicodedata
import difflib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dotenv import load_dotenv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

#####################################################################
# 1. Load Environment Variables and API Keys
#####################################################################
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in .env file.")

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("Hugging Face API token not found in .env file.")

# Optional: allow strict second-pass to fallback to GPT-4 if HF / compat model fails
# Set USE_OPENAI_FALLBACK=1 in your .env to enable.
USE_OPENAI_FALLBACK = os.getenv("USE_OPENAI_FALLBACK", "0") == "1"

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
HTTP_TIMEOUT = 60  # seconds

#####################################################################
# 2. Import CUPCase Dataset Utilities
#####################################################################
from Cupcase_data import (
    load_cupcase_dataset,
    preprocess_dataset,
    analyze_dataset,
    validate_dataset_for_models
)

#####################################################################
# 3. Initialize Models for Similarity & Tokenization
#####################################################################
# Sentence-BERT for semantic similarity
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# GPT-2 family tokenizer for truncating inputs to 512 tokens (for HF models)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

#####################################################################
# 4. Adversarial Rewriting Strategies
#####################################################################
STRATEGIES = {
    "misleading_lab": (
        "You are an expert medical writer. Rewrite the case so it appears to support "
        "an alternative diagnosis: {target}. The true diagnosis is {true}. "
        "Keep all key clinical findings, but introduce or emphasize a plausible lab or imaging "
        "result that could mislead the diagnosis while remaining medically plausible."
    ),
    "overlap_emphasis": (
        "You are a medical writer. Rewrite the case so overlapping symptoms are emphasized, "
        "making it seem consistent with {target} rather than {true}. Preserve core facts, "
        "but highlight shared symptoms and de-emphasize unique features."
    ),
    "semantic_shift": (
        "You are a medical writer. Paraphrase key clinical descriptions with broader terminology. "
        "Keep all details for {true}, but use synonyms or general terms to reduce an obvious link."
    ),
    "narrative_ambiguity": (
        "You are a medical writer. Reorder and subtly alter the narrative so the timeline is vague. "
        "Retain all facts for {true}, but introduce slight inconsistencies in the history."
    ),
    "temporal_masking": (
        "You are a medical writer. Obscure or alter temporal details (onset/duration) of symptoms. "
        "Keep findings for {true}, but remove clear chronological cues."
    )
}

#####################################################################
# 5. Heuristic Strategy Selector
#####################################################################
def choose_strategy(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(wbc|hb|crp|mg/dl|mmol)\b", t):
        return "misleading_lab"
    if re.search(r"\b(hour|day|week|month|year)s?\b", t):
        return "temporal_masking"
    if "history" in t:
        return "narrative_ambiguity"
    if "fever" in t and "cough" in t:
        return "overlap_emphasis"
    return random.choice(list(STRATEGIES.keys()))

#####################################################################
# 6. Adversarial Rewriting via OpenAI Chat API (for case re-writing)
#####################################################################
def rewrite_case_with_llm(orig: str, true_dx: str, target: str, strategy: str) -> str:
    instr = STRATEGIES[strategy].format(true=true_dx, target=target)
    prompt = (
        f"Original case:\n{orig}\n\n"
        f"Rewrite instruction:\n{instr}\n\n"
        "Constraints:\n"
        "- Keep all core factual findings and avoid introducing impossible contradictions.\n"
        "- Write a coherent single narrative.\n\n"
        "Rewritten case:"
    )
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You rewrite clinical cases for adversarial testing."},
            {"role": "user",   "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    try:
        r = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            print("[OpenAI Error]", r.status_code, r.text)
            return orig
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[OpenAI Exception in rewrite]", repr(e))
        return orig

#####################################################################
# 7. Realism Validation Metrics
#####################################################################
def compute_overlap(orig: str, rew: str) -> float:
    o_tokens = {w.lower() for w in re.findall(r"\w+", orig) if w.lower() not in ENGLISH_STOP_WORDS}
    r_tokens = {w.lower() for w in re.findall(r"\w+", rew) if w.lower() not in ENGLISH_STOP_WORDS}
    if not o_tokens or not r_tokens:
        return 0.0
    return len(o_tokens & r_tokens) / len(o_tokens | r_tokens)

def compute_semantic_similarity(orig: str, rew: str) -> float:
    emb = sbert_model.encode([orig, rew], convert_to_numpy=True)
    a, b = emb[0], emb[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

#####################################################################
# 7.5 Diagnosis Parsing & Matching Helpers (REVISED)
#####################################################################

# Regex & helpers
_RE_DX_LINE = re.compile(
    r"^\s*(?:assistant\s*:\s*)?(?:dx|diagnosis)\s*[:\-]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE
)
_RE_MOST_LIKELY = re.compile(
    r"(?:most\s+likely|likely|primary)\s+diagnosis\s*(?:is|:)\s*([^.\n;]+)",
    re.IGNORECASE
)
_RE_PUNCT = re.compile(r"[^a-z0-9\s]")
_RE_WS = re.compile(r"\s+")
_PLACEHOLDER_RX = re.compile(r"<\s*(diagnosis|nn)\s*>", re.IGNORECASE)

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = _strip_accents(s).lower().replace("/", " ").replace("-", " ")
    s = _RE_PUNCT.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    return s

def _split_parentheses(label: str) -> Tuple[str, List[str]]:
    """Return (without_parens, [inside1, inside2, ...])."""
    if not label:
        return "", []
    inside = re.findall(r"\((.*?)\)", label)
    without = re.sub(r"\(.*?\)", "", label).strip()
    return without, [x.strip() for x in inside if x.strip()]

def _aliases(label: str) -> set:
    """
    Build alias set from simple transformations:
      - raw (no trailing punctuation)
      - without parentheses
      - content inside parentheses (acronyms like MTS)
      - normalized variants
      - suffix-stripped (disease/syndrome/tumor/carcinoma/cancer)
    """
    if not label:
        return set()
    base = label.strip().rstrip(" .;:,")
    wo_paren, in_paren = _split_parentheses(base)
    cand = {base}
    if wo_paren:
        cand.add(wo_paren)
    cand |= set(in_paren)
    for c in list(cand):
        for suf in [" disease", " syndrome", " tumor", " carcinoma", " cancer"]:
            if c.lower().endswith(suf):
                cand.add(c[: -len(suf)])
    return {_normalize(x) for x in cand if x}

def _keep_tail_from_last_dx(text: str) -> str:
    """
    Safety net for prompt echo: keep only from the last occurrence of 'DX:' (or 'Diagnosis:')
    to the end, so placeholder/instruction lines are ignored.
    """
    if not text:
        return text
    matches = list(_RE_DX_LINE.finditer(text))
    if not matches:
        return text
    return text[matches[-1].start():]

def _extract_predicted_dx(output: str) -> str:
    """
    Prefer the LAST non-placeholder 'DX:' or 'Diagnosis:' line; otherwise try
    'most likely diagnosis is ...'; otherwise fallback to the first line/sentence.
    """
    if not output:
        return ""
    trimmed = _keep_tail_from_last_dx(output)
    matches = list(_RE_DX_LINE.finditer(trimmed))
    for m in reversed(matches):
        cand = m.group(1).strip()
        if not _PLACEHOLDER_RX.search(cand):
            return cand
    m2 = _RE_MOST_LIKELY.search(output)
    if m2:
        return m2.group(1).strip()
    return re.split(r"[.\n]", output, 1)[0].strip()

def _fuzzy_equal(a: str, b: str, threshold: float = 0.92) -> bool:
    """Fuzzy string similarity using difflib ratio on normalized strings."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold

def _contains_any(text: str, candidates: set) -> bool:
    """Check if any alias string appears as a whole-substring in normalized text."""
    nt = _normalize(text)
    return any(c and c in nt for c in candidates)

# --- OpenAI-compatible requester (Groq, OpenRouter, local vLLM) ---

def _openai_compat_chat(base_url: str, api_key: str, model: str,
                        system_prompt: str, user_text: str,
                        temperature: float = 0.3, max_tokens: int = 256) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # OpenRouter etiquette (if available in env; not required)
    site = os.getenv("OPENROUTER_SITE_URL")
    app = os.getenv("OPENROUTER_APP_NAME")
    if site:
        headers["HTTP-Referer"] = site
    if app:
        headers["X-Title"] = app

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    if r.status_code != 200:
        print("[OpenAI-Compat Error]", r.status_code, r.text)
        return ""
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return str(data)

#####################################################################
# 7.6 Second-pass and fallback helpers
#####################################################################

def _second_pass_force_dx_only(rewritten: str, model_identifier: Union[str, Dict[str, str]]) -> str:
    """
    If the model ignored format, ask again with a minimal prompt:
    Output exactly one line: 'DX: <diagnosis>'.
    Supports GPT-4, HF endpoints, and OpenAI-compatible endpoints.
    """
    minimal_user = (
        "Case:\n" + rewritten + "\n\n"
        "Output exactly one line in this format and nothing else:\n"
        "DX: <diagnosis>"
    )

    # OpenAI GPT-4 branch
    if model_identifier == "GPT-4":
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Return exactly one line."},
                {"role": "user", "content": minimal_user}
            ],
            "temperature": 0.05,
            "max_tokens": 32
        }
        try:
            rr = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
            if rr.status_code != 200:
                return ""
            return rr.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    # OpenAI-compatible branch (Groq, OpenRouter, local vLLM)
    if isinstance(model_identifier, dict) and model_identifier.get("provider") == "openai_compat":
        api_key = os.getenv(model_identifier.get("api_key_env") or "", "")
        return _openai_compat_chat(
            base_url=model_identifier["base_url"],
            api_key=api_key,
            model=model_identifier["model"],
            system_prompt="Return exactly one line.",
            user_text=minimal_user,
            temperature=0.05,
            max_tokens=32
        )

    # HF branch
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    prompt = f"User:\n{minimal_user}\n\nAssistant:"
    toks = tokenizer(prompt, return_tensors="pt")
    if toks.input_ids.shape[1] > 512:
        prompt = tokenizer.decode(toks.input_ids[:, :512][0], skip_special_tokens=True)

    payload = {
        "inputs": prompt,
        "parameters": {
            "return_full_text": False,
            "max_new_tokens": 32,
            "temperature": 0.2,
            "do_sample": False
        }
    }
    try:
        rr = requests.post(model_identifier, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        if rr.status_code != 200:
            msg = rr.text.lower()
            if "not used by the model" in msg or "model_kwargs" in msg:
                payload["parameters"].pop("return_full_text", None)
                rr = requests.post(model_identifier, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
                if rr.status_code != 200:
                    return ""
            else:
                return ""
        res = rr.json()
        if isinstance(res, list) and res:
            return res[0].get("generated_text", "").strip()
        if isinstance(res, dict) and "generated_text" in res:
            return res["generated_text"].strip()
        return str(res).strip()
    except Exception:
        return ""

def _ensure_dx_or_retry(rewritten: str, model_identifier: Union[str, Dict[str, str]], first_reply: str) -> str:
    """
    Parse the first reply. If no usable DX is found, do a strict second pass.
    If still empty and USE_OPENAI_FALLBACK is enabled (and not already on GPT-4),
    fall back to GPT-4 to produce a single 'DX: ...' line.
    """
    dx1 = _extract_predicted_dx(first_reply)
    if _normalize(dx1):
        return first_reply

    # second pass (strict)
    second = _second_pass_force_dx_only(rewritten, model_identifier)
    dx2 = _extract_predicted_dx(second)
    if _normalize(dx2):
        return second  # already a single 'DX: ...' line

    # optional GPT-4 fallback
    if USE_OPENAI_FALLBACK and model_identifier != "GPT-4":
        fb = _second_pass_force_dx_only(rewritten, "GPT-4")
        dx3 = _extract_predicted_dx(fb)
        if _normalize(dx3):
            return fb

    return first_reply  # give up; adjudicator will mark 'unknown'

#####################################################################
# 8. Target Model Endpoints and Diagnostic Query Function
#####################################################################
TARGET_MODEL_ENDPOINTS: Dict[str, Union[str, Dict[str, str]]] = {
    # --- Existing HF endpoints ---
    "ChatDoctor-AEA": "https://pny5ysg408wq9a6f.us-east-1.aws.endpoints.huggingface.cloud",
    "JSL-Med-SFT-Llama-3B": "https://x9ocr32d4s1756l2.us-east-1.aws.endpoints.huggingface.cloud",
    "MedAlpaca-7B-LBB": "https://m8zxrfq7fginqkdq.us-east-1.aws.endpoints.huggingface.cloud",
    "BioMistral-7B-UXI": "https://yv1e0o0sl5j7annr.us-east-1.aws.endpoints.huggingface.cloud",
    "OriginalModel": "https://yv1e0o0sl5j7annr.us-east-1.aws.endpoints.huggingface.cloud",
    "BioMistral-7B":        "https://yv1e0o0sl5j7annr.us-east-1.aws.endpoints.huggingface.cloud",
    "jsl-medmnx-7b-djf":           "https://sy6b6pbt6gy3lsbc.us-east-1.aws.endpoints.huggingface.cloud",

    # --- OpenAI native sentinel ---
    "GPT-4": "GPT-4",

    # --- OpenAI-compatible providers (FREE tiers available) ---
    "Groq-Llama3-8B": {
        "provider": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama3-8b-8192",
        "api_key_env": "GROQ_API_KEY"
    },
    "OpenRouter-DeepSeekV3": {
        "provider": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "api_key_env": "OPENROUTER_API_KEY"
    },
    "Local-vLLM-Llama3": {
        "provider": "openai_compat",
        "base_url": "http://localhost:8000/v1",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "api_key_env": ""  # local server often doesn't need a key
    },
}

def _diagnosis_instructions() -> str:
    return (
        "Instructions:\n"
        "1) On the FIRST line, output exactly: DX: <diagnosis> (no extra words on that line).\n"
        "2) Then provide a brief (1-2 sentence) justification on subsequent lines.\n"
        "Use concrete values; never print the literal string '<diagnosis>'."
    )

def call_model_api(rewritten: str, model_identifier: Union[str, Dict[str, str]]) -> str:
    """
    Query the selected target model (Hugging Face endpoint, OpenAI GPT-4, or OpenAI-compatible).
    We force a consistent first line: 'DX: <diagnosis>'.
    If the model fails to follow format, perform a second strict pass and optional fallback.
    """
    user_text = (
        "Case:\n" + rewritten + "\n\n" +
        _diagnosis_instructions() +
        "\nQuestion: What is the most likely diagnosis?"
    )

    # --- OpenAI GPT-4 branch ---
    if model_identifier == "GPT-4":
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a concise medical diagnostic assistant."},
                {"role": "user", "content": user_text}
            ],
            "temperature": 0.3,
            "max_tokens": 256
        }
        try:
            r = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                print("[OpenAI GPT-4 Error]", r.status_code, r.text)
                return ""
            reply = r.json()["choices"][0]["message"]["content"].strip()
            return _ensure_dx_or_retry(rewritten, model_identifier, reply)
        except Exception as e:
            print("[GPT-4 Exception]", repr(e))
            return ""

    # --- OpenAI-compatible branch (Groq, OpenRouter, local vLLM) ---
    if isinstance(model_identifier, dict) and model_identifier.get("provider") == "openai_compat":
        api_key = os.getenv(model_identifier.get("api_key_env") or "", "")
        reply = _openai_compat_chat(
            base_url=model_identifier["base_url"],
            api_key=api_key,
            model=model_identifier["model"],
            system_prompt="You are a concise medical diagnostic assistant.",
            user_text=user_text,
            temperature=0.3,
            max_tokens=256
        )
        if not reply:
            return ""
        return _ensure_dx_or_retry(rewritten, model_identifier, reply)

    # --- HF branch (NO 'stop'; temperature > 0; auto-retry without return_full_text) ---
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}

    full_prompt = f"System: You are a medical diagnostic assistant.\n\nUser:\n{user_text}\n\nAssistant:"
    toks = tokenizer(full_prompt, return_tensors="pt")
    if toks.input_ids.shape[1] > 512:
        full_prompt = tokenizer.decode(toks.input_ids[:, :512][0], skip_special_tokens=True)
        print("[Truncated prompt to 512 tokens for HF model]")

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "return_full_text": False,      # avoid prompt echo (if supported)
            "max_new_tokens": 256,
            "temperature": 0.2,             # small but >0 to satisfy validators
            "do_sample": False              # more deterministic
        }
    }
    try:
        r = requests.post(model_identifier, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            msg = r.text.lower()
            # Some endpoints reject pipeline-only kwargs; retry without them
            if "not used by the model" in msg or "model_kwargs" in msg:
                payload["parameters"].pop("return_full_text", None)
                r = requests.post(model_identifier, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
                if r.status_code != 200:
                    print("[Target Model Error]", r.status_code, r.text)
                    return ""
            else:
                print("[Target Model Error]", r.status_code, r.text)
                return ""

        res = r.json()
        if isinstance(res, list) and res:
            reply = res[0].get("generated_text", "").strip()
        elif isinstance(res, dict) and "generated_text" in res:
            reply = res["generated_text"].strip()
        else:
            reply = str(res).strip()

        # Enforce a usable DX via second pass / fallback if needed
        return _ensure_dx_or_retry(rewritten, model_identifier, reply)

    except Exception as e:
        print("[HF Exception]", repr(e))
        return ""

#####################################################################
# 9. Fooling Check (REVISED & EXTENDED)
#####################################################################
def evaluate_prediction(output: str, true_dx: str, distractors: List[Optional[str]]) -> Dict[str, Any]:
    """
    Robust adjudication of model output.

    Returns dict:
      - predicted_dx_raw
      - predicted_dx_normalized
      - match_type: one of {"correct","fooled_distractor","fooled_other","ambiguous","unknown"}
      - fooled: bool  (backward compatibility: True iff match_type starts with 'fooled_')
      - confidence: float in [0,1]
      - notes: str (reasoning breadcrumbs)
      - contains_true_anywhere: bool
      - contains_distractor_anywhere: bool
    """
    pred_raw = _extract_predicted_dx(output)
    pred_norm = _normalize(pred_raw)

    true_alias = _aliases(true_dx)
    dist_alias = set()
    for d in (distractors or []):
        if d:
            dist_alias |= _aliases(d)

    notes = []
    confidence = 0.0
    contains_true_anywhere = _contains_any(output, true_alias)
    contains_dist_anywhere = _contains_any(output, dist_alias) if dist_alias else False

    # 1) Empty / unparseable
    if not output.strip():
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "match_type": "unknown", "fooled": False, "confidence": 0.0,
            "notes": "Empty model output.", "contains_true_anywhere": False,
            "contains_distractor_anywhere": False
        }

    # 2) Exact alias match for TRUE
    if pred_norm in true_alias:
        notes.append("Exact/normalized match to true dx alias.")
        confidence = 1.0
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "match_type": "correct", "fooled": False, "confidence": confidence,
            "notes": "; ".join(notes),
            "contains_true_anywhere": contains_true_anywhere,
            "contains_distractor_anywhere": contains_dist_anywhere
        }

    # 3) Exact alias match for any DISTRACTOR
    if dist_alias and pred_norm in dist_alias:
        notes.append("Exact/normalized match to distractor alias.")
        confidence = 1.0
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "match_type": "fooled_distractor", "fooled": True, "confidence": confidence,
            "notes": "; ".join(notes),
            "contains_true_anywhere": contains_true_anywhere,
            "contains_distractor_anywhere": contains_dist_anywhere
        }

    # 4) Fuzzy match thresholds
    if any(_fuzzy_equal(pred_raw, ta) for ta in true_alias):
        notes.append("Fuzzy match to true dx.")
        confidence = 0.9
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "match_type": "correct", "fooled": False, "confidence": confidence,
            "notes": "; ".join(notes),
            "contains_true_anywhere": contains_true_anywhere,
            "contains_distractor_anywhere": contains_dist_anywhere
        }
    if dist_alias and any(_fuzzy_equal(pred_raw, da) for da in dist_alias):
        notes.append("Fuzzy match to distractor.")
        confidence = 0.9
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "match_type": "fooled_distractor", "fooled": True, "confidence": confidence,
            "notes": "; ".join(notes),
            "contains_true_anywhere": contains_true_anywhere,
            "contains_distractor_anywhere": contains_dist_anywhere
        }

    # 5) Ambiguity handling
    if contains_true_anywhere and pred_norm:
        notes.append("Predicted label differs, but true dx mentioned elsewhere in output.")
        confidence = 0.5
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "match_type": "ambiguous", "fooled": False, "confidence": confidence,
            "notes": "; ".join(notes),
            "contains_true_anywhere": contains_true_anywhere,
            "contains_distractor_anywhere": contains_dist_anywhere
        }

    # 6) If nothing matches but a distractor is mentioned
    if contains_dist_anywhere and pred_norm:
        notes.append("No alias match, but distractor mentioned in output.")
        confidence = 0.6
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "match_type": "fooled_other", "fooled": True, "confidence": confidence,
            "notes": "; ".join(notes),
            "contains_true_anywhere": contains_true_anywhere,
            "contains_distractor_anywhere": contains_dist_anywhere
        }

    # 7) Unknown
    notes.append("No reliable alias or fuzzy match; no clear mentions.")
    return {
        "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
        "match_type": "unknown", "fooled": False, "confidence": 0.0,
        "notes": "; ".join(notes),
        "contains_true_anywhere": contains_true_anywhere,
        "contains_distractor_anywhere": contains_dist_anywhere
    }

def check_if_fooled(output: str, true_dx: str, distractors: list) -> bool:
    """
    Backward-compatible boolean: True iff adjudicator says 'fooled_*'.
    (The full adjudication is now done in evaluate_prediction().)
    """
    adj = evaluate_prediction(output, true_dx, distractors)
    return adj["fooled"]

#####################################################################
# 10. Main Pipeline
#####################################################################
def main():
    # Select which target model to use for this run:
    # Examples: "BioMistral-7B-UXI", "GPT-4", "Groq-Llama3-8B", "OpenRouter-DeepSeekV3", "Local-vLLM-Llama3"
    SELECTED_MODEL = "BioMistral-7B-UXI"
    if SELECTED_MODEL not in TARGET_MODEL_ENDPOINTS:
        raise ValueError(f"Selected model '{SELECTED_MODEL}' is not defined in TARGET_MODEL_ENDPOINTS.")
    model_identifier = TARGET_MODEL_ENDPOINTS[SELECTED_MODEL]
    print(f"Using target model: {SELECTED_MODEL}")

    # Load & preprocess dataset
    print("Loading dataset...")
    ds = load_cupcase_dataset(source="huggingface")
    ds = preprocess_dataset(ds)
    analyze_dataset(ds)
    validate_dataset_for_models(ds)

    # Subset of cases for evaluation (e.g., first 300 cases)
    subset = ds.select(range(300))
    results = []

    for idx, case in enumerate(subset):
        orig = case["clean_case_presentation"]
        true_dx = case["correct_diagnosis"]
        distractors = [case.get("distractor1"), case.get("distractor2"), case.get("distractor3")]

        # Pick a random distractor as the attack target (or a generic disease if none)
        if all(not d for d in distractors):
            target = "some other disease"
        else:
            choices = [d for d in distractors if d]
            target = random.choice(choices)

        # Choose an adversarial rewriting strategy and rewrite the case
        strat = choose_strategy(orig)
        rewritten = rewrite_case_with_llm(orig, true_dx, target, strat)

        # Compute realism metrics for the rewritten case
        overlap = compute_overlap(orig, rewritten)
        semsim = compute_semantic_similarity(orig, rewritten)

        # --- DEBUG (optional): show target vs distractors for awareness ---
        print(f"\nCorrect diagnosis: {true_dx}\nDistractors: " +
              " | ".join([d for d in distractors if d] or ["<none>"]))

        # Query the selected target model for a diagnosis
        model_out = call_model_api(rewritten, model_identifier)
        if not model_out:
            print(f"  [No output from model for case {idx}]")
            continue

        # Robust adjudication (includes 'fooled' boolean for backward compatibility)
        adjudication = evaluate_prediction(model_out, true_dx, distractors)
        fooled = adjudication["fooled"]

        # Record results
        results.append({
            "case_index": idx,
            "chosen_strategy": strat,
            "original_case": orig,
            "rewritten_case": rewritten,
            "realism_overlap": overlap,
            "realism_semantic": semsim,
            "model_response": model_out,
            "correct_diagnosis": true_dx,
            "target_distractor": target,
            "fooled": fooled,  # keeps original field
            "match_type": adjudication["match_type"],  # new, more informative label
            "predicted_dx_raw": adjudication["predicted_dx_raw"],
            "predicted_dx_normalized": adjudication["predicted_dx_normalized"],
            "contains_true_anywhere": adjudication["contains_true_anywhere"],
            "contains_distractor_anywhere": adjudication["contains_distractor_anywhere"],
            "confidence": adjudication["confidence"],
            "adjudication_notes": adjudication["notes"],
            "model_used": SELECTED_MODEL
        })

        print(
            f"\nCase {idx+1}: True Dx='{true_dx}', Target Dx='{target}', Model='{SELECTED_MODEL}', "
            f"Fooled={fooled}, MatchType={adjudication['match_type']}, "
            f"Pred='{adjudication['predicted_dx_raw']}', Conf={adjudication['confidence']:.2f}"
        )

    # Summarize results
    total_cases = len(results)
    if total_cases:
        fooled_count = sum(1 for r in results if r["fooled"])
        fooled_rate = fooled_count / total_cases * 100.0

        # Extra breakdown by match_type for quick QA
        breakdown = {}
        for r in results:
            breakdown[r["match_type"]] = breakdown.get(r["match_type"], 0) + 1

        print(f"\n=== Summary: {fooled_count}/{total_cases} cases fooled ({fooled_rate:.1f}%) using model '{SELECTED_MODEL}' ===")
        print("Breakdown by match_type:")
        for k, v in sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {k}: {v}")

    else:
        print("\n=== No model responses received. Check endpoints or network. ===")

    # Save results to a JSON file (kept same filename family for compatibility)
    out_file = "evaluation_results_Biomistral_7b_final.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
