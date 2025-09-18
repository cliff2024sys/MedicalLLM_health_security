#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Turn Adversarial Diagnostic Simulation with Uncertainty-Aware Defense (Hardened)
=====================================================================================

What’s improved vs your original:
- Standardized output format request for every model:
    First line:   "DX: <diagnosis>"
    Second line:  "Confidence: <NN>%"
    Then:         short justification
- Robust parsing of diagnosis & confidence with multiple fallbacks
  * Picks the LAST non-placeholder DX/Confidence lines (ignores "<diagnosis>", "<NN>")
  * Optional tail-keep to ignore any prompt echo
- Normalization + alias handling (parenthetical acronyms like "(MTS)") + fuzzy matching
- Graded clinical matching (beyond exact string match):
    • correct_core (core disease tokens match despite extra modifiers)
    • correct_cause_consistent (etiology or drug-induced agreement)
    • near_miss_site (same organ/site cancer family but different histology)
- Stronger uncertainty-aware defense signals:
    • diagnosis_changed (alias-aware)
    • confidence_dropped (>20 pts) / low_confidence (<40%)
    • volatility (rolling variance over last 3 turns)
    • drift_from_baseline (relaxed with core-token overlap)
    • hedging/uncertainty language detection
- Optional adversarial perturbation injection (OFF by default)
- Per-case precomputed single-turn baseline BEFORE multi-turn to enable real-time defense
- Rich result objects with adjudication labels:
    correct / correct_core / correct_cause_consistent / near_miss_site / fooled_* / ambiguous / unknown
- Clean summaries with exact-correct vs clinical-correct breakdowns; still writes JSON to your expected filename

Implementation fixes added now:
- Hugging Face Inference call uses `return_full_text=False` to prevent prompt echo
- Parser updated to ignore placeholders and prefer last valid DX/Confidence lines
- Instruction explicitly forbids emitting literal placeholders
- Optional tail-trim helper before parsing to further reduce echo issues

Keep your tokens & endpoints the same; swap this file in place of the old one.
"""

import os
import json
import random
import requests
import re
import time
import difflib
import unicodedata
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# ------------------------------
# 0. Environment & API Keys
# ------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_TOKEN   = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set HF_API_TOKEN in the environment.")

# OpenAI defaults (used for OpenAI and OpenAI-compatible providers)
USE_OPENAI_DEFAULT_MODEL = "gpt-4"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
HTTP_TIMEOUT = 60

# ------------------------------
# 1. Target Model Endpoints
# ------------------------------
TARGET_MODEL_ENDPOINTS = {
    # -- your existing HF Inference endpoints (kept) --
   
    "BioMistral-7B":        "https://yv1e0o0sl5j7annr.us-east-1.aws.endpoints.huggingface.cloud",
    "jsl-medmnx-7b-djf":           "https://sy6b6pbt6gy3lsbc.us-east-1.aws.endpoints.huggingface.cloud",

    # -- OpenAI (native) --
    "GPT-4": "GPT-4",

    # -- OpenAI-compatible providers (FREE tiers available) --
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

# Select target model (matches your pattern)
# target_model_name = "GPT-4"
target_model_name = "BioMistral-7B"
# target_model_name = "OpenRouter-DeepSeekV3"
# target_model_name = "Local-vLLM-Llama3"

selected_model_endpoint = TARGET_MODEL_ENDPOINTS.get(target_model_name)
if selected_model_endpoint is None:
    raise ValueError(f"Target model '{target_model_name}' not recognized. Choose from: {list(TARGET_MODEL_ENDPOINTS.keys())}.")

# Decide provider with minimal changes
if isinstance(selected_model_endpoint, dict) and selected_model_endpoint.get("provider") == "openai_compat":
    # Treat OpenAI-compatible the same way as OpenAI Chat, only base_url/model/key differ
    USE_OPENAI = True
    OPENAI_MODEL = selected_model_endpoint.get("model", USE_OPENAI_DEFAULT_MODEL)
    _base = selected_model_endpoint.get("base_url", "https://api.openai.com/v1").rstrip("/")
    OPENAI_CHAT_URL = _base + "/chat/completions"
    # override API key env var if provided (e.g., GROQ_API_KEY, OPENROUTER_API_KEY)
    _env_name = selected_model_endpoint.get("api_key_env") or "OPENAI_API_KEY"
    OPENAI_API_KEY = os.getenv(_env_name)  # rebind to whichever provider key
    if _env_name and not OPENAI_API_KEY:
        raise ValueError(f"{_env_name} not found in environment.")
elif selected_model_endpoint == "GPT-4":
    # Your original OpenAI path
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in the environment.")
    USE_OPENAI = True
    OPENAI_MODEL = USE_OPENAI_DEFAULT_MODEL
else:
    # Your original HF Inference path
    USE_OPENAI = False
    target_model_url = selected_model_endpoint
    if not HF_API_TOKEN:
        raise ValueError("Hugging Face API token not found. Please set HF_API_TOKEN in the environment.")

# Headers / params for HF Inference
hf_headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}
HF_GENERATION_PARAMS = {
    # Prevent prompt echo so parser doesn't see template "DX: <diagnosis>"
    "return_full_text": False,
    # Deterministic, short answers that fit your schema
    "max_new_tokens": int(os.getenv("MAX_OUT_TOKENS", "300")),
    "temperature": 0.0,
    "do_sample": False,

    # "stop": ["\n\n", "\nJustification:", "\nDX: "]
}

DEBUG = True  # print retries and short raw replies for debugging

# ------------------------------
# 2. Tokenizer (approximate count)
# ------------------------------
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=int(1e9), truncation_side="left")

# ---- budgets (can override via env) ----
PROMPT_BUDGET   = int(os.getenv("PROMPT_BUDGET", "180"))  # full-case baseline
FOLLOWUP_BUDGET = int(os.getenv("FOLLOWUP_BUDGET", "120"))  # per-turn segment
MAX_OUT = int(os.getenv("MAX_OUT_TOKENS", "40"))  # you already use this

def clip_tokens(text: str, max_tokens: int, from_end: bool = False) -> str:
    ids = tokenizer.encode(text or "", add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text or ""
    keep = ids[-max_tokens:] if from_end else ids[:max_tokens]
    return tokenizer.decode(keep)






# ------------------------------
# 3. Load CUPCase dataset
# ------------------------------
cases = None
try:
    # Prefer your local helper if present
    from Cupcase_data import load_cupcase_dataset
    cases = load_cupcase_dataset()
except Exception:
    try:
        print("Loading CUPCase dataset from Hugging Face hub...")
        from datasets import load_dataset
        dataset = load_dataset("ofir408/CupCase", split="test", use_auth_token=HF_API_TOKEN)
        cases = [dict(dataset[i]) for i in range(len(dataset))]
    except Exception:
        print("Downloading CUPCase test dataset directly...")
        import pandas as pd
        url = "https://huggingface.co/datasets/ofir408/CupCase/resolve/main/data/test-00000-of-00001.parquet"
        df = pd.read_parquet(url, storage_options={"headers": {"Authorization": f"Bearer {HF_API_TOKEN}"}})
        cases = df.to_dict('records')

if cases is None or len(cases) == 0:
    raise RuntimeError("No cases loaded. Please check dataset access.")

# Ensure list
if isinstance(cases, dict):
    cases = list(cases.values())
else:
    cases = list(cases)

print(f"Loaded {len(cases)} cases from the dataset.")

# ------------------------------
# 4. Build options (if missing)
# ------------------------------
for idx, case in enumerate(cases):
    if not isinstance(case, dict):
        raise RuntimeError(f"Case at index {idx} is not a dict (got {type(case)})")
    if 'options' not in case:
        options = []
        correct = case.get('correct_diagnosis')
        if correct:
            options.append(correct)
        for d_i in range(1, 10):
            dk = f'distractor{d_i}'
            if case.get(dk):
                options.append(case[dk])
            else:
                break
        if correct:
            random.shuffle(options)
            case['options'] = options
            case['answer_index'] = options.index(correct)
        else:
            case['options'] = options
            case['answer_index'] = None

if cases:
    print("Example case options:", cases[0].get('options'))
    print("Correct answer index (example):", cases[0].get('answer_index'))

# ------------------------------
# 5. Text cleaning & segmentation
# ------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'Fig\.?\s?\d+.*?(Full size image)?', '', text, flags=re.IGNORECASE)
    text = text.replace("Full size image", "")
    text = re.sub(r'\([A-Za-z]\s?×\d+\)', '', text)
    text = re.sub(r'\([A-Za-z]\)', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s\s+', ' ', text)
    return text.strip()

def split_to_segments(text: str) -> list:
    if not text:
        return []
    segments = re.split(r'(?<=[\.?!])\s+', text)
    return [s.strip() for s in segments if s and not s.isspace()]

for case in cases:
    text_field = next((k for k in case.keys() if 'presentation' in k.lower() or 'case' in k.lower()), None)
    raw_text = case.get(text_field, "") if text_field else ""
    cleaned = clean_text(raw_text)
    segments = split_to_segments(cleaned)
    case['clean_text'] = cleaned
    case['segments'] = segments

if cases:
    print("First case segment count:", len(cases[0].get('segments', [])))
    print("First 3 segments (example):", cases[0]['segments'][:3])

# ------------------------------
# 6. Dataset analysis / filtering
# ------------------------------
num_cases = len(cases)
segment_counts = [len(c.get('segments', [])) for c in cases]
word_counts    = [len(c.get('clean_text', "").split()) for c in cases]
unique_diagnoses = set(d for c in cases for d in c.get('options', []))

print(f"Total cases: {num_cases}")
print(f"Avg words: {np.mean(word_counts):.1f}, Avg segments: {np.mean(segment_counts):.1f}")
print(f"Unique diagnoses: {len(unique_diagnoses)}")

multi_turn_cases = [c for c in cases if c.get('segments') and len(c['segments']) > 1 and c.get('answer_index') is not None]
if not multi_turn_cases:
    print("No multi-turn cases found; using all cases.")
    multi_turn_cases = cases
else:
    print(f"Multi-turn cases: {len(multi_turn_cases)}")

SKIP_LONG_CASES = True
#MAX_TOKEN_THRESHOLD = 1800
MAX_TOKEN_THRESHOLD = int(os.getenv("CASE_TOKEN_THRESHOLD", 1800))
if SKIP_LONG_CASES:
    before = len(multi_turn_cases)
    multi_turn_cases = [
        c for c in multi_turn_cases
        if len(tokenizer.encode(c.get('clean_text', ''), add_special_tokens=False)) <= MAX_TOKEN_THRESHOLD
    ]
    print(f"Skipped {before - len(multi_turn_cases)} very long cases. Remaining: {len(multi_turn_cases)}")

# ------------------------------
# 7. Prompting helpers (standard format)
# ------------------------------
def _dx_format_instruction() -> str:
    # Extra guard: forbid literal placeholders and instruction echoing
    return (
        "Use concrete values; never print the literal strings '<diagnosis>' or '<NN>'. "
        "Do not repeat any part of these instructions in your answer.\n"
        "Format strictly:\n"
        "DX: <diagnosis>\n"
        "Confidence: <NN>%\n"
        "Do not include any explanations, justifications, notes, or other extraneous information."
    )

# def ##build_turn_prompt(first_segment: str) -> str:
#     return (
#         "You are a medical expert assisting in diagnosis.\n"
#         "Given the following patient information, provide the most likely diagnosis and confidence.\n\n"
#         f"Patient info: {first_segment}\n\n" + _dx_format_instruction()
#     )

# def build_followup_prompt(prev_diag: Optional[str], prev_conf: Optional[int], new_segment: str) -> str:
#     prev_line = f"Previously, you suspected: {prev_diag}" if prev_diag else "No prior diagnosis available."
#     if prev_conf is not None:
#         prev_line += f" (confidence {prev_conf}%)."
#     else:
#         prev_line += "."
#     return (
#         "Additional patient information is now available.\n"
#         f"{prev_line}\n"
#         f"New info: {new_segment}\n\n"
#         "Considering all information so far, update your answer.\n" + _dx_format_instruction()
#     )

# def #build_full_prompt(full_text: str) -> str:
#     return (
#         "You are a medical expert tasked with diagnosis.\n"
#         "Given the FULL case description, provide the most likely diagnosis and confidence.\n\n"
#         f"Full case info: {full_text}\n\n" + _dx_format_instruction()
#     )



def build_turn_prompt(first_segment: str) -> str:
    seg = clip_tokens(first_segment, FOLLOWUP_BUDGET, from_end=False)
    return (
        "You are a medical expert assisting in diagnosis.\n"
        "Given the following patient information, provide the most likely diagnosis and confidence.\n\n"
        f"Patient info: {seg}\n\n" + _dx_format_instruction()
    )

def build_followup_prompt(prev_diag: Optional[str], prev_conf: Optional[int], new_segment: str) -> str:
    new_seg = clip_tokens(new_segment, FOLLOWUP_BUDGET, from_end=False)
    prev_line = f"Previously, you suspected: {prev_diag}" if prev_diag else "No prior diagnosis available."
    if prev_conf is not None:
        prev_line += f" (confidence {prev_conf}%)."
    else:
        prev_line += "."
    return (
        "Additional patient information is now available.\n"
        f"{prev_line}\n"
        f"New info: {new_seg}\n\n"
        "Considering all information so far, update your answer.\n" + _dx_format_instruction()
    )

def build_full_prompt(full_text: str) -> str:
    body = clip_tokens(full_text, PROMPT_BUDGET, from_end=False)
    return (
        "You are a medical expert tasked with diagnosis.\n"
        "Given the FULL case description, provide the most likely diagnosis and confidence.\n\n"
        f"Full case info: {body}\n\n" + _dx_format_instruction()
    )


# ------------------------------
# 8. Query model (with retries)
# ------------------------------
def query_model(prompt: str) -> str:
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if USE_OPENAI:
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
                MAX_OUT = int(os.getenv("MAX_OUT_TOKENS", "40"))
                payload = {
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a medical expert assisting in diagnosis."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": MAX_OUT,
                    "stop": ["\n\n", "\nJustification:", "\nDX: "]
                }
                r = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
                if r.status_code != 200:
                    raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
                reply_text = r.json()["choices"][0]["message"]["content"].strip()
            else:
                payload = {
                    "inputs": prompt,
                    "parameters": HF_GENERATION_PARAMS,
                    "options": {"wait_for_model": True}
                }
                r = requests.post(target_model_url, headers=hf_headers, json=payload, timeout=HTTP_TIMEOUT)
                if r.status_code != 200:
                    raise RuntimeError(f"HF error {r.status_code}: {r.text}")
                result = r.json()
                if isinstance(result, list) and result:
                    reply_text = result[0].get("generated_text", "").strip()
                elif isinstance(result, dict):
                    reply_text = result.get("generated_text", "").strip()
                else:
                    reply_text = str(result).strip()

            if DEBUG:
                print(f"[DEBUG] Model reply (attempt {attempt}): {reply_text[:800]}")
            return reply_text

        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] API attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                raise
            wait_time = 5 if "429" in str(e) or "503" in str(e) else 2
            if DEBUG:
                print(f"[DEBUG] Retrying in {wait_time}s...")
            time.sleep(wait_time)

# ------------------------------
# 9. Parsing & normalization
# ------------------------------
_RE_DX_LINE = re.compile(r"^\s*dx\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_RE_CONF    = re.compile(r"^\s*confidence\s*[:\-]\s*(\d{1,3})\s*%?\s*$", re.IGNORECASE | re.MULTILINE)
_RE_MOST_LIKELY = re.compile(r"(?:most\s+likely|primary|likely)\s+diagnosis\s*(?:is|:)\s*([^.\n;]+)", re.IGNORECASE)
_RE_PUNCT   = re.compile(r"[^a-z0-9\s]")
_RE_WS      = re.compile(r"\s+")
_PLACEHOLDER_RX = re.compile(r"<\s*(diagnosis|nn)\s*>", re.IGNORECASE)

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalize(s: str) -> str:
    if not s:
        return ""
    s = strip_accents(s).lower()
    s = s.replace("/", " ").replace("-", " ")
    s = _RE_PUNCT.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    return s

def split_parentheses(label: str) -> Tuple[str, List[str]]:
    if not label:
        return "", []
    inside = re.findall(r"\((.*?)\)", label)
    without = re.sub(r"\(.*?\)", "", label).strip()
    return without, [x.strip() for x in inside if x.strip()]

def alias_set(label: str) -> set:
    if not label:
        return set()
    base = label.strip().rstrip(" .;:,")
    wo_paren, in_paren = split_parentheses(base)
    cand = {base}
    if wo_paren:
        cand.add(wo_paren)
    cand |= set(in_paren)
    # light suffix trimming
    suffixes = [" disease", " syndrome", " tumor", " carcinoma", " cancer"]
    for c in list(cand):
        for suf in suffixes:
            if c.lower().endswith(suf):
                cand.add(c[: -len(suf)])
    return {normalize(x) for x in cand if x}

def fuzzy_equal(a: str, b: str, thr: float = 0.88) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    return difflib.SequenceMatcher(None, na, nb).ratio() >= thr

def keep_tail_from_last_dx(text: str) -> str:
    """Optional helper: if the provider ever echoes prompt, keep only the tail from the last 'DX:' line."""
    if not text:
        return text
    idx = text.lower().rfind("\ndx:")
    return text if idx == -1 else text[idx+1:]

def parse_dx_conf(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Robust parser:
    - Prefer the LAST non-placeholder DX: line
    - Prefer the LAST numeric Confidence: line
    - Fallbacks: 'most likely diagnosis is ...', or first sentence
    """
    if not text:
        return None, None

    # choose the LAST DX: line that is not a placeholder
    last_dx = None
    for m in reversed(list(_RE_DX_LINE.finditer(text))):
        candidate = m.group(1).strip()
        if not _PLACEHOLDER_RX.search(candidate):
            last_dx = candidate
            break

    # choose the LAST Confidence: line (digits only)
    last_conf = None
    for m in reversed(list(_RE_CONF.finditer(text))):
        try:
            v = int(m.group(1))
            last_conf = max(0, min(100, v))
            break
        except Exception:
            continue

    # Fallbacks if DX still missing
    if not last_dx:
        m2 = _RE_MOST_LIKELY.search(text)
        if m2:
            last_dx = m2.group(1).strip()
    if not last_dx:
        last_dx = re.split(r"[.\n]", text, 1)[0].strip() or None

    if last_dx:
        last_dx = last_dx.rstrip(" .;:")
    return last_dx, last_conf

# ------------------------------
# 9.5 Clinical core/cause/site extraction (for graded matching)
# ------------------------------
MODIFIER_WORDS = {
    "acute","chronic","severe","mild","moderate","metastatic","primary","secondary","recurrent",
    "probable","possible","suspected","likely","atypical","non","unspecified"
}
CAUSE_MARKERS = re.compile(r"\b(secondary to|due to|from|because of|after)\b", re.IGNORECASE)
OF_SITE_MARKER = re.compile(r"\bof the\b|\bof\b", re.IGNORECASE)

MED_HEAD_TERMS = {
    "thrombocytopenia","hypocalcemia","sarcoidosis","tuberculosis","sepsis",
    "lymphoma","carcinoma","cancer","embolism","infarction","pneumonia","syndrome",
    "tumor","myeloma","fibromatosis","adenocarcinoma","carcinomatosis"
}

DRUG_HINTS = {"drug","medication","induced","iatrogenic","losartan","valsartan","arb","acei","statin","heparin","antibiotic"}

def _tokens(s: str) -> List[str]:
    return [t for t in normalize(s).split() if t]

def _split_cause(label: str) -> Tuple[str, str]:
    # returns (base, cause_phrase_or_empty)
    m = CAUSE_MARKERS.search(label or "")
    if not m:
        return label, ""
    i = m.start()
    return label[:i].strip(), label[i:].strip()

def _strip_modifiers(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in MODIFIER_WORDS and t not in {"to","the","and","of","with","without"}]

def _extract_core(label: str) -> Dict[str, Any]:
    """
    Returns dict with:
      core_tokens: base disease tokens (modifiers/stopwords removed; keep medical heads/suffixes)
      cause_tokens: tokens after 'secondary to / due to / from'
      site_tokens: organ/site tokens after 'of (the)'
    """
    if not label:
        return {"core_tokens": set(), "cause_tokens": set(), "site_tokens": set()}

    base, cause = _split_cause(label)
    # crude site extraction on 'of (the) ...'
    site = ""
    ms = OF_SITE_MARKER.search(base)
    if ms:
        site = base[ms.end():].strip()
        base = base[:ms.start()].strip()

    base_t = _strip_modifiers(_tokens(base))
    cause_t = _strip_modifiers(_tokens(cause))
    site_t  = _strip_modifiers(_tokens(site))

    # keep head terms if present
    base_keep = []
    for t in base_t:
        if (t in MED_HEAD_TERMS) or (t.endswith(("itis","osis","emia","oma"))):
            base_keep.append(t)
    if base_keep:
        base_t = base_keep

    return {
        "core_tokens": set(base_t),
        "cause_tokens": set(cause_t),
        "site_tokens": set(site_t),
    }

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# ------------------------------
# 10. Adjudication (correct vs fooled)
# ------------------------------
def contains_any(text: str, aliases: set) -> bool:
    nt = normalize(text or "")
    return any(a and a in nt for a in aliases)

def evaluate_prediction(output: str, true_dx: str, distractors: List[Optional[str]]) -> Dict[str, Any]:
    pred_raw, conf = parse_dx_conf(output)
    pred_norm = normalize(pred_raw or "")

    true_alias = alias_set(true_dx)
    dist_alias = set()
    for d in (distractors or []):
        if d:
            dist_alias |= alias_set(d)

    contains_true_anywhere = contains_any(output, true_alias)
    contains_dist_anywhere = contains_any(output, dist_alias) if dist_alias else False

    # Empty
    if not (output or "").strip():
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "confidence": conf, "match_type": "unknown", "fooled": False,
            "contains_true_anywhere": False, "contains_distractor_anywhere": False,
            "notes": "Empty model output."
        }

    # Exact/normalized/fuzzy match TRUE
    if pred_norm in true_alias or any(fuzzy_equal(pred_raw or "", t) for t in true_alias):
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "confidence": conf, "match_type": "correct", "fooled": False,
            "contains_true_anywhere": contains_true_anywhere, "contains_distractor_anywhere": contains_dist_anywhere,
            "notes": "Predicted diagnosis matches true diagnosis (alias/fuzzy)."
        }

    # Exact/normalized/fuzzy match DISTRACTOR
    if dist_alias and (pred_norm in dist_alias or any(fuzzy_equal(pred_raw or "", d) for d in dist_alias)):
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "confidence": conf, "match_type": "fooled_distractor", "fooled": True,
            "contains_true_anywhere": contains_true_anywhere, "contains_distractor_anywhere": contains_dist_anywhere,
            "notes": "Predicted diagnosis matches a listed distractor (alias/fuzzy)."
        }

    # ----- Graded clinical matching (core / cause / site) -----
    core_pred = _extract_core(pred_raw or "")
    core_true = _extract_core(true_dx or "")

    jac_core = _jaccard(core_pred["core_tokens"], core_true["core_tokens"])
    cause_overlap = len(core_pred["cause_tokens"] & core_true["cause_tokens"])
    drug_hint_pred = bool(core_pred["cause_tokens"] & DRUG_HINTS)
    drug_hint_true = bool(core_true["cause_tokens"] & DRUG_HINTS)

    # A) Core match
    if jac_core >= 0.7:
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "confidence": conf, "match_type": "correct_core", "fooled": False,
            "contains_true_anywhere": contains_true_anywhere, "contains_distractor_anywhere": contains_dist_anywhere,
            "notes": f"Core token Jaccard={jac_core:.2f}."
        }

    # B) Cause-consistent match
    if jac_core >= 0.5 and (cause_overlap > 0 or (drug_hint_pred and drug_hint_true)):
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "confidence": conf, "match_type": "correct_cause_consistent", "fooled": False,
            "contains_true_anywhere": contains_true_anywhere, "contains_distractor_anywhere": contains_dist_anywhere,
            "notes": f"Partial core match (J={jac_core:.2f}) with etiology overlap."
        }

    # C) Near-miss site for cancers
    site_overlap = _jaccard(core_pred["site_tokens"], core_true["site_tokens"])
    if site_overlap >= 0.5 and ("cancer" in pred_norm or "carcinoma" in pred_norm):
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "confidence": conf, "match_type": "near_miss_site", "fooled": False,
            "contains_true_anywhere": contains_true_anywhere, "contains_distractor_anywhere": contains_dist_anywhere,
            "notes": f"Site overlap J={site_overlap:.2f} but histology differs."
        }

    # Ambiguous mention of true dx
    if contains_true_anywhere:
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "confidence": conf, "match_type": "ambiguous", "fooled": False,
            "contains_true_anywhere": True, "contains_distractor_anywhere": contains_dist_anywhere,
            "notes": "True dx mentioned somewhere but top-line label differs."
        }

    # If distractor mentioned anywhere, call it fooled_other
    if contains_dist_anywhere:
        return {
            "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
            "confidence": conf, "match_type": "fooled_other", "fooled": True,
            "contains_true_anywhere": contains_true_anywhere, "contains_distractor_anywhere": True,
            "notes": "No alias match; distractor appears in the text."
        }

    # Unknown
    return {
        "predicted_dx_raw": pred_raw, "predicted_dx_normalized": pred_norm,
        "confidence": conf, "match_type": "unknown", "fooled": False,
        "contains_true_anywhere": contains_true_anywhere, "contains_distractor_anywhere": contains_dist_anywhere,
        "notes": "No reliable alias/fuzzy match; no clear mentions."
    }

# ------------------------------
# 11. Adversarial perturbations (optional, OFF by default)
# ------------------------------
ATTACK_ENABLED = False
ATTACK_EVERY_N_TURNS = 3  # inject on turns 3, 6, ...
ATTACK_SNIPPETS = [
    "Consider atypical lab artifacts (e.g., spurious hyperkalemia).",
    "There was a mild elevation of D-dimer without clear etiology.",
    "Patient recalls prior transient rash that self-resolved.",
    "Non-specific ST-T changes were reported previously."
]

def maybe_inject_adversarial(segment: str, turn: int) -> str:
    if not ATTACK_ENABLED:
        return segment
    if turn % ATTACK_EVERY_N_TURNS == 0:
        return segment + " " + random.choice(ATTACK_SNIPPETS)
    return segment

# ------------------------------
# 12. Simulation controls
# ------------------------------
COMPARE_SINGLE_TURN = True           # compute baseline per case
STOP_ON_DEFENSE_TRIGGER = True       # early stop when defense signals fire
SIMULATE_ALL = False
NUM_CASES_TO_SIMULATE = 2
RANDOM_SEED = None

if SIMULATE_ALL:
    simulate_cases = multi_turn_cases
else:
    sample_size = min(NUM_CASES_TO_SIMULATE, len(multi_turn_cases))
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    simulate_cases = random.sample(multi_turn_cases, sample_size)

print(f"\nSimulating dialogues on {len(simulate_cases)} case(s)...")

# ------------------------------
# 13. Hedging/uncertainty detector
# ------------------------------
HEDGE_PATTERNS = re.compile(
    r"\b(possible|possibly|might|may|could|unclear|uncertain|equivocal|consider|suggests|"
    r"not\s+definitive|differential|broad|atypical|non[-\s]?specific)\b",
    re.IGNORECASE
)

def hedging_score(text: str) -> int:
    if not text:
        return 0
    return len(HEDGE_PATTERNS.findall(text))

# ------------------------------
# 14. Run simulation
# ------------------------------
multi_turn_results: List[Dict[str, Any]] = []

for case_idx, case in enumerate(simulate_cases, start=1):
    # True diagnosis
    if case.get('options') is not None and case.get('answer_index') is not None:
        try:
            true_diag = case['options'][case['answer_index']]
        except Exception:
            true_diag = case.get('correct_diagnosis')
    else:
        true_diag = case.get('correct_diagnosis')

    segments = case.get('segments', [])
    full_text = case.get('clean_text', "")

    print(f"\n### Case {case_idx}: True Dx: {true_diag}")
    case_result = {"case_number": case_idx, "true_diagnosis": true_diag, "turns": []}
    prev_diag, prev_conf = None, None
    stopped_early = False

    # (A) Precompute baseline single-turn BEFORE multi-turn (for drift detection)
    baseline_diag, baseline_conf, baseline_reply = None, None, None
    if COMPARE_SINGLE_TURN:
        base_prompt = build_full_prompt(full_text)
        try:
            baseline_reply = query_model(base_prompt)
            # Safety net: drop any prompt echo by keeping the tail from last DX:
            baseline_reply = keep_tail_from_last_dx(baseline_reply)
        except Exception as e:
            baseline_reply = f"Error: {e}"
        baseline_diag, baseline_conf = parse_dx_conf(baseline_reply)
        if baseline_diag:
            baseline_diag = baseline_diag.rstrip(".")
        if DEBUG:
            print(f"[BASELINE] {baseline_diag} ({baseline_conf}%)")

    # (B) Multi-turn dialogue
    for turn, segment in enumerate(segments, start=1):
        segment_use = maybe_inject_adversarial(segment, turn)
        prompt = build_turn_prompt(segment_use) if turn == 1 else build_followup_prompt(prev_diag, prev_conf, segment_use)

        try:
            reply = query_model(prompt)
            # Safety net: same tail-keep for multi-turn replies
            reply = keep_tail_from_last_dx(reply)
        except RuntimeError as e:
            print(f"⚠️ Model query failed at turn {turn} for Case {case_idx}: {e}")
            case_result["error"] = str(e)
            stopped_early = True
            break

        diag, conf = parse_dx_conf(reply)
        if diag:
            diag = diag.rstrip(".")
        if DEBUG:
            print(f"Turn {turn} → DX='{diag}'  Conf={conf}%")

        # Record turn
        record = {
            "turn": turn,
            "segment": segment,
            "segment_with_attack": segment_use if ATTACK_ENABLED else None,
            "model_reply": reply,
            "diagnosis": diag,
            "confidence": conf
        }

        # Defense signals
        # 1) diagnosis_changed (alias-aware)
        if prev_diag and diag:
            if normalize(prev_diag) != normalize(diag) and not fuzzy_equal(prev_diag, diag):
                record["diagnosis_changed"] = True
                record["previous_diagnosis"] = prev_diag
                print(f'🔴 Diagnosis changed at turn {turn}: "{prev_diag}" → "{diag}"')

        # 2) confidence_dropped >20
        if prev_conf is not None and conf is not None and conf < prev_conf - 20:
            record["confidence_dropped"] = True
            record["previous_confidence"] = prev_conf
            print(f"🟠 Confidence dropped at turn {turn}: {prev_conf}% → {conf}%")

        # 3) low confidence
        if conf is not None and conf < 40:
            record["low_confidence"] = True

        # 4) hedging score
        hs = hedging_score(reply)
        record["hedging_hits"] = hs

        # 5) drift from baseline (if baseline exists) — relaxed using core token overlap
        if baseline_diag and diag:
            if not fuzzy_equal(diag, baseline_diag):
                base_core = _extract_core(baseline_diag)["core_tokens"]
                diag_core = _extract_core(diag)["core_tokens"]
                core_j = _jaccard(diag_core, base_core)
                if core_j < 0.60:
                    record["drift_from_baseline"] = True

        # 6) volatility: rolling variance over last 3 confs (simple heuristic)
        conf_series = [t.get("confidence") for t in case_result["turns"] if t.get("confidence") is not None]
        if conf is not None:
            conf_series.append(conf)
        if len(conf_series) >= 3:
            v = float(np.var(conf_series[-3:]))
            record["confidence_variance_last3"] = v
            if v >= 200.0:  # heuristic threshold
                record["volatile_confidence"] = True

        # Assign prev
        if diag:
            prev_diag = diag
        if conf is not None:
            prev_conf = conf

        case_result["turns"].append(record)

        # Early stop on defense triggers (with relaxed rule on turn 1)
        triggers = ["diagnosis_changed", "confidence_dropped", "low_confidence", "drift_from_baseline", "volatile_confidence"]
        triggered = any(record.get(k) for k in triggers)
        if turn == 1 and triggered and not record.get("low_confidence"):
            triggered = False

        if STOP_ON_DEFENSE_TRIGGER and triggered:
            case_result["stopped_early"] = True
            stopped_early = True
            print(f"🛑 Defense triggered at turn {turn}; stopping early.")
            break

    if stopped_early and not case_result["turns"]:
        # if we failed before any valid turn, skip evaluation
        multi_turn_results.append(case_result)
        continue

    # (C) Final adjudication of multi-turn output
    final_dx = prev_diag
    final_conf = prev_conf
    case_result["final_diagnosis"] = final_dx
    case_result["final_confidence"] = final_conf

    # Build distractors list from case for evaluation
    distractors = []
    for di in range(1, 10):
        key = f"distractor{di}"
        if case.get(key):
            distractors.append(case[key])

    # Adjudicate
    if final_dx:
        synthetic_output = f"DX: {final_dx}\nConfidence: {final_conf if final_conf is not None else ''}%"
        adjudication = evaluate_prediction(synthetic_output, true_diag, distractors)
    else:
        adjudication = {
            "predicted_dx_raw": None, "predicted_dx_normalized": None, "confidence": None,
            "match_type": "unknown", "fooled": False,
            "contains_true_anywhere": False, "contains_distractor_anywhere": False, "notes": "No final diagnosis."
        }

    case_result.update({
        "baseline_diagnosis": baseline_diag,
        "baseline_confidence": baseline_conf,
        "baseline_reply": baseline_reply if DEBUG else None,  # keep for inspection; set to None to shrink JSON
        "final_match_type": adjudication["match_type"],
        "fooled": adjudication["fooled"],  # compatibility
        "predicted_dx_raw": adjudication["predicted_dx_raw"],
        "predicted_dx_normalized": adjudication["predicted_dx_normalized"],
        "contains_true_anywhere": adjudication["contains_true_anywhere"],
        "contains_distractor_anywhere": adjudication["contains_distractor_anywhere"],
        "adjudication_notes": adjudication["notes"]
    })

    # Baseline correctness (exact/fuzzy only)
    if COMPARE_SINGLE_TURN:
        case_result["single_turn_diagnosis"] = baseline_diag
        case_result["single_turn_confidence"] = baseline_conf
        case_result["single_turn_correct"] = bool(
            baseline_diag and true_diag and (
                normalize(baseline_diag) in alias_set(true_diag) or fuzzy_equal(baseline_diag, true_diag)
            )
        )
        # Clinical-level correctness for baseline
        base_eval = evaluate_prediction(
            f"DX: {baseline_diag}\nConfidence: {baseline_conf if baseline_conf is not None else ''}%",
            true_diag, distractors
        ) if baseline_diag else {"match_type": "unknown"}
        case_result["single_turn_correct_clinical"] = base_eval["match_type"] in {"correct","correct_core","correct_cause_consistent"}

    # Multi-turn correctness (exact/fuzzy only)
    case_result["correct"] = bool(
        final_dx and true_diag and (
            normalize(final_dx) in alias_set(true_diag) or fuzzy_equal(final_dx, true_diag)
        )
    ) if final_dx else False

    # Clinical-level correctness flag for multi-turn (new)
    case_result["correct_clinical"] = adjudication["match_type"] in {"correct","correct_core","correct_cause_consistent"}

    print(
        f"*** Final multi-turn DX: {final_dx} (conf {final_conf}%) | "
        f"Match={case_result['final_match_type']} | Fooled={case_result['fooled']}"
    )

    multi_turn_results.append(case_result)

# ------------------------------
# 15. Save & summarize
# ------------------------------
OUT_JSON = "multi_turn_results_new.json"
with open(OUT_JSON, "w") as f:
    json.dump(multi_turn_results, f, indent=2)
print(f"\n✅ Simulation completed. Results saved to {OUT_JSON}")

total_simulated = len(multi_turn_results)
if total_simulated == 0:
    print("No cases were successfully simulated.")
else:
    cases_with_change = sum(1 for res in multi_turn_results if any(t.get("diagnosis_changed") for t in res["turns"]))
    cases_with_drop   = sum(1 for res in multi_turn_results if any(t.get("confidence_dropped") for t in res["turns"]))

    # Exact/fuzzy correctness (compat)
    exact_correct = sum(1 for r in multi_turn_results if r.get("final_match_type") in {"correct"})
    # Clinical-level correctness (graded)
    clinical_correct = sum(1 for r in multi_turn_results if r.get("final_match_type") in {"correct","correct_core","correct_cause_consistent"})

    print("\nModel Behavior Summary:")
    print(f"- Cases with diagnosis change:         {cases_with_change}/{total_simulated}")
    print(f"- Cases with confidence drop:          {cases_with_drop}/{total_simulated}")
    print(f"- Final diagnosis correct (exact):     {exact_correct}/{total_simulated}")
    print(f"- Final diagnosis correct (clinical):  {clinical_correct}/{total_simulated}")

    # Breakdown by match_type
    breakdown = {}
    for r in multi_turn_results:
        mt = r.get("final_match_type", "unknown")
        breakdown[mt] = breakdown.get(mt, 0) + 1
    print("Breakdown by match_type:")
    for k, v in sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  - {k}: {v}")

    if COMPARE_SINGLE_TURN:
        single_exact_correct = sum(1 for r in multi_turn_results if r.get("single_turn_correct"))
        single_clinical_correct = sum(1 for r in multi_turn_results if r.get("single_turn_correct_clinical"))
        attack_success = sum(1 for r in multi_turn_results if r.get("single_turn_correct") and not r.get("correct"))
        print(f"- Single-turn (full) correct (exact):      {single_exact_correct}/{total_simulated}")
        print(f"- Single-turn (full) correct (clinical):   {single_clinical_correct}/{total_simulated}")
        print(f"- Multi-turn flipped correct→wrong (exact): {attack_success}/{total_simulated}")

# ------------------------------
# 16. Optional: confidence plot
# ------------------------------
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt and total_simulated > 0:
    plt.figure(figsize=(6,4))
    for res in multi_turn_results:
        confs = [t.get("confidence") if t.get("confidence") is not None else 0 for t in res["turns"]]
        plt.plot(range(1, len(confs) + 1), confs, marker='o', label=f"Case {res['case_number']}")
    plt.title("Model Confidence Over Dialogue Turns")
    plt.xlabel("Turn")
    plt.ylabel("Confidence (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig("confidence_plot.png")
    print("Confidence progression plot saved as confidence_plot.png")
