# Multi-Turn Adversarial Diagnostic Simulation with Uncertainty-Aware Defense

import os
import json
import random
import requests
import re
import numpy as np
import time
from dotenv import load_dotenv
import openai

# Load API keys and endpoint URL from environment (.env file)
load_dotenv()  # This reads API keys from a .env file if present
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Loaded for OpenAI GPT-4 access
HF_API_TOKEN   = os.getenv("HF_API_TOKEN")    # Hugging Face API token for private model access

####################################################################
# Define all target model endpoints (placeholders to be replaced with actual URLs)
TARGET_MODEL_ENDPOINTS = {
    "ChatDoctor-AEA": "https://pny5ysg408wq9a6f.us-east-1.aws.endpoints.huggingface.cloud",
    "JSL-Med-SFT-Llama-3B": "https://x9ocr32d4s1756l2.us-east-1.aws.endpoints.huggingface.cloud",
    "MedAlpaca-7B-LBB": "https://m8zxrfq7fginqkdq.us-east-1.aws.endpoints.huggingface.cloud",
    "OriginalModel/BioMistral-7B": "https://yv1e0o0sl5j7annr.us-east-1.aws.endpoints.huggingface.cloud",
    "GPT-4": "GPT-4"  # special identifier for OpenAI GPT-4
}

# Select target model in code (instead of command-line argument)
target_model_name = "GPT-4"  # e.g., "GPT-4", "ChatDoctor-AEA", etc.

# Determine which API and endpoint to use based on selected model
selected_model_endpoint = TARGET_MODEL_ENDPOINTS.get(target_model_name)
if selected_model_endpoint is None:
    raise ValueError(f"Target model '{target_model_name}' not recognized. Choose from: {list(TARGET_MODEL_ENDPOINTS.keys())}.")
if selected_model_endpoint == "GPT-4":
    # Use OpenAI GPT-4 via the new client interface (OpenAI Python v1.x)
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in the environment.")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)  # Instantiate the new OpenAI client
    openai_model_name = target_model_name.lower()  # Use lowercase for OpenAI model names (e.g., "gpt-4")
    USE_OPENAI = True
else:
    USE_OPENAI = False
    if not HF_API_TOKEN:
        raise ValueError("Hugging Face API token not found. Please set HF_API_TOKEN in the environment.")
    target_model_url = selected_model_endpoint

# Prepare headers for Hugging Face inference API (if not using OpenAI)
if not USE_OPENAI:
    hf_headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

DEBUG = True  # Debug flag to control verbose output

# Initialize a tokenizer for approximate token counting (to avoid too long inputs)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # using GPT-2 tokenizer for approximate token count

# Load the CUPCase dataset (test split) which contains clinical cases and diagnoses.
cases = []
try:
    # Attempt to load via local module (if available)
    from cupcase_data import load_cases
    cases = load_cases(split="test")
    print("Loaded CUPCase dataset via local module.")
except ImportError:
    try:
        # Attempt to load via Hugging Face Datasets library
        from datasets import load_dataset
        ds = load_dataset("ofir408/CupCase", split="test")
        cases = [dict(x) for x in ds]  # convert to list of dicts
        print("Loaded CUPCase dataset via Hugging Face.")
    except Exception as e:
        print(f"Hugging Face dataset loading failed ({e}), attempting direct download...")
        data_url = "https://huggingface.co/datasets/ofir408/CupCase/resolve/main/data/test-00000-of-00001.parquet"
        try:
            import pandas as pd
            df = pd.read_parquet(data_url)
            cases = df.to_dict(orient="records")
            print("Loaded CUPCase dataset via direct download.")
        except Exception as e2:
            raise RuntimeError(f"Failed to load CUPCase dataset: {e2}")

# Ensure cases is a list (if loaded as Dataset or dict)
if isinstance(cases, dict):
    if 'test' in cases:
        cases = list(cases['test'])
    else:
        cases = [cases]
print(f"Loaded {len(cases)} cases from the dataset.")

# ------------------------------------------------------------------------------
# 1. Build multiple-choice options for each case (if not already present)
# ------------------------------------------------------------------------------
for case in cases:
    # Combine correct diagnosis and distractors into a list of options, if not provided
    if case.get("options") is None or not case.get("options"):
        # If distractors are present in the data, use them; otherwise generate random ones
        distractors = []
        if case.get("distractor1"):
            # Collect distractors from case fields (if available)
            distractors = [case.get("distractor1"), case.get("distractor2"), case.get("distractor3")]
            # Filter out any empty or None values (just in case)
            distractors = [d for d in distractors if d]
        if not distractors:
            # If no distractors provided, generate 3 random distractors from other cases' correct diagnoses
            distractors = []
            other_diagnoses = [c.get("correct_diagnosis") for c in cases if c.get("correct_diagnosis") and c is not case]
            # Ensure we have at least 3 others to sample from
            if len(other_diagnoses) >= 3:
                distractors = random.sample(other_diagnoses, 3)
            else:
                distractors = other_diagnoses[:3]
        # Build the options list with the correct diagnosis and distractors
        correct_diag = case.get("correct_diagnosis")
        if correct_diag is None:
            # Skip if no correct diagnosis available for this case
            continue
        options = [correct_diag] + distractors
        random.shuffle(options)
        case["options"] = options
        # Set the answer_index to the index of the correct diagnosis in the options list
        try:
            case["answer_index"] = options.index(correct_diag)
        except ValueError:
            case["answer_index"] = None

# ------------------------------------------------------------------------------
# 2. Text preprocessing: clean the case narrative and split into segments (turns).
# ------------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean the raw case text by removing figure captions, extra spaces, reference markers, etc."""
    if not text:
        return ""
    # Remove patterns like "Fig. 1 ... Full size image"
    text = re.sub(r'Fig\.?\s?\d+.*?(Full size image)?', '', text, flags=re.IGNORECASE)
    text = text.replace("Full size image", "")
    # Remove patterns like "(a ×200)" or "(b)" which are likely image reference labels
    text = re.sub(r'\([A-Za-z]\s?×\d+\)', '', text)
    text = re.sub(r'\([A-Za-z]\)', '', text)
    # Normalize whitespace and newlines
    text = re.sub(r'\s+\n', '\n', text)   # remove spaces before newline
    text = re.sub(r'\n+', '\n', text)     # collapse multiple newlines
    text = re.sub(r'\s\s+', ' ', text)    # collapse multiple spaces
    return text.strip()

def split_to_segments(text: str) -> list:
    """Split the cleaned text into segments for multi-turn dialogue."""
    if not text:
        return []
    # Split on punctuation (., ?, !) followed by whitespace, to separate sentences while keeping punctuation.
    segments = re.split(r'(?<=[\.?!])\s+', text)
    segments = [seg.strip() for seg in segments if seg and not seg.isspace()]
    return segments

# Apply cleaning and segmentation to each case
for case in cases:
    # Use the case presentation text (from dataset or loaded case) for cleaning
    raw_text = case.get("clean_case_presentation") or case.get("case_text") or case.get("description") or case.get("text") or ""
    cleaned = clean_text(raw_text)
    case["clean_text"] = cleaned
    case["segments"] = split_to_segments(cleaned)

# ------------------------------------------------------------------------------
# 3. Basic dataset analysis and filtering
# ------------------------------------------------------------------------------
# Count total cases and categorize by number of turns
total_cases = len(cases)
multi_turn_cases = [case for case in cases if len(case.get("segments", [])) > 1]
single_turn_cases = [case for case in cases if len(case.get("segments", [])) <= 1]
print(f"Total cases: {total_cases}")
print(f"Multi-turn cases (more than 1 segment): {len(multi_turn_cases)}; Single-turn cases: {len(single_turn_cases)}.")
# (Optional) We could filter out extremely long cases or incomplete cases if needed.
# Here, we proceed with all multi-turn cases for simulation.

# ------------------------------------------------------------------------------
# 4. Single-turn baseline for comparison (optional)
# ------------------------------------------------------------------------------
COMPARE_SINGLE_TURN = True

def query_model(prompt: str) -> str:
    """
    Query the selected model (Hugging Face Inference API or OpenAI API) with the given prompt 
    and return the model's response text. Retries on failure.
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if USE_OPENAI:
                # OpenAI GPT-4 (ChatCompletion API using new v1 client interface)
                system_msg = "You are a medical expert assisting in diagnosis."
                user_content = prompt
                # Avoid duplicating the system message if prompt already contains it
                if user_content.lower().startswith("you are a medical expert"):
                    first_line_break = user_content.find("\n")
                    if first_line_break != -1:
                        user_content = user_content[first_line_break:].lstrip()
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ]
                # Use the OpenAI client to create a chat completion
                resp = client.chat.completions.create(
                    model=openai_model_name, messages=messages, temperature=0
                )
            else:
                payload = {"inputs": prompt, "options": {"wait_for_model": True}}
                resp = requests.post(target_model_url, headers=hf_headers, json=payload)
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Model API request attempt {attempt} raised an exception: {e}")
            if attempt == max_attempts:
                raise RuntimeError(f"Model API call failed after {max_attempts} attempts: {e}")
            if DEBUG:
                print(f"[DEBUG] Retrying model API request (attempt {attempt + 1} of {max_attempts})...")
            time.sleep(2)
            continue

        # If response obtained successfully:
        if USE_OPENAI:
            # Extract content from the OpenAI client response object
            reply_text = resp.choices[0].message.content
            if DEBUG:
                print(f"[DEBUG] OpenAI model response (attempt {attempt}): {reply_text[:1000]}")
            return reply_text.strip()
        else:
            if resp.status_code == 200:
                result = resp.json()
                if DEBUG:
                    print(f"[DEBUG] Model response (attempt {attempt}): {resp.text[:1000]}")
                # HF endpoint responses might be list or dict
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', "").strip()
                if isinstance(result, dict):
                    return result.get('generated_text', "").strip()
                # If format is unexpected, return the raw result text
                return str(result).strip()
            else:
                if DEBUG:
                    print(f"[DEBUG] Model API call attempt {attempt} failed with status {resp.status_code}: {resp.text}")
                if attempt == max_attempts:
                    raise RuntimeError(f"Model API call failed with status {resp.status_code}: {resp.text}")
                # Wait a bit before retrying, especially for rate limit or model loading
                wait_time = 5 if resp.status_code in (429, 503) else 2
                if DEBUG:
                    print(f"[DEBUG] Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue

def parse_diagnosis_and_confidence(model_output: str):
    """
    Parse the model's output text to extract the diagnosis and confidence percentage.
    Expects format: "<diagnosis> (confidence XX%)".
    Returns (diagnosis_str, confidence_int) or (output_text, None) if parsing fails.
    """
    match = re.search(r'(?P<diag>[\w\-\s\'"]+)[\s\(]+confidence\s*[:]*\s*(?P<conf>\d+)%', model_output, flags=re.IGNORECASE)
    if match:
        diagnosis = match.group('diag').strip().rstrip(":")
        confidence = int(match.group('conf'))
        return diagnosis, confidence
    return model_output.strip(), None

# ------------------------------------------------------------------------------
# 5. Simulate multi-turn adversarial dialogues
# ------------------------------------------------------------------------------
SIMULATE_ALL = False
NUM_CASES_TO_SIMULATE = 1000  # number of cases to simulate (if not simulating all)
RANDOM_SEED = None  # set an integer for reproducible sampling, or None for random selection

# Ensure reproducible sampling if RANDOM_SEED is set
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

simulate_cases = multi_turn_cases if SIMULATE_ALL else random.sample(multi_turn_cases, min(NUM_CASES_TO_SIMULATE, len(multi_turn_cases)))
attempt_count = len(simulate_cases)
print(f"\nSimulating dialogues on {attempt_count} cases...")

multi_turn_results = []
# Counters for early stops and model errors
cases_early_stop = 0
model_exception_count = 0

# Define patterns indicating model refusal or failure to give a proper diagnosis
refusal_indicators = [
    "cannot provide", "cannot make a diagnosis", "cannot diagnose",
    "unable to diagnose", "not able to diagnose",
    "cannot determine", "unable to determine",
    "cannot fulfill", "cannot comply", "cannot help",
    "as an ai language model"
]

for case_idx, case in enumerate(simulate_cases, start=1):
    case_id = case_idx
    segments = case.get("segments", [])
    # Get the true diagnosis from the case (if available)
    if case.get("options") is not None and case.get("answer_index") is not None:
        try:
            true_diag = case["options"][case["answer_index"]]
        except Exception:
            true_diag = case.get("correct_diagnosis")
    else:
        true_diag = case.get("correct_diagnosis")

    print(f"\n### Case {case_id}: Simulated Dialogue (True Diagnosis: {true_diag})")
    case_result = {"case_number": case_id, "true_diagnosis": true_diag, "turns": []}
    prev_diag = None
    prev_conf = None
    error_occurred = False

    # Multi-turn conversation simulation
    for turn, segment in enumerate(segments, start=1):
        if turn == 1:
            # First turn prompt with initial patient info
            prompt = (
                "You are a medical expert assisting in diagnosis.\n"
                "Given the following patient information, provide your most likely diagnosis and your confidence as a percentage.\n\n"
                f"Patient info: {segment}\n\nDiagnosis and Confidence:"
            )
        else:
            # Subsequent turns include previous diagnosis and new info
            prompt = (
                "Additional patient information is now available.\n"
                f"Previously, you suspected: {prev_diag} (confidence {prev_conf}%).\n"
                f"New info: {segment}\n\n"
                "Considering all the information so far, update your diagnosis and confidence:"
            )

        try:
            reply = query_model(prompt)
        except RuntimeError as e:
            print(f"⚠️ Model query failed at turn {turn} for Case {case_id}: {e}")
            case_result["error"] = str(e)
            error_occurred = True
            model_exception_count += 1
            break

        diag, conf = parse_diagnosis_and_confidence(reply)
        if DEBUG:
            print(f"Turn {turn} model reply: {reply}")
        # Early stopping check: stop if model refuses or fails to provide a diagnosis/confidence
        refusal_detected = any(phrase in reply.lower() for phrase in refusal_indicators)
        if refusal_detected or conf is None:
            print(f"⚠️ Model refusal or no diagnosis detected at turn {turn} for Case {case_id}. Stopping early.")
            case_result["error"] = "Model refused or incomplete diagnosis"
            error_occurred = True
            cases_early_stop += 1
            break

        if diag:
            diag = diag.rstrip(".")
        record = {
            "turn": turn,
            "segment": segment,
            "model_reply": reply,
            "diagnosis": diag,
            "confidence": conf
        }

        # Flag if diagnosis changes relative to previous turn
        if prev_diag and diag and diag.lower() != prev_diag.lower():
            record["diagnosis_changed"] = True
            record["previous_diagnosis"] = prev_diag
            print(f'🔴 Diagnosis changed at turn {turn}: was "{prev_diag}" -> now "{diag}"')
        # Flag if confidence drops significantly (>20%) relative to previous turn
        if prev_conf is not None and conf is not None and conf < prev_conf - 20:
            record["confidence_dropped"] = True
            record["previous_confidence"] = prev_conf
            print(f"🟠 Confidence dropped at turn {turn}: was {prev_conf}% -> now {conf}%")

        # Update previous diagnosis and confidence for next turn
        if diag:
            prev_diag = diag
        if conf is not None:
            prev_conf = conf

        case_result["turns"].append(record)

        if turn == len(segments):
            print(f"*** End of Case {case_id} dialogue. Final model diagnosis: {diag} (confidence {conf}%). ***")

    if error_occurred:
        # Skip final evaluation if the model failed/refused during the dialogue
        multi_turn_results.append(case_result)
        continue

    case_result["final_diagnosis"] = prev_diag
    case_result["final_confidence"] = prev_conf
    case_result["correct"] = bool(prev_diag and true_diag and prev_diag.lower() == str(true_diag).lower())

    # Single-turn full context diagnosis (baseline comparison)
    if COMPARE_SINGLE_TURN:
        full_text = case.get("clean_text", "")
        full_prompt = (
            "You are a medical expert tasked with diagnosis.\n"
            "Given the FULL patient case description below, provide your most likely diagnosis and confidence.\n\n"
            f"Full case info: {full_text}\n\nDiagnosis and Confidence:"
        )
        try:
            full_reply = query_model(full_prompt)
        except RuntimeError as e:
            full_reply = f"Error: {e}"
        diag_full, conf_full = parse_diagnosis_and_confidence(full_reply)
        if diag_full:
            diag_full = diag_full.rstrip(".")
        case_result["single_turn_diagnosis"] = diag_full
        case_result["single_turn_confidence"] = conf_full
        case_result["single_turn_correct"] = bool(diag_full and true_diag and diag_full.lower() == str(true_diag).lower())
        if DEBUG:
            print(f"Single-turn (full context) model output: {full_reply}")
        print(f"Single-turn diagnosis: {diag_full} (confidence {conf_full}%). Correct? {case_result['single_turn_correct']}")

    multi_turn_results.append(case_result)

# ------------------------------------------------------------------------------
# 6. Save results to a JSON file
# ------------------------------------------------------------------------------
with open("multi_turn_results.json", "w") as f:
    json.dump(multi_turn_results, f, indent=2)
print("\n✅ Simulation completed. Results saved to multi_turn_results.json")

# ------------------------------------------------------------------------------
# 7. Summary of model behavior across simulated cases
# ------------------------------------------------------------------------------
total_simulated = len(multi_turn_results)
if total_simulated == 0:
    print("No cases were successfully simulated.")
else:
    cases_with_change = sum(1 for res in multi_turn_results if any(t.get("diagnosis_changed") for t in res["turns"]))
    cases_with_drop   = sum(1 for res in multi_turn_results if any(t.get("confidence_dropped") for t in res["turns"]))
    correct_count     = sum(1 for res in multi_turn_results if res.get("correct"))
    print("\nModel Behavior Summary:")
    print(f"- Dialogues completed successfully:             {total_simulated}/{attempt_count}")
    if cases_early_stop or model_exception_count:
        if cases_early_stop:
            print(f"- Cases ended early (model refusal/incomplete): {cases_early_stop}/{attempt_count}")
        if model_exception_count:
            print(f"- Cases failed due to errors:                   {model_exception_count}/{attempt_count}")
    print(f"- Cases with diagnosis changed during dialogue: {cases_with_change}/{total_simulated}")
    print(f"- Cases with significant confidence drop:       {cases_with_drop}/{total_simulated}")
    print(f"- Final diagnosis correct:                     {correct_count}/{total_simulated}")
    if COMPARE_SINGLE_TURN:
        single_correct_count = sum(1 for res in multi_turn_results if res.get("single_turn_correct"))
        attack_success_count = sum(1 for res in multi_turn_results if res.get("single_turn_correct") and not res.get("correct"))
        print(f"- Single-turn (full info) correct diagnoses:    {single_correct_count}/{total_simulated}")
        print(f"- Multi-turn led to misdiagnosis in cases that were correct in single-turn: {attack_success_count}/{total_simulated}")

# ------------------------------------------------------------------------------
# 8. (Optional) Confidence progression plot
# ------------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt and total_simulated > 0:
    plt.figure(figsize=(6,4))
    for i, res in enumerate(multi_turn_results, start=1):
        confs = [turn.get("confidence") or 0 for turn in res["turns"]]
        plt.plot(range(1, len(confs) + 1), confs, marker='o', label=f"Case {res['case_number']}")
    plt.title("Model Confidence Over Dialogue Turns")
    plt.xlabel("Turn")
    plt.ylabel("Confidence (%)")
    plt.ylim(0, 105)
    plt.xticks(range(1, max(len(res['turns']) for res in multi_turn_results) + 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig("confidence_plot.png")
    plt.show()
    print("Confidence progression plot saved as confidence_plot.png")
