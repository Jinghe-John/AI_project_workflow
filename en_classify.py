# -*- coding: utf-8 -*-
"""
Paper Classification Pipeline  (parallel + checkpoint + measurement)
- Reads paper metadata from a CSV file and classifies each paper via a
  multi-step hierarchical LLM workflow.
- Supports multiple API keys with round-robin worker assignment.
- Supports resume-from-checkpoint on interruption.
- Structural papers are further sub-classified as Measurement / Non-Measurement.

Result columns written to the output CSV:
    en_has_data, en_has_model, en_paper_type,
    en_research_article, en_variable, en_measurement

Usage:
    Set API_KEYS and INPUT_FILE, then run:
        python classify_papers_parallel.py
"""

import json
import os
import pickle
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
API_KEYS: list[str] = [
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # Add more keys to increase concurrency
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]

BASE_URL   = "https://api.moonshot.cn/v1"
MODEL      = "kimi-k2-0711-preview"

INPUT_FILE = "all_Structural.csv"

MAX_WORKERS             = len(API_KEYS)
CHECKPOINT_FILE         = "classification_checkpoint_v4.pkl"
CHECKPOINT_SAVE_EVERY   = 5    # Save checkpoint every N completed tasks
INTERMEDIATE_SAVE_EVERY = 20   # Write intermediate CSV every N completed tasks

# Mapping from internal result keys → output column names
RESULT_COLUMNS: dict[str, str] = {
    "has_data":         "en_has_data",
    "has_model":        "en_has_model",
    "paper_type":       "en_paper_type",
    "research_article": "en_research_article",
    "variable":         "en_variable",
    "measurement":      "en_measurement",
}

# LLM call settings
TEMPERATURE  = 0
SEED         = 12345
MAX_TOKENS   = 200
MAX_RETRIES  = 3
RETRY_DELAY  = 2   # Base seconds between retries (multiplied by attempt number)
STEP_DELAY   = 0.5 # Seconds to sleep between sequential classification steps

# ============================================================
# API Client Pool
# ============================================================
_clients: list[OpenAI] = [
    OpenAI(api_key=key, base_url=BASE_URL) for key in API_KEYS
]


def get_client(worker_id: int) -> OpenAI:
    """Return the pre-built client assigned to this worker."""
    return _clients[worker_id % len(_clients)]


# ============================================================
# Thread Safety
# ============================================================
results_lock = threading.Lock()


# ============================================================
# Checkpoint I/O
# ============================================================

def save_checkpoint(data: dict, path: str) -> bool:
    """Persist a checkpoint dict to disk via pickle. Returns True on success."""
    try:
        with open(path, "wb") as fh:
            pickle.dump(data, fh)
        return True
    except Exception as exc:
        tqdm.write(f"[Checkpoint] Save failed: {exc}")
        return False


def load_checkpoint(path: str) -> dict | None:
    """
    Load a checkpoint from disk if it exists.
    Returns the checkpoint dict, or None if the file is missing or unreadable.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        print(f"[Checkpoint] Resuming from: {path}")
        return data
    except Exception as exc:
        print(f"[Checkpoint] Load failed ({exc}) — starting fresh.")
        return None


def clear_checkpoint(path: str) -> None:
    """Remove the checkpoint file after a successful run."""
    if os.path.exists(path):
        os.remove(path)
        print("[Checkpoint] Checkpoint file removed.")


# ============================================================
# CSV I/O
# ============================================================

def read_csv(filepath: str) -> pd.DataFrame:
    """
    Attempt to read a CSV file using several common encodings.
    Raises RuntimeError if all encodings fail.
    """
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"):
        try:
            df = pd.read_csv(filepath, low_memory=False, encoding=encoding)
            print(f"  Read {len(df)} rows with encoding '{encoding}'.")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise RuntimeError(f"Failed to read '{filepath}' with encoding '{encoding}'.") from exc
    raise RuntimeError(f"Could not read '{filepath}' with any known encoding.")


def write_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a UTF-8-BOM CSV file."""
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing result columns (initialised to empty string)."""
    for col in RESULT_COLUMNS.values():
        if col not in df.columns:
            df[col] = ""
    return df


# ============================================================
# Prompt Builders  (English prompts preserved as-is)
# ============================================================

def build_data_model_prompt(introductory_part: str) -> str:
    prompt = {
        "persona": "Senior editorial assistant at the American Economic Review (AER)",
        "task": (
            "From the provided paper's abstract and introduction, answer two separate questions "
            "about the paper's methodology. First, determine if it uses empirical data. "
            "Second, determine if it develops a formal model."
        ),
        "output_format": {
            "description": "Provide a 'Yes' or 'No' classification for each of the two questions based on the text.",
            "classification_definitions": {
                "has_data": {
                    "Yes": "The paper uses real world data, contains empirical analysis, simulation or computable general equilibrium (CGE) model.",
                    "No":  "The paper does not use real-world data.",
                },
                "has_model": {
                    "Yes": "The paper develops or significantly extends a formal mathematical, theoretical, or computational model/statistical tests.",
                    "No":  "The paper does not introduce a new formal model or statistical tests; it may use existing models or focus on non-model-based analysis.",
                },
            },
            "schema": {"has_data": "[Yes|No]", "has_model": "[Yes|No]"},
        },
        "paper_title & paper_abstract & introduction": introductory_part,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_structural_prompt(introductory_part: str) -> str:
    prompt = {
        "persona": "Senior editorial assistant at the American Economic Review (AER)",
        "task": (
            "From the provided text of a paper's information (title, abstract and introduction), "
            "precisely determine whether the paper is a structural empirical paper. "
            "Use the detailed definitions below for an accurate classification."
        ),
        "output_format": {
            "description": "Classify whether the paper is a structural empirical paper based on the description of its approach in the text.",
            "classification_definitions": {
                "Structural":     "A structural empirical paper MUST contain counterfactual analysis or calibration.",
                "Non-Structural": "A paper that does NOT contain counterfactual analysis and calibration.",
            },
            "schema": {"structural_classification": "[Yes|No]"},
        },
        "paper_title & paper_abstract & introduction": introductory_part,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_measurement_prompt(introductory_part: str) -> str:
    prompt = {
        "persona": "Senior editorial assistant at the American Economic Review (AER)",
        "task": "Mechanically classify the paper as 'Measurement Article' or 'Non-Measurement Article' based on the research endpoint.",
        "output_format": {
            "classification_definitions": {
                "Measurement Article": (
                    "The research endpoint is solely to produce an isolated numerical indicator "
                    "(e.g., calculating a trade restriction index or tariff equivalent for a given year). "
                    "Criterion: The paper does NOT involve any 'effect of X on Y' or 'causal inference'. "
                    "If the paper merely computes the numeric value of a 'measurement tool' itself, classify it here."
                ),
                "Non-Measurement Article": (
                    "The paper involves 'effect/contribution of X on Y', 'mechanism analysis', "
                    "'policy effect evaluation', or 'growth accounting'. "
                    "Criterion: Any paper containing 'causal inference', 'factor contribution decomposition "
                    "(such as growth accounting)', 'impulse response', or 'institutional comparison' must NOT "
                    "be classified as a measurement article. The research endpoint is a logical conclusion "
                    "rather than an isolated numeric value."
                ),
            },
            "schema": {
                "classification":  "[Measurement Article|Non-Measurement Article]",
                "justification":   "Must explain: whether the output is an 'isolated numeric value' or involves 'effect/contribution/mechanism of X on Y'.",
            },
        },
        "introductory_part": introductory_part,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_non_structural_prompt(introductory_part: str) -> str:
    prompt = {
        "persona": "Senior editorial assistant at the American Economic Review (AER)",
        "task": (
            "From the provided text of a paper that is a non-structural empirical paper or contains data "
            "without a model, determine the specific type of the paper. "
            "Use the detailed definitions below for an accurate classification."
        ),
        "output_format": {
            "description": "Classify the empirical approach into one of two categories based on the primary contribution described in the paper.",
            "classification_definitions": {
                "Pure Empirical": "Covers empirical analysis of data that does not involve the estimation of a fully specified structural economic model.",
                "Pure Theory":    "Focuses on the development of abstract models and methods for economic analysis.",
                "Other":          "Any paper that does not fit into the above categories.",
            },
            "schema": {"classification": "[Pure Empirical|Pure Theory|Other]"},
        },
        "paper_title & paper_abstract & introduction": introductory_part,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_explanatory_variables_prompt(title: str) -> str:
    prompt = {
        "persona": "Senior editorial assistant at the American Economic Review (AER)",
        "task": "Based on the structure of the paper's title, determine the type of the paper.",
        "output_format": {
            "description": "Classify the paper into one of the following two categories.",
            "classification_definitions": {
                "Single Explanatory":   "The paper's title explicitly lists all the core, specific concepts to be analyzed.",
                "Multiple Explanators": "The paper's title uses broad, open-ended, exploratory terms like 'Determinants', 'Factors', 'Causes', or 'Influences'.",
            },
            "schema": {"classification": "[Single Explanatory|Multiple Explanators]"},
        },
        "paper_title": title,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_research_classification_prompt(introductory_part: str) -> str:
    prompt = {
        "persona": "Senior editorial assistant at the American Economic Review (AER)",
        "task": "Based on whether the paper contains a regression equation, determine if it is a Research Article or a Non-Research Article.",
        "output_format": {
            "description": "Classify the paper into one of the following two categories.",
            "classification_definitions": {
                "Non-Research Article": "The paper does not contain any empirical estimation/empirical regressions/empirical test/empirical comparison.",
                "Research Article":     "The paper explicitly includes some empirical estimations/empirical regressions/empirical tests/empirical comparisons.",
            },
            "schema": {"classification": "[Research Article|Non-Research Article]"},
        },
        "paper_title & paper_abstract & introduction": introductory_part,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


# ============================================================
# Generic API Caller
# ============================================================

def call_api(client: OpenAI, system: str, user_prompt: str) -> str:
    """
    Call the LLM API with exponential-backoff retry.
    Returns the raw text response, or raises RuntimeError after all retries.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                seed=SEED,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise RuntimeError(f"API call failed after {MAX_RETRIES} attempts.") from exc
    return ""  # unreachable, satisfies type checkers


def parse_json(text: str) -> dict:
    """Parse a JSON string; raises json.JSONDecodeError on failure."""
    return json.loads(text)


# ============================================================
# Individual Classification Functions
# ============================================================
_SYSTEM = "You are a senior editorial assistant. Respond with valid JSON only."


def classify_data_model(client: OpenAI, introductory_part: str) -> tuple[str, str]:
    """Returns (has_data, has_model) — each 'Yes', 'No', or 'ERROR'."""
    try:
        text   = call_api(client, _SYSTEM, build_data_model_prompt(introductory_part))
        result = parse_json(text)
        return result.get("has_data", "No"), result.get("has_model", "No")
    except Exception:
        return "ERROR", "ERROR"


def classify_structural(client: OpenAI, introductory_part: str) -> str:
    """Returns 'Yes', 'No', or 'ERROR'."""
    try:
        text   = call_api(client, _SYSTEM, build_structural_prompt(introductory_part))
        result = parse_json(text)
        return result.get("structural_classification", "No")
    except Exception:
        return "ERROR"


def classify_measurement(client: OpenAI, introductory_part: str) -> str:
    """Returns 'Measurement Article', 'Non-Measurement Article', or 'ERROR'."""
    try:
        text   = call_api(client, _SYSTEM, build_measurement_prompt(introductory_part))
        result = parse_json(text)
        return result.get("classification", "Non-Measurement Article")
    except Exception:
        return "ERROR"


def classify_non_structural_type(client: OpenAI, introductory_part: str) -> str:
    """Returns 'Pure Empirical', 'Pure Theory', 'Other', or 'ERROR'."""
    try:
        text   = call_api(client, _SYSTEM, build_non_structural_prompt(introductory_part))
        result = parse_json(text)
        return result.get("classification", "Other")
    except Exception:
        return "ERROR"


def classify_explanatory_variables(client: OpenAI, title: str) -> str:
    """Returns 'Single Explanatory', 'Multiple Explanators', or 'ERROR'."""
    try:
        text   = call_api(client, _SYSTEM, build_explanatory_variables_prompt(title))
        result = parse_json(text)
        return result.get("classification", "Other")
    except Exception:
        return "ERROR"


def classify_article_type(client: OpenAI, introductory_part: str) -> str:
    """Returns 'Research Article', 'Non-Research Article', or 'ERROR'."""
    try:
        text   = call_api(client, _SYSTEM, build_research_classification_prompt(introductory_part))
        result = parse_json(text)
        return result.get("classification", "Other")
    except Exception:
        return "ERROR"


# ============================================================
# Hierarchical Classification Orchestrator
# ============================================================

def classify_paper_hierarchical(
    client: OpenAI, title: str, introductory_part: str
) -> dict[str, str]:
    """
    Run the full hierarchical classification decision tree for a single paper.

    Returns a dict with keys matching RESULT_COLUMNS (internal names):
        has_data, has_model, paper_type, research_article, variable, measurement
    """
    result: dict[str, str] = {
        "has_data":         "",
        "has_model":        "",
        "paper_type":       "",
        "research_article": "",
        "variable":         "",
        "measurement":      "",
    }

    # Step 1: Data & model presence
    has_data, has_model = classify_data_model(client, introductory_part)
    result["has_data"]  = has_data
    result["has_model"] = has_model
    time.sleep(STEP_DELAY)

    # Step 2: Branch on data / model
    if has_data == "ERROR" or has_model == "ERROR":
        result["paper_type"] = "ERROR"
        return result

    if has_data == "Yes" and has_model == "Yes":
        is_structural = classify_structural(client, introductory_part)
        time.sleep(STEP_DELAY)

        if is_structural == "Yes":
            result["paper_type"] = "Structural"
            result["measurement"] = classify_measurement(client, introductory_part)
            time.sleep(STEP_DELAY)
        else:
            paper_type = classify_non_structural_type(client, introductory_part)
            result["paper_type"] = paper_type
            if paper_type in ("Pure Empirical", "Other"):
                research_article = classify_article_type(client, introductory_part)
                result["research_article"] = research_article
                if research_article == "Research Article":
                    result["variable"] = classify_explanatory_variables(client, title)

    elif has_data == "Yes" and has_model == "No":
        result["paper_type"] = "Pure Empirical"
        research_article = classify_article_type(client, introductory_part)
        result["research_article"] = research_article
        if research_article == "Research Article":
            result["variable"] = classify_explanatory_variables(client, title)

    elif has_data == "No" and has_model == "Yes":
        result["paper_type"] = "Pure Theory"

    else:  # No data, no model
        result["paper_type"] = "Other"
        research_article = classify_article_type(client, introductory_part)
        result["research_article"] = research_article
        if research_article == "Research Article":
            result["variable"] = classify_explanatory_variables(client, title)

    return result


# ============================================================
# JSON Reader
# ============================================================

def read_json_intro(json_path: str) -> str:
    """
    Load a JSON file and return the content of the first 'introductory-part' item.
    Returns an empty string on any error or if no such item exists.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data:
            if isinstance(item, dict) and item.get("type") == "introductory-part":
                return item.get("content", "")
    except Exception:
        pass
    return ""


# ============================================================
# Per-task Worker (called from thread pool)
# ============================================================

_DEFAULT_RESULT: dict[str, str] = {
    "has_data": "", "has_model": "", "paper_type": "",
    "research_article": "", "variable": "", "measurement": "",
}


def process_single_paper(
    task: tuple[int, str, str, int]
) -> tuple[int, dict[str, str], str]:
    """
    Classify one paper.

    Args:
        task: (original_df_index, title, json_path, worker_id)

    Returns:
        (original_df_index, result_dict, status)
        where status ∈ {'success', 'skip', 'error'}
    """
    original_idx, title, json_path, worker_id = task
    result = dict(_DEFAULT_RESULT)  # Shallow copy

    if pd.isna(title) or not str(title).strip():
        result["paper_type"] = "SKIP_NO_TITLE"
        return original_idx, result, "skip"

    if pd.isna(json_path) or not str(json_path).strip():
        result["paper_type"] = "SKIP_NO_JSON_PATH"
        return original_idx, result, "skip"

    json_path = str(json_path).strip()
    if not os.path.exists(json_path):
        result["paper_type"] = "FILE_NOT_FOUND"
        return original_idx, result, "error"

    intro = read_json_intro(json_path)
    client = get_client(worker_id)

    try:
        result = classify_paper_hierarchical(client, str(title).strip(), intro)
        return original_idx, result, "success"
    except Exception as exc:
        result["paper_type"] = f"ERROR: {str(exc)[:50]}"
        return original_idx, result, "error"


# ============================================================
# Result Writer
# ============================================================

def apply_result_to_df(df: pd.DataFrame, idx: int, result: dict[str, str]) -> None:
    """Write a single paper's classification result into the DataFrame in-place."""
    for col_key, col_name in RESULT_COLUMNS.items():
        df.at[idx, col_name] = result.get(col_key, "")


def print_distribution(df: pd.DataFrame, column: str, label: str) -> None:
    """Print a value-count distribution for one result column."""
    series = df.loc[df[column].astype(str).str.strip() != "", column]
    print(f"\n  {label}:")
    if series.empty:
        print("    (no records)")
    else:
        for value, cnt in series.value_counts().items():
            print(f"    {value}: {cnt}")


# ============================================================
# Main Processing Function
# ============================================================

def process_papers(csv_path: str) -> None:
    """
    Full pipeline: load CSV → resume checkpoint → classify in parallel → save results.
    """
    print("=" * 60)
    print("Reading input CSV...")

    script_dir        = os.path.dirname(os.path.abspath(csv_path))
    checkpoint_path   = os.path.join(script_dir, CHECKPOINT_FILE)
    intermediate_path = os.path.join(script_dir, "classification_intermediate_v4.csv")

    # ── Load data ────────────────────────────────────────────
    try:
        df = read_csv(csv_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return

    required_columns = ("en_json_path", "Ref_Title")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f"ERROR: missing required column(s): {missing}")
        print(f"  Available columns: {list(df.columns)}")
        return

    df = ensure_result_columns(df)
    original_count      = len(df)
    indices_to_classify = df.index.tolist()

    # ── Resume from checkpoint ───────────────────────────────
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        processed_indices: set[int] = checkpoint.get("processed_indices", set())
        start_time: str             = checkpoint.get("start_time", datetime.now().strftime("%Y%m%d_%H%M%S"))
        all_results: dict           = checkpoint.get("results", {})

        # Restore previously saved results back into the DataFrame
        for idx, res in all_results.items():
            if idx in df.index:
                apply_result_to_df(df, idx, res)

        print(f"  Restored {len(processed_indices)} previously processed record(s) from checkpoint.")
    else:
        processed_indices = set()
        start_time        = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results       = {}

    indices_to_process = [i for i in indices_to_classify if i not in processed_indices]
    total_to_process   = len(indices_to_process)

    print(f"\n  Total rows in file   : {original_count}")
    print(f"  Already processed    : {len(processed_indices)}")
    print(f"  Remaining this run   : {total_to_process}")
    print(f"  Parallel workers     : {MAX_WORKERS}  (one per API key)")
    print("=" * 60)

    if total_to_process == 0:
        print("All records already processed — writing final output.")
        _save_final(df, script_dir, start_time, original_count,
                    original_count, 0, 0, 0, checkpoint_path, intermediate_path)
        return

    # ── Build task list ──────────────────────────────────────
    tasks = [
        (idx, df.at[idx, "Ref_Title"], df.at[idx, "en_json_path"], i % MAX_WORKERS)
        for i, idx in enumerate(indices_to_process)
    ]

    # ── Parallel classification ──────────────────────────────
    success_count = error_count = skip_count = completed_count = 0

    pbar = tqdm(total=total_to_process, desc="Classifying papers", unit="paper", ncols=100)

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_paper, t): t for t in tasks}

            for future in as_completed(futures):
                try:
                    original_idx, result, status = future.result()
                except Exception as exc:
                    tqdm.write(f"  [Worker error] {exc}")
                    pbar.update(1)
                    continue

                with results_lock:
                    apply_result_to_df(df, original_idx, result)
                    all_results[original_idx] = result
                    processed_indices.add(original_idx)

                    if status == "success":
                        success_count += 1
                    elif status == "skip":
                        skip_count += 1
                    else:
                        error_count += 1

                    completed_count += 1

                    # Periodic checkpoint save
                    if completed_count % CHECKPOINT_SAVE_EVERY == 0:
                        save_checkpoint(
                            {"processed_indices": processed_indices,
                             "results": all_results,
                             "start_time": start_time},
                            checkpoint_path,
                        )

                    # Periodic intermediate CSV save
                    if completed_count % INTERMEDIATE_SAVE_EVERY == 0:
                        write_csv(df, intermediate_path)
                        tqdm.write(f"  [Intermediate save] {completed_count} records → {intermediate_path}")

                pbar.update(1)

    except KeyboardInterrupt:
        pbar.close()
        print("\n\nInterrupted by user — saving progress...")
        save_checkpoint(
            {"processed_indices": processed_indices, "results": all_results, "start_time": start_time},
            checkpoint_path,
        )
        interrupted_path = os.path.join(script_dir, f"classification_interrupted_{start_time}.csv")
        write_csv(df, interrupted_path)
        print(f"  Progress saved to  : {interrupted_path}")
        print(f"  Records completed  : {len(processed_indices)}")
        print("  Re-run the script to resume from this checkpoint.")
        return

    pbar.close()

    # ── Save final output ────────────────────────────────────
    _save_final(df, script_dir, start_time, original_count,
                len(indices_to_classify), success_count, skip_count, error_count,
                checkpoint_path, intermediate_path)


def _save_final(
    df: pd.DataFrame,
    output_dir: str,
    timestamp: str,
    original_count: int,
    filtered_count: int,
    success_count: int,
    skip_count: int,
    error_count: int,
    checkpoint_path: str,
    intermediate_path: str,
) -> None:
    """Save the final output CSV and print summary statistics."""
    output_path = os.path.join(output_dir, f"All_Structural_classified_{timestamp}.csv")

    try:
        write_csv(df, output_path)
        print(f"\n{'=' * 60}")
        print(f"Classification complete.")
        print(f"  Output file: {output_path}")

        # Verify row count
        df_check       = read_csv(output_path)
        pt_col         = RESULT_COLUMNS["paper_type"]
        non_empty      = df_check[df_check[pt_col].notna() & (df_check[pt_col] != "")].shape[0]
        print(f"  Rows in output : {len(df_check)}")
        print(f"  Rows classified: {non_empty}")

        # Clean up temporary files
        clear_checkpoint(checkpoint_path)
        if os.path.exists(intermediate_path):
            os.remove(intermediate_path)

        # ── Summary statistics ───────────────────────────────
        print(f"\n{'=' * 60}")
        print("Classification Summary")
        print(f"{'=' * 60}")
        print(f"  Total rows           : {original_count}")
        print(f"  Rows to classify     : {filtered_count}")
        print(f"  Successfully parsed  : {success_count}")
        print(f"  Skipped              : {skip_count}")
        print(f"  Errors               : {error_count}")

        for col_key, label in [
            ("paper_type",       "en_paper_type distribution"),
            ("has_data",         "en_has_data distribution"),
            ("has_model",        "en_has_model distribution"),
            ("research_article", "en_research_article distribution"),
            ("variable",         "en_variable distribution"),
            ("measurement",      "en_measurement distribution"),
        ]:
            print_distribution(df, RESULT_COLUMNS[col_key], label)

        print(f"{'=' * 60}")

    except Exception as exc:
        print(f"ERROR: failed to save output — {exc}")
        backup_path = os.path.join(output_dir, f"classification_backup_{timestamp}.csv")
        try:
            write_csv(df, backup_path)
            print(f"  Backup saved to: {backup_path}")
        except Exception:
            print("  Backup save also failed.")


# ============================================================
# Entry Point
# ============================================================

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, INPUT_FILE)

    print("=" * 60)
    print("Paper Classification Pipeline v4.1")
    print("=" * 60)
    print(f"  Input file      : {INPUT_FILE}")
    print(f"  API keys loaded : {len(API_KEYS)}")
    print(f"  Workers         : {MAX_WORKERS}")
    print(f"  Result columns  : {list(RESULT_COLUMNS.values())}")
    print()

    if not os.path.exists(csv_path):
        print(f"ERROR: input file not found — {csv_path}")
        print(f"  Make sure '{INPUT_FILE}' is in the same directory as this script.")
        return

    try:
        process_papers(csv_path)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        traceback.print_exc()

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
