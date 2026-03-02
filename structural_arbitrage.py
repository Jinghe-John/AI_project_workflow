#!/usr/bin/env python3
"""
Academic Arbitrage Detection Pipeline (Structural / Non-Measurement Papers)
Features:
  1. Multi-API-key round-robin with parallel worker threads
  2. Resume-from-checkpoint (persisted between runs)
  3. Periodic intermediate result saves
  4. Automatic CSV encoding detection
  5. All original CSV rows preserved in output
  6. Full pairwise English-to-English arbitrage recording (all combinations)
  7. ERROR values are recorded directly — no retry logic

Differences from reduced_form_arbitrage.py:
  - Filter: paper_type == 'Structural' AND measurement == 'Non-Measurement Article'
  - Third dimension: 'Counterfactual Analysis' (not 'Mechanism')
  - En-En check: all-pairs combinatorial (not tournament elimination)
  - Prompt: detailed structural-model-specific criteria with examples

Usage:
    Set INPUT_PATH and OUTPUT_DIR at the bottom of the file, then run:
        python structural_arbitrage.py
"""

import glob
import json
import os
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ============================================================
# Configuration
# ============================================================
BASE_URL = "https://api.moonshot.cn/v1"
MODEL    = "kimi-k2-0711-preview"

API_KEYS: list[str] = [
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    # Add additional keys here; each key maps to one worker thread
]

INPUT_PATH = "path/to/your/input.csv"
OUTPUT_DIR = os.path.dirname(INPUT_PATH)

TEMPERATURE = 0
SEED        = 12345
MAX_TOKENS  = 4096

# ============================================================
# API Client Pool
# ============================================================
api_clients: list[OpenAI] = [OpenAI(api_key=k, base_url=BASE_URL) for k in API_KEYS]
NUM_WORKERS: int           = len(api_clients)

_api_index_lock    = threading.Lock()
_current_api_index = 0


def get_next_client() -> tuple[OpenAI, str]:
    """Return the next API client using round-robin rotation."""
    global _current_api_index
    with _api_index_lock:
        client = api_clients[_current_api_index]
        _current_api_index = (_current_api_index + 1) % len(api_clients)
    return client, MODEL


# ============================================================
# Output Column Definitions
# ============================================================
RESULT_COLUMNS = [
    "research_question", "research_design", "counterfactual_analysis",
    "arbitrage", "most_likely_source", "skip_reason",
    "round1_rq", "round1_rd", "round1_mech",
    "round2_rq", "round2_rd", "round2_mech",
    "consistency_rq", "consistency_rd", "consistency_mech",
    "en_en_arbitrage_detected", "en_en_arbitrage_pairs",
    "api_error", "api_error_detail",
]

# ============================================================
# I/O Helpers
# ============================================================

def read_csv(filepath: str) -> pd.DataFrame:
    """
    Read a CSV file, trying common encodings in order.
    Falls back to UTF-8 with error-ignoring as a last resort.
    """
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin-1"):
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"  Loaded CSV with encoding '{encoding}': {filepath}")
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            continue

    df = pd.read_csv(filepath, encoding="utf-8", errors="ignore")
    print(f"  Loaded CSV with UTF-8 (errors ignored): {filepath}")
    return df


def read_json_content(filepath: str) -> str:
    """
    Read a JSON file and return its content as a formatted string.
    Returns an empty string if the path is missing, empty, or unreadable.
    """
    if not filepath or pd.isna(filepath) or not str(filepath).strip():
        return ""
    filepath = str(filepath).strip()
    if not os.path.exists(filepath):
        return ""

    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin-1"):
        try:
            with open(filepath, "r", encoding=encoding) as fh:
                data = json.load(fh)
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            continue
    return ""


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Write the DataFrame to a UTF-8-BOM CSV."""
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all result columns to df if they do not already exist."""
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


# ============================================================
# Checkpoint Manager
# ============================================================

class CheckpointManager:
    """Persists the set of processed row indices to a JSON sidecar file."""

    def __init__(self, output_csv_path: str) -> None:
        self.output_path     = output_csv_path
        self.checkpoint_path = output_csv_path.replace(".csv", "_checkpoint.json")
        self.processed: set[int] = set()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.processed = set(data.get("processed_indices", []))
                print(f"  Checkpoint loaded — {len(self.processed)} row(s) already processed.")
            except Exception:
                self.processed = set()

    def save(self) -> None:
        try:
            with open(self.checkpoint_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"processed_indices": list(self.processed),
                     "last_update": datetime.now().isoformat()},
                    fh, ensure_ascii=False,
                )
        except Exception as exc:
            print(f"  WARNING: Failed to save checkpoint: {exc}")

    def mark(self, index: int) -> None:
        self.processed.add(index)

    def is_done(self, index: int) -> bool:
        return index in self.processed

    def cleanup(self) -> None:
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print("  Checkpoint file removed.")


# ============================================================
# Pre-processing Filter
# ============================================================

def should_analyze_pair(
    chn_paper_type: str,
    eng_paper_type: str,
    chn_measurement: str,
    eng_measurement: str,
) -> tuple[bool, str]:
    """
    Return (True, '') only when both papers are Structural Non-Measurement Articles.
    Otherwise return (False, reason_string).
    """
    chn_type = str(chn_paper_type).strip() if pd.notna(chn_paper_type) else ""
    eng_type = str(eng_paper_type).strip() if pd.notna(eng_paper_type) else ""
    chn_meas = str(chn_measurement).strip() if pd.notna(chn_measurement) else ""
    eng_meas = str(eng_measurement).strip() if pd.notna(eng_measurement) else ""

    if (
        chn_type == "Structural"
        and eng_type == "Structural"
        and chn_meas == "Non-Measurement Article"
        and eng_meas == "Non-Measurement Article"
    ):
        return True, ""

    if not chn_type or chn_type == "nan":
        return False, "paper_type is empty"
    if not eng_type or eng_type == "nan":
        return False, "en_paper_type is empty"
    if chn_type != "Structural":
        return False, f"paper_type is '{chn_type}'"
    if eng_type != "Structural":
        return False, f"en_paper_type is '{eng_type}'"
    return False, "other"


# ============================================================
# LLM API Calls
# ============================================================

def _call_api(
    system_prompt: str,
    user_prompt: dict,
    default_error: dict,
    client: Optional[OpenAI] = None,
) -> dict:
    """
    Call the LLM API with a structured JSON prompt.
    Returns the parsed JSON response, or default_error (augmented with error info)
    on any failure. No retries — errors are recorded directly.
    """
    try:
        use_client = client if client is not None else get_next_client()[0]

        response = use_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": json.dumps(user_prompt, ensure_ascii=False, indent=2)},
            ],
            temperature=TEMPERATURE,
            seed=SEED,
            response_format={"type": "json_object"},
            max_tokens=MAX_TOKENS,
        )

        if not response.choices:
            result = default_error.copy()
            result["api_error"]        = "empty_choices"
            result["api_error_detail"] = "API returned an empty choices list."
            return result

        return json.loads(response.choices[0].message.content)

    except Exception as exc:
        result = default_error.copy()
        result["api_error"]        = "api_call_failed"
        result["api_error_detail"] = str(exc)
        return result


def run_arbitrage_analysis(
    chn_passage: str,
    eng_passage: str,
    client: OpenAI,
) -> dict:
    """
    Ask the LLM to evaluate structural arbitrage across three dimensions:
      - Research Question (mechanism mapping test)
      - Research Design   (model skeleton / identification recipe)
      - Counterfactual Analysis (functional equivalence of policy simulations)
    """
    system_prompt = (
        "You are a top-tier journal referee. Your task is to conduct a multi-dimensional review "
        "to detect potential academic arbitrage. You must strictly evaluate the paper's contribution "
        "against the provided definitions for three key dimensions: Research Question, Research Design, "
        "and Counterfactual Analysis, and return a boolean judgment for each."
    )

    user_prompt = {
        "task": (
            "Based on the provided materials for two papers, conduct a holistic evaluation to determine "
            "if the Chinese paper constitutes academic arbitrage of the English paper. For each of the "
            "three dimensions, provide a boolean judgment based strictly on the provided definitions."
        ),
        "evaluation_framework": {
            "name": "The Holistic Arbitrage Review (HAR)",
            "description": (
                "You must sequentially analyze the three dimensions below and provide a boolean judgment for each."
            ),
            "dimensions": [
                [
                    {
                        "dimension": 1,
                        "name": "Research Question (RQ)",
                        "definition": "Determine if the Chinese paper is strictly 'arbitraging' the English paper by investigating the theoretically equivalent economic relationship under a different institutional label. \n\nTo make this judgment, perform the 'Mechanism Mapping Test':\n1. Identify the 'Unique Chinese Factors' claimed in the Chinese paper (e.g., 'Hukou', 'Latecomer Advantage', 'Government Subsidy').\n2. Map these factors to standard economic parameters found in the English paper (e.g., 'Hukou' → 'Labor Mobility Friction/Wedge'; 'Latecomer Advantage' → 'Convergence/Catch-up effect'; 'Subsidy' → 'Cost Reduction/Distortion').\n3. Compare the Core Question: Once the Chinese terms are translated into standard terms, does the Chinese paper ask the exact same causality question (Does X cause Y?) using the same identification logic as the English paper?\n\nJudgement Criteria: It IS Arbitrage if the Chinese paper merely treats the institutional context as a specific proxy for a standard variable (e.g., using Hukou to measure Friction) to validate if the coefficient sign/magnitude holds in China. It is NOT Arbitrage only if the Chinese context introduces a new structural mechanism that changes the functional form of the model, not just the parameter values.",
                        "examples": [
                            {
                                "case": "Arbitrage",
                                "description": "English paper: Quantifies the misallocation of capital due to financial frictions in the US manufacturing sector. Chinese paper: Quantifies the misallocation of capital due to financial frictions in the Chinese manufacturing sector using the same definition of misallocation."
                            },
                            {
                                "case": "Arbitrage (Framework Displacement)",
                                "description": "English paper analyzes optimal taxation in a federal system; Chinese paper analyzes optimal taxation in a provincial system using the same trade-offs and logic."
                            },
                            {
                                "case": "Not Arbitrage (Different X)",
                                "description": "English paper: Analyzes financial frictions. Chinese paper: Analyzes state-ownership (a specific institutional feature) that requires a fundamental change to the firm's objective function in the model."
                            },
                            {
                                "case": "Not Arbitrage (Different Y)",
                                "description": "English paper: Explains labor share decline via automation. Chinese paper: Explains labor share decline via structural transformation from agriculture to manufacturing."
                            }
                        ]
                    },
                    {
                        "dimension": 2,
                        "name": "Research Design (RD)",
                        "definition": "A Chinese paper arbitrages an English paper if it adopts the same 'Model Backbone' or 'Identification Recipe'. For structural work, A Chinese paper arbitrages an English paper if: it adopts an identical structural model and estimation strategy. Compare the model primitives (preferences, technology, constraints) and the identification strategy (choice of moments for GMM/SMM or likelihood function). If the Chinese paper uses the exact same system of equations and matches the exact same set of data moments (e.g., firm size distribution, investment volatility) to estimate parameters, it is arbitrage. ",
                        "examples": [
                            {
                                "case": "Arbitrage",
                                "description": "English paper uses a Hopenhayn (1992) style model with adjustment costs, estimated via SMM by matching investment rates and Tobin's Q. Chinese paper uses the exact same setup and matches the exact same moments using Chinese listed firm data."
                            },
                            {
                                "case": "Arbitrage (Identification Mimicry)",
                                "description": "The Chinese paper replicates the 'moment matching' or 'structural inversion' strategy of the English paper, using the same mathematical mappings from data distributions (e.g., city population/wages) to parameters."
                            },
                            {
                                "case": "Not Arbitrage",
                                "description": "The Chinese paper extends a standard model by adding a specific 'Hukou' constraint equation that alters the labor supply elasticity, necessitating a new block of equations to solve the equilibrium."
                            },
                            {
                                "case": "Not Arbitrage",
                                "description": "English paper identifies the elasticity using time-series variation in tariffs. Chinese paper identifies it using cross-sectional variation in unique Chinese geography, requiring a fundamentally different estimator."
                            }
                        ]
                    },
                    {
                        "dimension": 3,
                        "name": "Counterfactual Analysis (CF)",
                        "definition": "A Chinese paper arbitrages an English paper if its counterfactual experiments are 'Functional Equivalents' of those in the English paper. It is arbitrage if the simulated 'shock' (e.g., removing a distortion, changing a tax, or shifting a constraint) targets the same economic margin and derives the same qualitative welfare or efficiency implications. If the simulation's 'transmission channel' (how the shock moves through the model to the outcome) is identical to the English paper, it is arbitrage.",
                        "examples": [
                            {
                                "case": "Arbitrage",
                                "description": "English paper simulates a counterfactual where size-dependent policies are removed, finding TFP rises by 10%. Chinese paper simulates the removal of size-dependent policies in China, finding TFP rises by 15%. The logic and the experiment are identical."
                            },
                            {
                                "case": "Arbitrage (Functional Equivalence)",
                                "description": "English paper: Simulates removing general migration costs. Chinese paper: Simulates removing Hukou restrictions. Both target the same 'Friction-to-Reallocation' transmission channel."
                            },
                            {
                                "case": "Not Arbitrage",
                                "description": "English paper simulates the removal of size-dependent policies. Chinese paper simulates a unique Chinese policy: the privatization of State-Owned Enterprises (SOEs), representing a distinct policy shock."
                            }
                        ]
                    }
                ]
              ],
                },
            ],
        },
        "inputs": {
            "english_paper_materials": eng_passage,
            "chinese_paper_materials": chn_passage,
        },
        "output_format": {
            "description": "Return your analysis as a single JSON object with three boolean keys.",
            "schema": {
                "research_question_arbitrage":        "boolean",
                "research_design_arbitrage":          "boolean",
                "counterfactual_analysis_arbitrage":  "boolean",
            },
        },
    }

    default_error = {
        "error":                               "Analysis failed.",
        "research_question_arbitrage":         False,
        "research_design_arbitrage":           False,
        "counterfactual_analysis_arbitrage":   False,
    }
    return _call_api(system_prompt, user_prompt, default_error, client=client)


# ============================================================
# English-to-English All-Pairs Arbitrage Check
# ============================================================

def run_en_en_pairwise(
    english_papers: List[Dict],
    client: OpenAI,
    worker_id: int,
) -> tuple[List[Dict], Dict]:
    """
    Run all-pairs combinatorial arbitrage checks among English papers.
    All papers are retained regardless of results (no elimination).
    Returns (original_papers, pairwise_records).
    """
    if len(english_papers) < 2:
        return english_papers, {}

    records: Dict = {}

    for i, p1 in enumerate(english_papers):
        for p2 in english_papers[i + 1:]:
            c1 = read_json_content(p1["filepath"])
            c2 = read_json_content(p2["filepath"])
            key = f"{p1['filename']}|{p2['filename']}"

            try:
                if c1 and c2:
                    result    = run_arbitrage_analysis(c1, c2, client)
                    has_error = "api_error" in result

                    if has_error:
                        rq = rd = mech = "ERROR"
                        is_arb = "ERROR"
                    else:
                        rq    = result.get("research_question_arbitrage",       False)
                        rd    = result.get("research_design_arbitrage",         False)
                        mech  = result.get("counterfactual_analysis_arbitrage", False)
                        is_arb = all([rq, rd, mech])

                    records[key] = {
                        "is_arbitrage":     is_arb,
                        "rq": rq, "rd": rd, "mech": mech,
                        "api_error":        result.get("api_error"),
                        "api_error_detail": result.get("api_error_detail"),
                    }

                    if is_arb is True:
                        print(f"      [Worker-{worker_id}] En-En arbitrage found: {p1['filename']} <-> {p2['filename']}")
                    elif is_arb == "ERROR":
                        print(f"      [Worker-{worker_id}] En-En comparison API error: {p1['filename']} <-> {p2['filename']}")

            except Exception as exc:
                print(f"      [Worker-{worker_id}] En-En comparison failed: {exc}")
                records[key] = {
                    "is_arbitrage": "ERROR",
                    "rq": "ERROR", "rd": "ERROR", "mech": "ERROR",
                    "api_error":        "exception",
                    "api_error_detail": str(exc),
                }

    return english_papers, records


# ============================================================
# Most-Likely Source Identification
# ============================================================

def find_most_likely_source(
    chn_passage: str,
    candidates: List[Dict],
    client: OpenAI,
) -> str | dict:
    """
    Use sequential tournament comparison to identify which English paper is the
    most direct source of the Chinese paper's arbitrage.

    Returns the winning filename, or an error dict on API failure.
    """
    if not candidates:
        return {"most_likely_source_filename": "no_candidates"}
    if len(candidates) == 1:
        return candidates[0]["filename"]

    current = candidates[0]

    for challenger in candidates[1:]:
        system_prompt = (
            "You are a senior academic referee identifying the textual blueprint "
            "and narrative skeleton of a paper."
        )
        user_prompt = {
            "task": (
                "Compare these two English papers and determine which one is the most direct "
                "and primary source of the Chinese paper's arbitrage."
            ),
            "inputs": {
                "chinese_paper_passage": chn_passage,
                "english_paper_A": {"filename": current["filename"],    "passage": current["passage"]},
                "english_paper_B": {"filename": challenger["filename"], "passage": challenger["passage"]},
            },
            "evaluation_criteria": (
                "The primary source is the paper from which the Chinese paper derives its core "
                "intellectual structure most heavily. Choose the single most likely source."
            ),
            "output_format": {
                "schema": {
                    "most_likely_source_filename": "string (must be either paper A or paper B filename)"
                }
            },
        }
        default_error = {"most_likely_source_filename": "analysis_failed"}
        response = _call_api(system_prompt, user_prompt, default_error, client=client)

        if "api_error" in response:
            return response

        winner_filename = response.get("most_likely_source_filename", "extraction_failed")
        if winner_filename == "extraction_failed":
            return {"most_likely_source_filename": "extraction_failed"}
        if winner_filename == challenger["filename"]:
            current = challenger

    return current["filename"]


# ============================================================
# Single Chinese Paper Processing
# ============================================================

def process_chinese_paper_group(
    chn_filepath: str,
    tasks: List[Dict],
    client: OpenAI,
    worker_id: int,
) -> Dict:
    """
    Process all English candidate tasks for one Chinese paper.

    For each English candidate:
      1. Run two independent arbitrage analysis rounds.
      2. Compute per-dimension final verdict (both rounds must agree True).
      3. Collect arbitrage candidates for en-en pairwise check.

    Returns a dict with 'results' (one entry per task) and 'en_en_arbitrage_records'.
    """
    print(f"    [Worker-{worker_id}] Processing: {os.path.basename(chn_filepath)} ({len(tasks)} English candidate(s)).")

    per_task_results:     List[Dict] = []
    arbitrage_candidates: List[Dict] = []

    for task in tasks:
        index        = task["index"]
        eng_filepath = task["eng_filepath"]
        eng_year     = task["eng_year"]

        r: Dict = {
            "index":                index,
            "is_arbitrage":         False,
            "final_result":         {},
            "round1_result":        None,
            "round2_result":        None,
            "consistency_check":    None,
            "error":                None,
            "api_error":            None,
            "api_error_detail":     None,
            "eng_filepath":         eng_filepath,
            "eng_year":             eng_year,
            "filename":             os.path.basename(eng_filepath),
            "most_likely_source":   None,
            "en_en_arbitrage_info": {"detected": False, "pairs": ""},
        }

        try:
            chn_passage = read_json_content(chn_filepath)
            eng_passage = read_json_content(eng_filepath)

            if not chn_passage or not eng_passage:
                r["error"] = "file_content_read_failed"
                per_task_results.append(r)
                continue

            round1 = run_arbitrage_analysis(chn_passage, eng_passage, client)
            round2 = run_arbitrage_analysis(chn_passage, eng_passage, client)

            r1_err = "api_error" in round1
            r2_err = "api_error" in round2

            if r1_err or r2_err:
                errors = []
                if r1_err:
                    errors.append(f"round1: {round1.get('api_error_detail', 'unknown')}")
                if r2_err:
                    errors.append(f"round2: {round2.get('api_error_detail', 'unknown')}")
                r["api_error"]        = "round_error"
                r["api_error_detail"] = "; ".join(errors)

            final: Dict       = {}
            consistency: Dict = {}

            for dim in (
                "research_question_arbitrage",
                "research_design_arbitrage",
                "counterfactual_analysis_arbitrage",
            ):
                v1 = "ERROR" if r1_err else round1.get(dim, False)
                v2 = "ERROR" if r2_err else round2.get(dim, False)

                final[dim] = "ERROR" if (v1 == "ERROR" or v2 == "ERROR") else (v1 and v2)

                key = dim.replace("_arbitrage", "")
                if v1 == "ERROR" or v2 == "ERROR":
                    consistency[key] = "error"
                else:
                    consistency[key] = "consistent" if v1 == v2 else "inconsistent"

            is_arb = all(
                final.get(d) is True
                for d in (
                    "research_question_arbitrage",
                    "research_design_arbitrage",
                    "counterfactual_analysis_arbitrage",
                )
            )

            r["is_arbitrage"]      = is_arb
            r["final_result"]      = final
            r["round1_result"]     = round1
            r["round2_result"]     = round2
            r["consistency_check"] = consistency

            if is_arb:
                arbitrage_candidates.append({
                    "index":     index,
                    "filename":  os.path.basename(eng_filepath),
                    "filepath":  eng_filepath,
                    "passage":   eng_passage,
                    "year":      eng_year,
                    "row_index": index,
                })

        except Exception as exc:
            r["error"] = f"processing_failed: {exc}"

        per_task_results.append(r)

    # ── En-En all-pairs arbitrage check ──────────────────────
    en_en_records: Dict = {}
    en_en_info = {"detected": False, "pairs": ""}

    if len(arbitrage_candidates) >= 2:
        print(f"    [Worker-{worker_id}] {len(arbitrage_candidates)} arbitrage candidate(s) — running en-en pairwise check...")

        _, en_en_records = run_en_en_pairwise(arbitrage_candidates, client, worker_id)

        arb_pairs = [k for k, v in en_en_records.items() if v.get("is_arbitrage") is True]
        if arb_pairs:
            en_en_info = {"detected": True, "pairs": "; ".join(arb_pairs)}

        # All candidates are retained (no elimination in structural mode)
        for cand in arbitrage_candidates:
            for r in per_task_results:
                if r["index"] == cand["row_index"]:
                    r["en_en_arbitrage_info"] = en_en_info

    # ── Most-likely source identification ────────────────────
    if len(arbitrage_candidates) == 1:
        for r in per_task_results:
            if r["index"] == arbitrage_candidates[0]["row_index"]:
                r["most_likely_source"]   = True
                r["en_en_arbitrage_info"] = en_en_info

    elif len(arbitrage_candidates) > 1:
        print(f"    [Worker-{worker_id}] Identifying most likely source among {len(arbitrage_candidates)} candidate(s)...")
        try:
            chn_passage   = read_json_content(chn_filepath)
            likely_result = find_most_likely_source(chn_passage, arbitrage_candidates, client)

            if isinstance(likely_result, dict) and "api_error" in likely_result:
                likely_filename = "ERROR"
                print(f"    [Worker-{worker_id}] Most-likely source API error: {likely_result.get('api_error_detail')}")
            else:
                likely_filename = likely_result
                print(f"    [Worker-{worker_id}] Most likely source: {likely_filename}")

            for cand in arbitrage_candidates:
                is_most = "ERROR" if likely_filename == "ERROR" else (cand["filename"] == likely_filename)
                for r in per_task_results:
                    if r["index"] == cand["row_index"]:
                        r["most_likely_source"]   = is_most
                        r["en_en_arbitrage_info"] = en_en_info

        except Exception as exc:
            print(f"    [Worker-{worker_id}] Most-likely source identification failed: {exc}")
            for cand in arbitrage_candidates:
                for r in per_task_results:
                    if r["index"] == cand["row_index"]:
                        r["most_likely_source"]   = "ERROR"
                        r["en_en_arbitrage_info"] = en_en_info

    return {
        "chn_filepath":            chn_filepath,
        "results":                 per_task_results,
        "en_en_arbitrage_records": en_en_records,
    }


# ============================================================
# Result Writing
# ============================================================

_save_lock = threading.Lock()


def write_result_row(
    df: pd.DataFrame,
    row_index: int,
    final_result: dict,
    most_likely_source,
    skip_reason: str,
    output_path: str,
    round1: Optional[dict] = None,
    round2: Optional[dict] = None,
    consistency: Optional[dict] = None,
    en_en_info: Optional[dict] = None,
    api_error: Optional[str] = None,
    api_error_detail: Optional[str] = None,
) -> None:
    """
    Write one row's arbitrage verdict into df and save to CSV.
    Handles three cases: skip, processing error, and full result.
    """
    _SKIP_COLS = [
        "research_question", "research_design", "counterfactual_analysis", "arbitrage",
        "most_likely_source", "round1_rq", "round1_rd", "round1_mech",
        "round2_rq", "round2_rd", "round2_mech",
        "consistency_rq", "consistency_rd", "consistency_mech",
    ]

    if skip_reason:
        for col in _SKIP_COLS:
            df.at[row_index, col] = "skipped"
        df.at[row_index, "skip_reason"] = skip_reason

    elif final_result.get("error"):
        for col in _SKIP_COLS:
            df.at[row_index, col] = "error"
        df.at[row_index, "skip_reason"] = final_result.get("error", "unknown_error")

    else:
        rq   = final_result.get("research_question_arbitrage",       False)
        rd   = final_result.get("research_design_arbitrage",         False)
        mech = final_result.get("counterfactual_analysis_arbitrage", False)

        df.at[row_index, "research_question"]      = rq
        df.at[row_index, "research_design"]        = rd
        df.at[row_index, "counterfactual_analysis"] = mech
        df.at[row_index, "arbitrage"] = (
            "ERROR" if "ERROR" in (rq, rd, mech) else all([rq, rd, mech])
        )
        df.at[row_index, "most_likely_source"] = most_likely_source
        df.at[row_index, "skip_reason"]        = ""

        if round1:
            if "api_error" in round1:
                df.at[row_index, "round1_rq"] = df.at[row_index, "round1_rd"] = df.at[row_index, "round1_mech"] = "ERROR"
            else:
                df.at[row_index, "round1_rq"]   = round1.get("research_question_arbitrage",       False)
                df.at[row_index, "round1_rd"]   = round1.get("research_design_arbitrage",         False)
                df.at[row_index, "round1_mech"] = round1.get("counterfactual_analysis_arbitrage", False)

        if round2:
            if "api_error" in round2:
                df.at[row_index, "round2_rq"] = df.at[row_index, "round2_rd"] = df.at[row_index, "round2_mech"] = "ERROR"
            else:
                df.at[row_index, "round2_rq"]   = round2.get("research_question_arbitrage",       False)
                df.at[row_index, "round2_rd"]   = round2.get("research_design_arbitrage",         False)
                df.at[row_index, "round2_mech"] = round2.get("counterfactual_analysis_arbitrage", False)

        if consistency:
            df.at[row_index, "consistency_rq"]   = consistency.get("research_question",       "consistent")
            df.at[row_index, "consistency_rd"]    = consistency.get("research_design",         "consistent")
            df.at[row_index, "consistency_mech"]  = consistency.get("counterfactual_analysis", "consistent")

    if en_en_info:
        df.at[row_index, "en_en_arbitrage_detected"] = en_en_info.get("detected", False)
        df.at[row_index, "en_en_arbitrage_pairs"]    = en_en_info.get("pairs",    "")

    if api_error:
        df.at[row_index, "api_error"] = api_error
    if api_error_detail:
        df.at[row_index, "api_error_detail"] = api_error_detail

    save_csv(df, output_path)


# ============================================================
# Checkpoint Discovery
# ============================================================

def find_latest_checkpoint(input_csv_path: str, output_dir: str) -> tuple[str | None, str | None]:
    """
    Scan output_dir for a checkpoint matching the input file's base name.
    Returns (output_csv_path, checkpoint_path) for the most recently modified pair,
    or (None, None) if none found.
    """
    base    = os.path.splitext(os.path.basename(input_csv_path))[0]
    pattern = os.path.join(output_dir, f"{base}_arbitrage_results_*_checkpoint.json")
    hits    = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    if hits:
        checkpoint = hits[0]
        output     = checkpoint.replace("_checkpoint.json", ".csv")
        if os.path.exists(output):
            return output, checkpoint

    return None, None


# ============================================================
# Summary Statistics
# ============================================================

def print_statistics(df: pd.DataFrame) -> None:
    """Print a concise summary of arbitrage detection outcomes."""
    total = len(df)
    print("\n=== Statistics ===")

    if "skip_reason" in df.columns:
        skipped  = (df["skip_reason"].astype(str) != "").sum()
        analyzed = total - skipped
        print(f"  Total rows : {total}  |  Skipped: {skipped}  |  Analyzed: {analyzed}")

        top_reasons = df[df["skip_reason"].astype(str) != ""]["skip_reason"].value_counts().head(10)
        if not top_reasons.empty:
            print("\n  Skip reason breakdown:")
            for reason, cnt in top_reasons.items():
                print(f"    {reason}: {cnt}")

    if "arbitrage" in df.columns:
        print(f"\n  Arbitrage verdicts:")
        print(f"    Detected     : {(df['arbitrage'] == True).sum()}")
        print(f"    Not detected : {(df['arbitrage'] == False).sum()}")
        print(f"    API error    : {(df['arbitrage'] == 'ERROR').sum()}")

    if "api_error" in df.columns:
        api_errs = (df["api_error"].astype(str) != "").sum()
        print(f"\n  Rows with API errors: {api_errs}")

    if "en_en_arbitrage_detected" in df.columns:
        print(f"\n  En-En arbitrage detected: {(df['en_en_arbitrage_detected'] == True).sum()} row(s).")

    if "consistency_rq" in df.columns:
        print(f"\n  Two-round consistency:")
        for dim, label in [("rq", "RQ"), ("rd", "RD"), ("mech", "CF")]:
            col = f"consistency_{dim}"
            if col in df.columns:
                c = (df[col] == "consistent").sum()
                i = (df[col] == "inconsistent").sum()
                e = (df[col] == "error").sum()
                print(f"    {label}: consistent={c}  inconsistent={i}  error={e}")


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(input_csv_path: str, output_dir: str) -> str:
    """
    Full arbitrage detection pipeline for structural / non-measurement paper pairs.
    Returns the path to the output CSV.
    """
    print(f"\n{'=' * 60}")
    print(f"Starting pipeline: {os.path.basename(input_csv_path)}")
    print(f"Worker threads  : {NUM_WORKERS}")
    print(f"En-En strategy  : all-pairs combinatorial (no elimination)")
    print(f"Filter          : Structural + Non-Measurement Article (both papers)")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Resume-from-checkpoint or fresh start ────────────────
    existing_csv, existing_ckpt = find_latest_checkpoint(input_csv_path, output_dir)

    if existing_csv and existing_ckpt:
        print(f"\n  Checkpoint detected — resuming from: {existing_csv}")
        try:
            df = read_csv(existing_csv)
            output_path = existing_csv
        except Exception as exc:
            print(f"  WARNING: Could not load existing output ({exc}); starting fresh.")
            existing_csv = None

    if not existing_csv:
        print("\n  No checkpoint found — starting from scratch.")
        try:
            df = read_csv(input_csv_path)
        except Exception as exc:
            print(f"  ERROR: Could not read input CSV: {exc}")
            return ""

        df = ensure_result_columns(df)
        base      = os.path.splitext(os.path.basename(input_csv_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{base}_arbitrage_results_{timestamp}.csv")
        save_csv(df, output_path)
        print(f"  Output file created: {output_path}")

    checkpoint = CheckpointManager(output_path)

    # ── Phase 1: Pre-processing ───────────────────────────────
    print("\n[Phase 1] Pre-processing — grouping by Chinese paper...")

    chn_groups:    defaultdict = defaultdict(list)
    skipped_tasks: List[Dict]  = []
    already_done   = 0

    row_iter = (
        tqdm(df.iterrows(), total=len(df), desc="Pre-processing", unit="row")
        if TQDM_AVAILABLE else df.iterrows()
    )

    for idx, row in row_iter:
        if checkpoint.is_done(idx):
            already_done += 1
            continue

        chn_path = str(row.get("cn_json_path", "")).strip() if pd.notna(row.get("cn_json_path")) else ""
        eng_path = str(row.get("en_json_path", "")).strip() if pd.notna(row.get("en_json_path")) else ""
        chn_type = row.get("paper_type",    "")
        eng_type = row.get("en_paper_type", "")
        chn_meas = row.get("measurement",    "")
        eng_meas = row.get("en_measurement", "")
        eng_year = row.get("Year",           "9999")

        should, reason = should_analyze_pair(chn_type, eng_type, chn_meas, eng_meas)

        if not should:
            skipped_tasks.append({"index": idx, "skip_reason": reason})
            continue
        if not chn_path or chn_path == "nan":
            skipped_tasks.append({"index": idx, "skip_reason": "cn_json_path_empty"})
            continue
        if not eng_path or eng_path == "nan":
            skipped_tasks.append({"index": idx, "skip_reason": "en_json_path_empty"})
            continue
        if not os.path.exists(chn_path):
            skipped_tasks.append({"index": idx, "skip_reason": "cn_file_not_found"})
            continue
        if not os.path.exists(eng_path):
            skipped_tasks.append({"index": idx, "skip_reason": "en_file_not_found"})
            continue

        chn_groups[chn_path].append({
            "index":        idx,
            "chn_filepath": chn_path,
            "eng_filepath": eng_path,
            "eng_year":     eng_year,
        })

    # Write skipped rows immediately
    print("  Saving skipped rows...")
    for t in skipped_tasks:
        df.at[t["index"], "skip_reason"]      = t["skip_reason"]
        for col in ("research_question", "research_design", "counterfactual_analysis",
                    "arbitrage", "most_likely_source"):
            df.at[t["index"], col] = "skipped"
        checkpoint.mark(t["index"])

    save_csv(df, output_path)
    checkpoint.save()

    total_to_analyze = sum(len(v) for v in chn_groups.values())
    print(f"  Already processed (checkpoint) : {already_done}")
    print(f"  Skipped                        : {len(skipped_tasks)}")
    print(f"  To analyze                     : {total_to_analyze} row(s) across {len(chn_groups)} Chinese paper(s).")

    if not chn_groups:
        print("  No tasks to process.")
        checkpoint.cleanup()
        return output_path

    # ── Phase 2: Parallel processing ─────────────────────────
    print(f"\n[Phase 2] Parallel processing ({NUM_WORKERS} worker(s))...")

    all_en_en_records: Dict = {}
    chn_list = list(chn_groups.items())

    def _process(args: tuple) -> Dict:
        group_idx, (chn_filepath, tasks) = args
        worker_id = group_idx % NUM_WORKERS
        client    = api_clients[worker_id]
        return process_chinese_paper_group(chn_filepath, tasks, client, worker_id)

    pbar = (
        tqdm(total=len(chn_list), desc="Processing Chinese papers", unit="paper")
        if TQDM_AVAILABLE else None
    )

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(_process, (i, item)): item[0]
            for i, item in enumerate(chn_list)
        }

        for future in as_completed(futures):
            chn_filepath = futures[future]
            try:
                group = future.result()
                all_en_en_records.update(group.get("en_en_arbitrage_records", {}))

                with _save_lock:
                    for r in group["results"]:
                        if r.get("error"):
                            write_result_row(
                                df, r["index"], {"error": r["error"]}, False, "",
                                output_path,
                                api_error=r.get("api_error"),
                                api_error_detail=r.get("api_error_detail"),
                            )
                        else:
                            write_result_row(
                                df, r["index"], r["final_result"],
                                r.get("most_likely_source", False), "",
                                output_path,
                                round1=r["round1_result"],
                                round2=r["round2_result"],
                                consistency=r["consistency_check"],
                                en_en_info=r.get("en_en_arbitrage_info", {"detected": False, "pairs": ""}),
                                api_error=r.get("api_error"),
                                api_error_detail=r.get("api_error_detail"),
                            )
                        checkpoint.mark(r["index"])
                    checkpoint.save()

            except Exception as exc:
                print(f"  ERROR processing {os.path.basename(chn_filepath)}: {exc}")
                traceback.print_exc()

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    with _save_lock:
        save_csv(df, output_path)

    # ── Save en-en summary ────────────────────────────────────
    if all_en_en_records:
        summary_path = output_path.replace(".csv", "_en_en_arbitrage_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(all_en_en_records, fh, ensure_ascii=False, indent=2)
        print(f"\n  En-En arbitrage summary saved to: {summary_path}")

    checkpoint.cleanup()

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete. Results saved to: {output_path}")
    print_statistics(df)
    return output_path


# ============================================================
# Entry Point
# ============================================================

def main() -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        return

    print(f"Input           : {os.path.basename(INPUT_PATH)}")
    print(f"API keys        : {len(API_KEYS)}")
    print(f"En-En strategy  : all-pairs combinatorial (structural mode)")
    print(f"Notes: ERROR values are recorded directly; checkpoint resumes on re-run.")

    try:
        output = run_pipeline(INPUT_PATH, OUTPUT_DIR)
        if output:
            print(f"\n✓ Done. Output: {output}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress has been saved.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

