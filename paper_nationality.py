# -*- coding: utf-8 -*-
"""
Study Area Country Identification
Reads paper metadata from an Excel file, extracts introductory text from JSON files,
and uses an LLM to identify the country of data origin for each paper.

Only rows where the 'arbitrage' column equals TRUE are processed.
Results are written to two new columns: cn_country and en_country.

Usage:
    Set MOONSHOT_API_KEY, INPUT_FILE, and OUTPUT_FILE, then run:
        python identify_country.py
"""

import json
import os

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
MOONSHOT_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

INPUT_FILE  = "Arbitrage_results.xlsx"
OUTPUT_FILE = "Arbitrage_results_nationality.xlsx"

BASE_URL  = "https://api.moonshot.cn/v1"
MODEL     = "kimi-k2-0711-preview"

TEMPERATURE   = 0.0
MAX_TEXT_LEN  = 15_000   # Characters — input is truncated beyond this limit
SAVE_EVERY    = 10        # Persist the output file every N processed rows

# Normalisation map: uppercased raw LLM output → standard country name
COUNTRY_MAP: dict[str, str] = {
    "UNITED STATES":              "USA",
    "UNITED STATES OF AMERICA":   "USA",
    "US":                         "USA",
    "AMERICA":                    "USA",
    "UNITED KINGDOM":             "UK",
    "GREAT BRITAIN":              "UK",
    "ENGLAND":                    "UK",
    "PEOPLES REPUBLIC OF CHINA":  "CHINA",
    "PRC":                        "CHINA",
    "OECD":                       "Multiple Countries",
    "EU":                         "Multiple Countries",
    "GLOBAL":                     "Multiple Countries",
    "INTERNATIONAL":              "Multiple Countries",
}

# ============================================================
# API Client
# ============================================================
client = OpenAI(api_key=MOONSHOT_API_KEY, base_url=BASE_URL)


# ============================================================
# JSON Reader
# ============================================================

def extract_intro_content(file_path: str) -> str | None:
    """
    Load a JSON file and return the content of the first 'introductory-part' item.
    Handles both list-of-blocks and dict-with-nested-list structures.
    Returns None if the file is missing, unreadable, or contains no such item.
    """
    if not isinstance(file_path, str) or not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("type") == "introductory-part":
                    return str(item.get("content", "")).strip() or None

        elif isinstance(data, dict):
            if data.get("type") == "introductory-part":
                return str(data.get("content", "")).strip() or None
            for value in data.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and item.get("type") == "introductory-part":
                            return str(item.get("content", "")).strip() or None

    except Exception:
        pass

    return None


# ============================================================
# LLM-based Country Identification
# ============================================================

def identify_country(text: str) -> str:
    """
    Ask the LLM to identify the country of data origin from the given text.

    Applies a post-processing normalisation step to standardise common variants
    (e.g. "United States" → "USA", "OECD" → "Multiple Countries").

    Returns a standardised country name, "No Content" if text is empty,
    or "API Error" if the call fails.
    """
    if not text:
        return "No Content"

    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                       XXXXX
                    ),
                },
                {
                    "role": "user",
                    "content": f"Text: {text}\n\nQuestion: Which country does the data come from?",
                },
            ],
            temperature=TEMPERATURE,
        )

        raw = completion.choices[0].message.content.strip()
        return _normalise_country(raw)

    except Exception as exc:
        print(f"  API Error: {exc}")
        return "API Error"


def _normalise_country(raw: str) -> str:
    """
    Normalise the raw LLM response to a standard country label.
    Strips trailing punctuation, upper-cases for lookup, then applies COUNTRY_MAP.
    Falls back to the original (capitalised) string if no mapping is found.
    """
    cleaned = raw.replace(".", "").upper()

    if cleaned in COUNTRY_MAP:
        return COUNTRY_MAP[cleaned]
    if "MULTIPLE" in cleaned:
        return "Multiple Countries"
    return raw  # Preserve the model's original capitalisation


# ============================================================
# I/O Helpers
# ============================================================

def load_input(file_path: str) -> pd.DataFrame | None:
    """Load the input Excel file and validate required columns."""
    print(f"Loading input file: {file_path}")
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"ERROR: file not found — {file_path}")
        return None

    required = ("cn_json_path", "en_json_path", "arbitrage")
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing required column(s): {missing}")
        return None

    return df


def save_output(df: pd.DataFrame, file_path: str) -> None:
    """Save the DataFrame to an Excel file."""
    df.to_excel(file_path, index=False)


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add cn_country and en_country columns if they do not already exist."""
    for col in ("cn_country", "en_country"):
        if col not in df.columns:
            df[col] = None
    return df


def needs_classification(value) -> bool:
    """Return True if a country cell has not yet been filled in."""
    return value is None or pd.isna(value)


# ============================================================
# Main
# ============================================================

def main() -> None:
    # ── Load ────────────────────────────────────────────────
    df = load_input(INPUT_FILE)
    if df is None:
        return

    df = ensure_output_columns(df)

    print("Starting country identification (only rows where arbitrage = TRUE)...")

    process_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Identifying countries", unit="row"):
        # Filter: only process arbitrage = TRUE rows
        arbitrage_val = str(row["arbitrage"]).strip().upper() if pd.notna(row["arbitrage"]) else "FALSE"
        if arbitrage_val != "TRUE":
            continue

        process_count += 1

        cn_text = extract_intro_content(row["cn_json_path"])
        en_text = extract_intro_content(row["en_json_path"])

        # Chinese file — identify country if not yet classified
        if needs_classification(row["cn_country"]):
            df.at[idx, "cn_country"] = identify_country(cn_text) if cn_text else "Content Missing"

        # English file — identify country if not yet classified
        if needs_classification(row["en_country"]):
            df.at[idx, "en_country"] = identify_country(en_text) if en_text else "Content Missing"

        # Periodic save to guard against interruptions
        if process_count % SAVE_EVERY == 0:
            save_output(df, OUTPUT_FILE)

    # ── Final save ───────────────────────────────────────────
    save_output(df, OUTPUT_FILE)
    print(f"\nDone. Processed {process_count} row(s). Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

