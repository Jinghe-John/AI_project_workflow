#!/usr/bin/env python3
"""
English Paper JSON Batch Processor
Reads paper metadata from a CSV file, locates the corresponding JSON files
produced by MinerU, and performs the following operations on each file:

  1. Filter out pre-1980 entries.
  2. Clean JSON: retain only 'text', 'equation', and 'title' blocks.
  3. Organise content by section (merge consecutive text/equation blocks).
  4. Match Ref_Title against internal title blocks (three-tier strategy).
  5. Trim content to start at the matched title.
  6. Build an 'introductory-part' block from the paper header.
  7. Save the processed JSON and write a CSV processing report.

Usage:
    python process_en_json.py -i all_pdf_info.csv -j ./en_output -o ./en_processed_json
    python process_en_json.py -i all_pdf_info.csv -j ./en_output -o ./en_processed_json -c ./report.csv
"""

import argparse
import json
import os
import string
import traceback
import unicodedata

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

# ============================================================
# Constants
# ============================================================
KEEP_TYPES       = {"text", "equation", "title"}
FUZZY_THRESHOLD  = 80    # Minimum fuzz.ratio score to count as a fuzzy match
SUBSET_TOLERANCE = 0.05  # Maximum relative length difference for subset matching
CHUNK_MATCH_RATIO = 0.8  # Fraction of chunks that must appear in the longer string
MIN_CHUNK_LEN    = 5
CHUNK_DIVISIONS  = 5
MIN_SHORTER_LEN  = 10    # Minimum length to attempt chunk matching
YEAR_CUTOFF      = 1980  # Entries published before this year are excluded
OUTPUT_SUFFIX    = "_processed"
DEFAULT_CSV_NAME = "processed_results.csv"

# Mapping from internal result keys → human-readable column names
RESULT_COLUMNS = {
    "status":               "status",
    "match_type":           "match_type",
    "match_method":         "match_method",
    "match_count":          "match_count",
    "removed_headers":      "removed_headers",
    "trim_note":            "trim_note",
    "start_header":         "start_header",
    "headers_before_trim":  "headers_before_trim",
    "headers_after_trim":   "headers_after_trim",
    "intro_converted":      "intro_converted",
    "used_fallback":        "used_fallback",
    "conversion_note":      "conversion_note",
    "has_intro_part":       "has_intro_part",
    "output_json_path":     "output_json_path",
    "error":                "error",
}

# Unicode special character maps (applied during normalisation)
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUBSCRIPT_MAP   = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SPECIAL_SYMBOLS = "†‡§¶*#•·°"
_QUOTE_CHARS     = '"""\'\'„‟‹›«»\u201c\u201d\u2018\u2019'
_CHINESE_PUNCT   = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
    "｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏.·"
)


# ============================================================
# Text Normalisation
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalise a title string for comparison:
      - Replace unicode quotes, superscripts, subscripts, and special symbols
      - Lowercase and NFKC-normalise
      - Strip all punctuation (ASCII and Chinese) and whitespace
    """
    if not text:
        return ""

    text = str(text)

    for char in _QUOTE_CHARS:
        text = text.replace(char, " ")

    text = text.translate(_SUPERSCRIPT_MAP)
    text = text.translate(_SUBSCRIPT_MAP)

    for sym in _SPECIAL_SYMBOLS:
        text = text.replace(sym, "")

    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    for punct in _CHINESE_PUNCT:
        text = text.replace(punct, "")

    return "".join(text.split())


# ============================================================
# Title Matching
# ============================================================

def _flexible_substring_match(str1: str, str2: str) -> tuple[bool, str]:
    """
    Attempt a flexible subset / chunk-based match between two strings.
    Both strings are normalised before comparison.

    Returns (matched: bool, method_description: str).
    """
    if not str1 or not str2:
        return False, ""

    norm1 = normalize_text(str1)
    norm2 = normalize_text(str2)

    if not norm1 or not norm2:
        return False, ""

    if norm1 == norm2:
        return True, "exact_substring"
    if norm1 in norm2:
        return True, "str1_in_str2"
    if norm2 in norm1:
        return True, "str2_in_str1"

    # Length-similar strings: use edit distance
    len_diff = abs(len(norm1) - len(norm2))
    max_len  = max(len(norm1), len(norm2))
    if max_len > 0 and len_diff / max_len <= SUBSET_TOLERANCE:
        similarity = fuzz.ratio(norm1, norm2)
        if similarity >= 90:
            return True, f"flexible_match(sim={similarity})"

    # Chunk-based containment check
    shorter = norm1 if len(norm1) < len(norm2) else norm2
    longer  = norm2 if len(norm1) < len(norm2) else norm1

    if len(shorter) > MIN_SHORTER_LEN:
        chunk_size = max(MIN_CHUNK_LEN, len(shorter) // CHUNK_DIVISIONS)
        chunks  = [shorter[i:i + chunk_size] for i in range(0, len(shorter), chunk_size) if shorter[i:i + chunk_size]]
        matches = sum(1 for chunk in chunks if chunk in longer)
        if chunks and matches / len(chunks) >= CHUNK_MATCH_RATIO:
            return True, "partial_chunks_match"

    return False, ""


def match_title(
    reference_title: str,
    section_titles: list[str],
    fuzzy_threshold: int = FUZZY_THRESHOLD,
) -> tuple[list[int], str, str]:
    """
    Match a reference title against a list of section titles using a three-tier strategy:
      1. Exact match (after normalisation)
      2. Fuzzy match  (fuzz.ratio ≥ fuzzy_threshold)
      3. Subset / chunk match

    Returns:
        (matched_indices, match_type, match_method)
        match_type  ∈ {'one_to_one', 'one_to_many', 'no_match'}
        match_method ∈ {'exact', 'fuzzy', 'subset', 'none'}
    """
    if not reference_title or not section_titles:
        return [], "no_match", "none"

    norm_ref     = normalize_text(reference_title)
    norm_headers = [normalize_text(h) for h in section_titles]

    # Tier 1 — exact
    exact_hits = [i for i, h in enumerate(norm_headers) if norm_ref and h and norm_ref == h]
    if exact_hits:
        return exact_hits, ("one_to_one" if len(exact_hits) == 1 else "one_to_many"), "exact"

    if not norm_ref:
        return [], "no_match", "none"

    # Tier 2 — fuzzy
    fuzzy_hits = [
        (i, fuzz.ratio(norm_ref, h))
        for i, h in enumerate(norm_headers)
        if h and fuzz.ratio(norm_ref, h) >= fuzzy_threshold
    ]
    if fuzzy_hits:
        fuzzy_hits.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in fuzzy_hits]
        return indices, ("one_to_one" if len(indices) == 1 else "one_to_many"), "fuzzy"

    # Tier 3 — subset / chunk
    subset_hits = [
        i for i, h in enumerate(section_titles)
        if h and _flexible_substring_match(reference_title, h)[0]
    ]
    if subset_hits:
        return subset_hits, ("one_to_one" if len(subset_hits) == 1 else "one_to_many"), "subset"

    return [], "no_match", "none"


# ============================================================
# JSON Loading
# ============================================================

def load_json_file(json_path: str) -> list[dict]:
    """
    Read a JSON file and return a flat list of block dicts.
    Handles the MinerU list-of-lists format (one sub-array per page).

    Raises ValueError on JSON parse failure.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        raw = fh.read().strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}") from exc

    if isinstance(data, list) and data and isinstance(data[0], list):
        flat: list[dict] = []
        for sub in data:
            flat.extend(sub) if isinstance(sub, list) else flat.append(sub)
        return flat

    return data if isinstance(data, list) else []


# ============================================================
# JSON Cleaning & Structuring
# ============================================================

def clean_json(data: list[dict]) -> list[dict]:
    """
    Clean a block list:
      1. Keep only 'text', 'equation', and 'title' blocks.
      2. Skip all blocks before the first 'title'.
      3. Merge consecutive text/equation blocks within each section into one 'text' block.
    """
    # Pass 1: filter and collect
    filtered: list[dict] = []
    title_seen = False

    for item in data:
        item_type = item.get("type", "")
        content   = item.get("content")

        if content is None:
            continue
        content = str(content).strip()
        if not content or item_type not in KEEP_TYPES:
            continue

        if not title_seen:
            if item_type != "title":
                continue
            title_seen = True

        filtered.append({"type": item_type, "content": content})

    # Pass 2: merge text/equation by section
    structured: list[dict] = []
    accumulated: list[str] = []

    def _flush() -> None:
        if accumulated:
            structured.append({"type": "text", "content": " ".join(accumulated)})
            accumulated.clear()

    for item in filtered:
        if item["type"] == "title":
            _flush()
            structured.append(item)
        else:
            accumulated.append(item["content"])

    _flush()
    return structured


def extract_titles(data: list[dict]) -> list[str]:
    """
    Return a list of cleaned title strings from a block list.
    Strips leading #, *, and whitespace; collapses internal whitespace.
    """
    titles: list[str] = []
    for item in data:
        if item.get("type") == "title":
            text = item.get("content", "")
            text = text.replace("#", "").replace("*", "").replace("\n", " ").replace("\r", " ")
            text = " ".join(text.split())
            if text:
                titles.append(text)
    return titles


# ============================================================
# Content Trimming
# ============================================================

def trim_from_title(data: list[dict], title_index: int) -> list[dict]:
    """
    Return a slice of data starting at the title_index-th 'title' block.
    All blocks before it are discarded.
    """
    seen = 0
    for pos, item in enumerate(data):
        if item.get("type") == "title":
            if seen == title_index:
                return data[pos:]
            seen += 1
    return data


def remove_title_at(data: list[dict], title_index: int) -> list[dict]:
    """
    Remove the title_index-th 'title' block from data in-place and return data.
    """
    seen = 0
    for pos, item in enumerate(data):
        if item.get("type") == "title":
            if seen == title_index:
                data.pop(pos)
                return data
            seen += 1
    return data


# ============================================================
# Introductory-Part Generation
# ============================================================

def _collect_intro_content(
    data: list[dict],
    first_title_idx: int,
    intro_title_idx: int,
    next_title_idx: int,
) -> list[str]:
    """
    Collect text content between first_title_idx and intro_title_idx,
    append the intro title itself, then append text up to next_title_idx.
    """
    parts: list[str] = []

    for i in range(first_title_idx + 1, intro_title_idx):
        if data[i].get("type") == "text":
            parts.append(data[i]["content"])

    parts.append(data[intro_title_idx]["content"])

    end = next_title_idx if next_title_idx != -1 else len(data)
    for i in range(intro_title_idx + 1, end):
        if data[i].get("type") == "text":
            parts.append(data[i]["content"])

    return parts


def build_introductory_part(data: list[dict]) -> tuple[list[dict], bool, str]:
    """
    Primary strategy: locate a title that contains 'introduction' or starts with
    1 / 1. / I / I. and use it as the boundary for the Introductory-Part block.

    Returns (new_data, success, description).
    """
    if not data:
        return data, False, "Empty data."

    title_indices = [i for i, b in enumerate(data) if b.get("type") == "title"]
    if not title_indices:
        return data, False, "No title blocks found."

    first_idx  = title_indices[0]
    first_text = data[first_idx]["content"]

    # Find the introduction title
    intro_idx  = -1
    intro_text = ""
    for i in title_indices:
        if i <= first_idx:
            continue
        raw   = data[i].get("content", "").strip()
        clean = raw.lstrip("#").lstrip("*").strip()
        lower = clean.lower()
        if (
            "introduction" in lower
            or clean.startswith("1")
            or clean.startswith("1.")
            or clean.startswith("I")
            or clean.startswith("I.")
        ):
            intro_idx  = i
            intro_text = raw
            break

    if intro_idx == -1:
        return data, False, f"No introduction title found. First title: '{first_text}'."

    # Find the title immediately after the introduction
    next_idx = next((i for i in title_indices if i > intro_idx), -1)

    parts    = _collect_intro_content(data, first_idx, intro_idx, next_idx)
    new_data = [
        {"type": "Title",             "content": first_text},
        {"type": "introductory-part", "content": " ".join(parts)},
    ]
    if next_idx != -1:
        new_data.extend(data[next_idx:])

    return new_data, True, f"Introductory-part built; intro title: '{intro_text}'."


def build_introductory_part_fallback(data: list[dict]) -> tuple[list[dict], bool, str]:
    """
    Fallback strategy: select a target title based on the total number of titles:
      - 1 title  → use it as both Title and intro
      - 2 titles → use the last one
      - 3+ titles → use the third one

    Returns (new_data, success, description).
    """
    if not data:
        return data, False, "Empty data."

    title_indices = [i for i, b in enumerate(data) if b.get("type") == "title"]
    if not title_indices:
        return data, False, "No title blocks found."

    first_idx  = title_indices[0]
    first_text = data[first_idx]["content"]

    n = len(title_indices)
    if n == 1:
        target_idx  = title_indices[0]
        description = "only title (1 total)"
    elif n == 2:
        target_idx  = title_indices[-1]
        description = f"last title ({n} total)"
    else:
        target_idx  = title_indices[2]
        description = f"third title ({n} total)"

    target_text = data[target_idx]["content"]

    # Collect content between first and target titles
    parts: list[str] = []
    if n == 1:
        parts.append(first_text)
    else:
        for i in range(first_idx + 1, target_idx):
            b = data[i]
            if b.get("type") in ("text", "title"):
                parts.append(b["content"])
        parts.append(target_text)

    # Collect text immediately after the target title (until next title)
    target_pos   = title_indices.index(target_idx)
    next_idx     = title_indices[target_pos + 1] if target_pos + 1 < n else -1
    end          = next_idx if next_idx != -1 else len(data)
    for i in range(target_idx + 1, end):
        if data[i].get("type") == "text":
            parts.append(data[i]["content"])

    new_data = [
        {"type": "Title",             "content": first_text},
        {"type": "introductory-part", "content": " ".join(parts)},
    ]
    if next_idx != -1:
        new_data.extend(data[next_idx:])

    return new_data, True, f"Fallback: used {description}: '{target_text}'."


# ============================================================
# Per-file Processing
# ============================================================

def _empty_result(output_path: str) -> dict:
    return {
        "status":              "failed",
        "match_type":          "none",
        "match_method":        "none",
        "match_count":         0,
        "removed_headers":     0,
        "trim_note":           "",
        "start_header":        "",
        "headers_before_trim": 0,
        "headers_after_trim":  0,
        "intro_converted":     False,
        "used_fallback":       False,
        "conversion_note":     "",
        "has_intro_part":      False,
        "output_json_path":    output_path,
        "error":               "",
    }


def process_single_file(json_path: str, ref_title: str, output_path: str) -> dict:
    """
    Full processing pipeline for one JSON file.
    Returns a result dict summarising the outcome.
    """
    result = _empty_result(output_path)

    try:
        data = load_json_file(json_path)
        if not data:
            result["error"] = "Empty JSON file."
            return result

        cleaned = clean_json(data)
        if not cleaned:
            result["error"] = "No valid content after cleaning."
            return result

        titles_before = extract_titles(cleaned)
        result["headers_before_trim"] = len(titles_before)

        # ── Title matching ────────────────────────────────────
        matched, match_type, match_method = match_title(ref_title, titles_before)
        result["match_type"]   = match_type
        result["match_method"] = match_method
        result["match_count"]  = len(matched)

        # ── Content trimming ──────────────────────────────────
        trimmed         = cleaned
        removed_headers = 0

        if match_type == "one_to_one":
            start = matched[0]
            trimmed = trim_from_title(cleaned, start)
            result["start_header"] = titles_before[start]
            result["trim_note"]    = f"One-to-one {match_method} match; starting from title #{start + 1}."

        elif match_type == "one_to_many":
            start   = matched[0]
            trimmed = trim_from_title(cleaned, start)
            result["start_header"] = titles_before[start]

            # Remove duplicate matched titles (relative indices, descending order)
            for relative in sorted(
                [m - start for m in matched[1:] if m - start > 0], reverse=True
            ):
                trimmed = remove_title_at(trimmed, relative)
                removed_headers += 1

            result["removed_headers"] = removed_headers
            result["trim_note"] = (
                f"One-to-many {match_method} match ({len(matched)} hits); "
                f"using title #{start + 1}; {removed_headers} duplicate(s) removed."
            )
        else:
            result["trim_note"] = "No match — full content retained."

        # ── Introductory-part generation ──────────────────────
        intro_converted = used_fallback = False

        if match_type in ("one_to_one", "one_to_many"):
            converted, success, note = build_introductory_part(trimmed)
            if success:
                trimmed         = converted
                intro_converted = True
                result["conversion_note"] = note
            else:
                fallback, fb_ok, fb_note = build_introductory_part_fallback(trimmed)
                if fb_ok:
                    trimmed         = fallback
                    intro_converted = True
                    used_fallback   = True
                    result["conversion_note"] = f"Fallback used: {fb_note}"
                else:
                    result["conversion_note"] = (
                        f"Primary failed: {note}  Fallback failed: {fb_note}"
                    )

        result["intro_converted"] = intro_converted
        result["used_fallback"]   = used_fallback
        result["has_intro_part"]  = any(
            b.get("type") == "introductory-part" for b in trimmed
        )
        result["headers_after_trim"] = len(extract_titles(trimmed))

        # ── Write output ──────────────────────────────────────
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(trimmed, fh, ensure_ascii=False, indent=2)

        result["status"] = "success"

    except Exception as exc:
        result["error"]  = str(exc)
        result["status"] = "failed"

    return result


# ============================================================
# CSV I/O
# ============================================================

def read_csv(path: str) -> pd.DataFrame:
    """
    Attempt to read a CSV file using UTF-8, then GBK as a fallback.
    Raises RuntimeError if both fail.
    """
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not read '{path}' with UTF-8 or GBK encoding.")


def save_report(records: list[dict], csv_path: str) -> None:
    """Write the full result table to a UTF-8-BOM CSV."""
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")


# ============================================================
# Summary Printing
# ============================================================

def print_summary(df: pd.DataFrame, output_dir: str, csv_path: str) -> None:
    """Print a formatted processing summary to stdout."""
    total     = len(df)
    success   = (df["status"] == "success").sum()
    failed    = (df["status"] == "failed").sum()
    not_found = (df["status"] == "file_not_found").sum()
    pct       = lambda n: f"{n / total * 100:.1f}%" if total else "0%"

    print("\n" + "=" * 80)
    print("Processing complete — Summary")
    print("=" * 80)
    print(f"  Total records    : {total}")
    print(f"  Succeeded        : {success}  ({pct(success)})")
    print(f"  Failed           : {failed}  ({pct(failed)})")
    print(f"  File not found   : {not_found}  ({pct(not_found)})")

    ok = df[df["status"] == "success"]
    if not ok.empty:
        print("\n  Match type distribution:")
        for mtype, cnt in ok["match_type"].value_counts().items():
            print(f"    {mtype}: {cnt}")

        intro_yes   = ok["intro_converted"].sum()
        fallback    = ok["used_fallback"].sum()
        intro_no    = len(ok) - intro_yes

        print(f"\n  Introductory-part conversion:")
        print(f"    Converted (primary strategy) : {intro_yes - fallback}")
        print(f"    Converted (fallback strategy): {fallback}")
        print(f"    Not converted                : {intro_no}")

    print(f"\n  Processed JSON  : {output_dir}")
    print(f"  Result CSV      : {csv_path}")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process English MinerU JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s -i all_pdf_info.csv -j ./en_output -o ./en_processed_json\n"
            "  %(prog)s -i all_pdf_info.csv -j ./en_output -o ./en_processed_json -c ./report.csv\n"
        ),
    )
    parser.add_argument("-i", "--input_csv",      required=True, help="Input CSV file path.")
    parser.add_argument("-j", "--input_json_dir", required=True, help="Input JSON directory.")
    parser.add_argument("-o", "--output_json_dir",required=True, help="Output JSON directory.")
    parser.add_argument(
        "-c", "--output_csv", default=None,
        help=f"Output CSV path (default: <output_json_dir>/{DEFAULT_CSV_NAME}).",
    )
    return parser.parse_args()


def main() -> None:
    args          = parse_args()
    input_csv     = args.input_csv
    json_dir      = args.input_json_dir
    output_dir    = args.output_json_dir
    csv_out       = args.output_csv or os.path.join(output_dir, DEFAULT_CSV_NAME)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("English Paper JSON Batch Processor")
    print("=" * 80)
    print(f"  Input CSV       : {input_csv}")
    print(f"  Input JSON dir  : {json_dir}")
    print(f"  Output JSON dir : {output_dir}")
    print(f"  Output CSV      : {csv_out}")
    print("=" * 80)

    # ── Load & validate CSV ──────────────────────────────────
    print("\nReading input CSV...")
    try:
        df = read_csv(input_csv)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return

    print(f"  Loaded {len(df)} rows.")

    required_cols = ("pdf_id", "Year", "Ref_Title")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: missing required column(s): {missing}")
        print(f"  Available columns: {list(df.columns)}")
        return

    # ── Filter pre-YEAR_CUTOFF entries ───────────────────────
    original_count = len(df)
    df["_year_num"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["_year_num"] >= YEAR_CUTOFF].copy()
    print(f"  Filtered {original_count - len(df)} pre-{YEAR_CUTOFF} entries. Remaining: {len(df)}.")

    # ── Process each row ─────────────────────────────────────
    records: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files", unit="file"):
        pdf_id    = str(row["pdf_id"])
        ref_title = str(row["Ref_Title"]) if pd.notna(row["Ref_Title"]) else ""
        year      = pdf_id[:4]  # First 4 characters are the publication year

        json_path   = os.path.join(json_dir,   year, f"{pdf_id}.json")
        output_path = os.path.join(output_dir, year, f"{pdf_id}{OUTPUT_SUFFIX}.json")

        record = row.to_dict()
        record["source_json_path"] = json_path

        if not os.path.exists(json_path):
            record["status"]           = "file_not_found"
            record["output_json_path"] = ""
            records.append(record)
            continue

        result = process_single_file(json_path, ref_title, output_path)
        record.update(result)
        records.append(record)

    # ── Save report & print summary ──────────────────────────
    print("\nSaving result report...")
    save_report(records, csv_out)
    print_summary(pd.DataFrame(records), output_dir, csv_out)


if __name__ == "__main__":
    main()
