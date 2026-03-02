#!/usr/bin/env python3
"""
Chinese JSON Batch Processor
Cleans and restructures JSON files exported from MinerU by:
  1. Retaining only blocks of type: title, text, equation, page_footnote
  2. Promoting the first title to 'Title' and merging the introductory section
     into a single 'Introductory-Part' block
  3. Keeping page_footnote blocks independent (not merged into body text)
  4. Preserving the original folder hierarchy in the output directory
  5. Flagging files that contain no Chinese characters as anomalous

Usage:
    python process_cn_json.py -i <input_dir> -o <output_dir> [-c <report.csv>]
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ============================================================
# Constants
# ============================================================
KEEP_TYPES       = {"text", "equation", "title", "page_footnote"}
CHINESE_PATTERN  = re.compile(r"[\u4e00-\u9fff]")
OUTPUT_SUFFIX    = "_processed"          # Appended to every output filename stem
DEFAULT_CSV_NAME = "cn_processed_results.csv"

# Result record keys (used as DataFrame columns)
RESULT_KEYS = (
    "status",
    "has_chinese",
    "chinese_char_count",
    "is_anomalous",
    "count_Title",
    "count_Introductory-Part",
    "count_title",
    "count_text",
    "count_page_footnote",
    "total_blocks",
    "error",
    "source_path",
    "output_path",
    "filename",
)


# ============================================================
# Chinese-language Detection
# ============================================================

def contains_chinese(text: str) -> bool:
    """Return True if text contains at least one Chinese character."""
    return bool(text and CHINESE_PATTERN.search(str(text)))


def count_chinese_chars(data: list[dict]) -> tuple[bool, int]:
    """
    Count Chinese characters across all 'content' fields in a block list.
    Returns (has_chinese: bool, total_count: int).
    """
    total = sum(
        len(CHINESE_PATTERN.findall(str(item.get("content", ""))))
        for item in data
    )
    return total > 0, total


# ============================================================
# JSON Loading
# ============================================================

def load_json_file(json_path: str) -> list[dict]:
    """
    Read a JSON file and return a flat list of block dicts.
    Handles the MinerU format where the top-level array may itself contain
    per-page sub-arrays (i.e. a list-of-lists).

    Raises ValueError on JSON parse failure.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        raw = fh.read().strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}") from exc

    # Flatten list-of-lists (one sub-array per page)
    if isinstance(data, list) and data and isinstance(data[0], list):
        flat: list[dict] = []
        for sub in data:
            flat.extend(sub) if isinstance(sub, list) else flat.append(sub)
        return flat

    return data if isinstance(data, list) else []


# ============================================================
# JSON Cleaning & Restructuring
# ============================================================

def _strip_title_markers(text: str) -> str:
    """Remove markdown-style * and # markers from a title string."""
    return text.replace("*", "").replace("#", "").strip()


def _extract_valid_blocks(data: list[dict]) -> list[dict]:
    """
    Pass 1: Filter to kept types, strip empty content, clean title markers,
    and skip everything that appears before the first title.
    """
    blocks: list[dict] = []
    title_found = False

    for item in data:
        item_type = item.get("type", "")
        content   = item.get("content")

        if content is None:
            continue
        content = str(content).strip()
        if not content or item_type not in KEEP_TYPES:
            continue

        if not title_found:
            if item_type != "title":
                continue
            title_found = True

        if item_type == "title":
            content = _strip_title_markers(content)

        blocks.append({"type": item_type, "content": content})

    return blocks


def _merge_leading_titles(blocks: list[dict]) -> list[dict]:
    """
    Pass 2: If the first two blocks are both titles (no text between them),
    concatenate them into a single title block.
    """
    if (
        len(blocks) >= 2
        and blocks[0]["type"] == "title"
        and blocks[1]["type"] == "title"
    ):
        blocks[0]["content"] = blocks[0]["content"] + " " + blocks[1]["content"]
        blocks.pop(1)
    return blocks


def _build_introductory_section(blocks: list[dict]) -> list[dict]:
    """
    Pass 3: Promote the first 'title' to 'Title' and merge everything between
    the first and third title into a single 'Introductory-Part' block.
    page_footnote items are preserved individually after the Introductory-Part.

    If there is no third title, all remaining blocks become the Introductory-Part.
    """
    if not blocks or blocks[0]["type"] != "title":
        return blocks

    blocks[0]["type"] = "Title"

    # Find indices of the 2nd and 3rd titles
    title_indices = [i for i, b in enumerate(blocks) if b["type"] == "title"]
    second_title_idx = title_indices[0] if title_indices else None
    third_title_idx  = title_indices[1] if len(title_indices) > 1 else None

    end_of_intro = third_title_idx if third_title_idx is not None else len(blocks)

    intro_texts: list[str] = []
    footnotes:   list[dict] = []

    for i in range(1, end_of_intro):
        b = blocks[i]
        if b["type"] == "page_footnote":
            footnotes.append(b)
        else:
            intro_texts.append(b["content"])

    new_blocks: list[dict] = [blocks[0]]

    if intro_texts:
        new_blocks.append({
            "type":    "Introductory-Part",
            "content": " ".join(intro_texts),
        })

    new_blocks.extend(footnotes)

    if third_title_idx is not None:
        new_blocks.extend(blocks[third_title_idx:])

    return new_blocks


def _merge_body_text(blocks: list[dict]) -> list[dict]:
    """
    Pass 4: Merge consecutive text/equation blocks within each section into a
    single 'text' block. Title-family and page_footnote blocks act as flush
    boundaries and are emitted unchanged.
    """
    structured: list[dict] = []
    accumulated: list[str]  = []

    def _flush() -> None:
        if accumulated:
            structured.append({"type": "text", "content": " ".join(accumulated)})
            accumulated.clear()

    for block in blocks:
        btype = block["type"]
        if btype in ("title", "Title", "Introductory-Part"):
            _flush()
            structured.append(block)
        elif btype == "page_footnote":
            _flush()
            structured.append(block)
        else:  # text / equation
            accumulated.append(block["content"])

    _flush()
    return structured


def clean_cn_json(data: list[dict]) -> list[dict]:
    """
    Full cleaning pipeline for a single file's block list.
    Returns the restructured block list (may be empty if no valid content exists).
    """
    blocks = _extract_valid_blocks(data)
    if not blocks:
        return []
    blocks = _merge_leading_titles(blocks)
    blocks = _build_introductory_section(blocks)
    blocks = _merge_body_text(blocks)
    return blocks


# ============================================================
# Per-file Processing
# ============================================================

def _empty_result() -> dict:
    """Return a result dict with all fields initialised to default values."""
    return {
        "status":               "failed",
        "has_chinese":          False,
        "chinese_char_count":   0,
        "is_anomalous":         False,
        "count_Title":          0,
        "count_Introductory-Part": 0,
        "count_title":          0,
        "count_text":           0,
        "count_page_footnote":  0,
        "total_blocks":         0,
        "error":                "",
        "source_path":          "",
        "output_path":          "",
        "filename":             "",
    }


def process_single_file(json_path: str, output_path: str) -> dict:
    """
    Load, clean, and save one JSON file.
    Returns a result dict summarising the outcome.
    """
    result = _empty_result()

    try:
        data = load_json_file(json_path)

        if not data:
            result["error"]        = "Empty JSON file."
            result["is_anomalous"] = True
            return result

        has_chinese, char_count = count_chinese_chars(data)
        result["has_chinese"]        = has_chinese
        result["chinese_char_count"] = char_count

        if not has_chinese:
            result["is_anomalous"] = True
            result["error"]        = "No Chinese characters found in file."
            # Continue processing but flag the file as anomalous

        cleaned = clean_cn_json(data)

        if not cleaned:
            result["error"] = "No valid content after cleaning."
            if not has_chinese:
                result["is_anomalous"] = True
            return result

        # Tally block types
        type_key_map = {
            "Title":              "count_Title",
            "Introductory-Part":  "count_Introductory-Part",
            "title":              "count_title",
            "text":               "count_text",
            "page_footnote":      "count_page_footnote",
        }
        for block in cleaned:
            key = type_key_map.get(block.get("type", ""))
            if key:
                result[key] += 1
        result["total_blocks"] = len(cleaned)

        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(cleaned, fh, ensure_ascii=False, indent=2)

        result["status"] = "success"

    except Exception as exc:
        result["error"]  = str(exc)
        result["status"] = "failed"

    return result


# ============================================================
# Report I/O
# ============================================================

def save_report(records: list[dict], csv_path: str) -> None:
    """Write the full result table to a UTF-8-BOM CSV file."""
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")


def save_anomaly_report(records: list[dict], csv_path: str) -> str | None:
    """
    Write a filtered CSV containing only anomalous records.
    Returns the output path if any anomalies exist, otherwise None.
    """
    anomalies = [r for r in records if r.get("is_anomalous")]
    if not anomalies:
        return None
    anomaly_path = csv_path.replace(".csv", "_anomalous.csv")
    pd.DataFrame(anomalies).to_csv(anomaly_path, index=False, encoding="utf-8-sig")
    return anomaly_path


# ============================================================
# Summary Printing
# ============================================================

def print_summary(records: list[dict], output_dir: str, csv_path: str) -> None:
    """Print a formatted processing summary to stdout."""
    df      = pd.DataFrame(records)
    total   = len(df)
    success = (df["status"] == "success").sum()
    failed  = total - success
    pct     = lambda n: f"{n / total * 100:.1f}%" if total else "0%"

    anomalous  = (df["is_anomalous"]).sum()
    no_chinese = (~df["has_chinese"]).sum()

    print("\n" + "=" * 80)
    print("Processing complete — Summary")
    print("=" * 80)
    print(f"  Total files      : {total}")
    print(f"  Succeeded        : {success}  ({pct(success)})")
    print(f"  Failed           : {failed}  ({pct(failed)})")
    print(f"\n  No Chinese chars : {no_chinese}  ({pct(no_chinese)})")
    print(f"  Total anomalous  : {anomalous}  ({pct(anomalous)})")

    ok = df[df["status"] == "success"]
    if not ok.empty:
        print(f"\n  Average block counts (successful files):")
        for col, label in [
            ("count_Title",             "Title"),
            ("count_Introductory-Part", "Introductory-Part"),
            ("count_title",             "title"),
            ("count_text",              "text"),
            ("count_page_footnote",     "page_footnote"),
            ("total_blocks",            "total blocks"),
            ("chinese_char_count",      "Chinese chars"),
        ]:
            print(f"    {label:<22}: {ok[col].mean():.2f}")

    print(f"\n  Output JSON dir  : {output_dir}")
    print(f"  Result CSV       : {csv_path}")
    anomaly_path = csv_path.replace(".csv", "_anomalous.csv")
    if os.path.exists(anomaly_path):
        print(f"  Anomaly CSV      : {anomaly_path}")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process Chinese MinerU JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s -i cn_output_1204 -o cn_processed_1204\n"
            "  %(prog)s -i ./input -o ./output -c ./reports/results.csv\n"
        ),
    )
    parser.add_argument("-i", "--input_dir",  required=True, help="Input JSON directory.")
    parser.add_argument("-o", "--output_dir", required=True, help="Output JSON directory.")
    parser.add_argument(
        "-c", "--output_csv", default=None,
        help=f"Path for the result CSV (default: <output_dir>/{DEFAULT_CSV_NAME}).",
    )
    return parser.parse_args()


def main() -> None:
    args       = parse_args()
    input_dir  = args.input_dir
    output_dir = args.output_dir
    csv_path   = args.output_csv or os.path.join(output_dir, DEFAULT_CSV_NAME)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Chinese JSON Batch Processor")
    print("=" * 80)
    print(f"  Input dir    : {input_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Result CSV   : {csv_path}")
    print(f"  Kept types   : title, text, equation, page_footnote")
    print("  Rules applied:")
    print("    · First title  → 'Title'")
    print("    · Intro section → 'Introductory-Part'")
    print("    · page_footnote blocks kept independent")
    print("=" * 80)

    # ── Discover all JSON files ──────────────────────────────
    json_files = [
        (root, fname)
        for root, _dirs, files in os.walk(input_dir)
        for fname in files
        if fname.endswith(".json")
    ]

    print(f"\nFound {len(json_files)} JSON file(s).")
    if not json_files:
        print("ERROR: no JSON files found — nothing to do.")
        return

    # ── Process ──────────────────────────────────────────────
    records:       list[dict] = []
    success_count = failed_count = 0

    for root, filename in tqdm(json_files, desc="Processing files", unit="file"):
        source_path = os.path.join(root, filename)

        relative     = os.path.relpath(root, input_dir)
        output_stem  = filename.replace(".json", f"{OUTPUT_SUFFIX}.json")
        output_path  = os.path.join(output_dir, relative, output_stem)

        result = process_single_file(source_path, output_path)
        result["source_path"] = source_path
        result["output_path"] = output_path
        result["filename"]    = filename
        records.append(result)

        if result["status"] == "success":
            success_count += 1
        else:
            failed_count += 1

    # ── Save reports ─────────────────────────────────────────
    print("\nSaving result reports...")
    save_report(records, csv_path)
    anomaly_path = save_anomaly_report(records, csv_path)
    if anomaly_path:
        print(f"  Anomaly report saved to: {anomaly_path}")

    print_summary(records, output_dir, csv_path)


if __name__ == "__main__":
    main()
