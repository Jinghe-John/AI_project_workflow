# -*- coding: utf-8 -*-
"""
Author Extraction Pipeline
Reads paper metadata from an Excel file, loads footnote content from JSON files,
and extracts structured author information via a multi-threaded LLM API workflow.

Usage:
    Set API_KEYS, INPUT_FILE, and OUTPUT_FILE before running, then:
        python extract_authors.py
"""

import json
import itertools
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
API_KEYS = [
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # Add more keys for higher concurrency
]

INPUT_FILE  = "Arbitrage_results.xlsx"
OUTPUT_FILE = "Arbitrage_results_author.xlsx"

BASE_URL    = "https://api.moonshot.cn/v1"
MODEL       = "kimi-k2-0711-preview"
TEMPERATURE = 0.1
MAX_TEXT_LEN = 4000   # Truncate footnote text beyond this length before sending to API


# ============================================================
# Thread-safe API Key Rotation
# ============================================================
_key_lock  = threading.Lock()
_key_cycle = itertools.cycle(API_KEYS)


def get_client() -> OpenAI:
    """Return a new OpenAI client using the next API key in round-robin order."""
    with _key_lock:
        api_key = next(_key_cycle)
    return OpenAI(api_key=api_key, base_url=BASE_URL)


# ============================================================
# Thread-safe Per-path Deduplication Cache
#
# Problem: a plain dict cache has a race condition — concurrent threads
#          that all miss the cache will each fire a separate API call.
# Solution: each unique path gets its own Lock; the first thread to arrive
#           calls the API and writes the result; later threads wait on the
#           same Lock, then read the already-populated cache entry.
# ============================================================
_cache:           dict[str, list] = {}           # { path: [author_list] }
_path_locks:      dict[str, threading.Lock] = {} # { path: Lock }
_cache_meta_lock = threading.Lock()              # Guards _cache and _path_locks themselves


def get_authors_for_path(cn_path: str) -> list:
    """
    Guarantee that the API is called at most once per unique cn_path,
    regardless of how many threads request the same path concurrently.

    Flow:
        First thread  → acquires path lock → calls API → stores result → releases lock
        Later threads → acquire path lock → find result in cache → return it → release lock
    """
    # Fast path: check cache under the meta-lock before creating a path lock
    with _cache_meta_lock:
        if cn_path in _cache:
            return _cache[cn_path]
        if cn_path not in _path_locks:
            _path_locks[cn_path] = threading.Lock()
        path_lock = _path_locks[cn_path]

    # Serialize threads that share the same path
    with path_lock:
        # Double-check: a previous thread may have already populated the cache
        if cn_path in _cache:
            return _cache[cn_path]

        # Cache miss — fetch from the API
        footnote_text = extract_footnote_content(cn_path)
        if not footnote_text:
            _cache[cn_path] = []
            return []

        client      = get_client()
        raw_json    = extract_authors_by_llm(client, footnote_text)
        author_list = parse_llm_json(raw_json)
        _cache[cn_path] = author_list
        return author_list


# ============================================================
# Core Feature Functions
# ============================================================

def extract_footnote_content(file_path: str) -> str | None:
    """
    Open a JSON file and concatenate all 'page_footnote' items into a single string.
    Returns None if the file does not exist, is not a valid string path, or contains
    no footnote entries.
    """
    if not isinstance(file_path, str) or not os.path.exists(file_path):
        return None

    footnotes: list[str] = []
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Normalise the JSON structure to a flat list of items
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if isinstance(data.get("content"), list):
                items = data["content"]
            elif isinstance(data.get("blocks"), list):
                items = data["blocks"]
            else:
                items = [data]
        else:
            items = []

        for item in items:
            if isinstance(item, dict) and item.get("type") == "page_footnote":
                content = str(item.get("content", "")).strip()
                if content:
                    footnotes.append(content)

    except Exception:
        return None

    return "\n".join(footnotes) if footnotes else None


def extract_authors_by_llm(client: OpenAI, text: str) -> str | None:
    """
    Send footnote text to the LLM and return the raw JSON string response.
    Truncates input text to MAX_TEXT_LEN characters before sending.
    Returns None on API error.
    """
    if not text:
        return None

    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是专业的学术信息提取助手。请从脚注中提取所有作者的详细信息。\n"
                        "请返回一个标准的 **JSON 列表**，列表中的每个对象代表一位作者，包含以下字段：\n"
                        "- Name (姓名)\n"
                        "- Institution (机构，尽量全称)\n"
                        "- Code (邮编)\n"
                        "- Email (邮箱)\n\n"
                        "如果某个字段没有信息，请填 null 或 空字符串。\n"
                        "严格输出 JSON 格式，不要 Markdown 标记。"
                        "示例：[{\"Name\": \"张三\", \"Institution\": \"北京大学\", \"Code\": \"100871\", \"Email\": \"a@b.com\"}, {\"Name\": \"李四\", ...}]"
                    ),
                },
                {
                    "role": "user",
                    "content": f"脚注文本：\n{text}",
                },
            ],
            temperature=TEMPERATURE,
        )
        return completion.choices[0].message.content.strip()

    except Exception as exc:
        tqdm.write(f"[API Error] {exc}")
        return None


def parse_llm_json(json_str: str | None) -> list:
    """
    Parse the LLM's raw JSON string into a Python list.
    Strips Markdown code-fence markers if present.
    Returns an empty list on any parse failure.
    """
    if not json_str:
        return []
    cleaned = json_str.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return []


# ============================================================
# Per-row Worker (called from thread pool)
# ============================================================

def process_row(index: int, row: pd.Series) -> tuple[int, list | None]:
    """
    Extract authors for a single DataFrame row.

    Returns (index, author_list) where author_list is:
        None   — cn_json_path was empty/NaN; row should be skipped entirely
        []     — path was valid but no authors could be extracted
        [...]  — successfully extracted author dicts
    """
    cn_path = row.get("cn_json_path")

    if pd.isna(cn_path) or str(cn_path).strip() == "":
        return index, None

    return index, get_authors_for_path(str(cn_path).strip())


# ============================================================
# I/O Helpers
# ============================================================

def load_input(file_path: str) -> pd.DataFrame | None:
    """Load the input Excel file and validate required columns."""
    print(f"Loading input file: {file_path}")
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"ERROR: input file not found — {file_path}")
        return None

    if "cn_json_path" not in df.columns:
        print("ERROR: required column 'cn_json_path' is missing from the spreadsheet.")
        return None

    return df


def write_authors_to_df(df: pd.DataFrame, results: dict[int, list | None]) -> int:
    """
    Write extracted author data back into the DataFrame in-place.
    Returns the number of rows where at least one author was found.
    """
    success_count = 0
    for index, author_list in results.items():
        if author_list is None:
            continue                          # Empty path — leave row unchanged
        if not author_list:
            df.at[index, "Author_1_Name"] = "Not Found"
            continue

        success_count += 1
        for i, author in enumerate(author_list, start=1):
            df.at[index, f"Author_{i}_Name"]        = str(author.get("Name")        or "").strip()
            df.at[index, f"Author_{i}_Institution"] = str(author.get("Institution") or "").strip()
            df.at[index, f"Author_{i}_Code"]        = str(author.get("Code")        or "").strip()
            df.at[index, f"Author_{i}_Email"]       = str(author.get("Email")       or "").strip()

    return success_count


def save_output(df: pd.DataFrame, file_path: str) -> None:
    """Save the DataFrame to an Excel file."""
    df.to_excel(file_path, index=False)
    print(f"Output saved to: {file_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    # ── Load ────────────────────────────────────────────────
    df = load_input(INPUT_FILE)
    if df is None:
        return

    total        = len(df)
    unique_paths = df["cn_json_path"].dropna().nunique()
    concurrency  = len(API_KEYS)

    print(f"  Total rows          : {total}")
    print(f"  Unique JSON paths   : {unique_paths}")
    print(f"  Concurrent threads  : {concurrency}  (one per API key)")
    print(f"  => At most {unique_paths} API call(s) — duplicate paths reuse cached results.")
    print("=" * 60)

    # ── Process (thread pool) ────────────────────────────────
    results: dict[int, list | None] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(process_row, idx, row): idx
            for idx, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=total, desc="Extracting authors"):
            idx, data = future.result()
            results[idx] = data

    # ── Write results (main thread only) ────────────────────
    print("Writing results to DataFrame...")
    success_count = write_authors_to_df(df, results)

    # ── Save ────────────────────────────────────────────────
    save_output(df, OUTPUT_FILE)
    print("=" * 60)
    print(
        f"Done. "
        f"Successfully parsed: {success_count} row(s) | "
        f"Actual API calls made: {len(_cache)} | "
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
