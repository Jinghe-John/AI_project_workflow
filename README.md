# AI Project Workflow · End-to-End Technical Documentation

---
**Read this in other languages: [English](README.md), [中文](README_cn.md).**

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow Overview](#data-flow-overview)
3. [Step 01 · Multi-GPU Parallel PDF Parsing `multi_gpu_process.py`](#step-01--multi-gpu-parallel-pdf-parsing)
4. [Step 02 · General JSON Field Cleaning `clean_general.py`](#step-02--general-json-field-cleaning)
5. [Step 03 · Chinese Paper JSON Structuring `process_cn.py`](#step-03--chinese-paper-json-structuring)
6. [Step 04 · English Paper JSON Structuring `process_en.py`](#step-04--english-paper-json-structuring)
7. [Step 05/06 · Hierarchical Paper Classification `cn_classify.py` / `en_classify.py`](#step-0506--hierarchical-paper-classification)
8. [Step 07 · Reduced-Form Arbitrage Detection `reduced_form_arbitrage.py`](#step-07--reduced-form-arbitrage-detection)
9. [Step 08 · Structural Arbitrage Detection `structural_arbitrage.py`](#step-08--structural-arbitrage-detection)
10. [Step 09 · Arbitrage Paper Nationality Identification `paper_nationality.py`](#step-09--arbitrage-paper-nationality-identification)
11. [Step 10 · Arbitrage Paper Author Extraction `cn_paper_author.py`](#step-10--arbitrage-paper-author-extraction)
12. [Appendix: Environment Setup & Dependencies](#appendix-environment-setup--dependencies)
13. [Appendix: Frequently Asked Questions](#appendix-frequently-asked-questions)
14. [Appendix: Data Table Reference](#appendix-data-table-reference)

---

## System Overview

This system is an end-to-end automated academic arbitrage detection pipeline designed for **structural economics papers**. Starting from raw PDFs, it performs multi-GPU parallel parsing, JSON cleaning and structuring, LLM-based automatic classification, and ultimately identifies arbitrage behaviour in Chinese–English paper pairs, extracting nationality and author information for flagged papers.

The complete pipeline consists of **10 Python scripts** executed sequentially, covering the full lifecycle from data pre-processing to result post-processing.

| Step | Script | Core Function | Output |
|:---:|---|---|---|
| 01 | `multi_gpu_process.py` | Multi-GPU parallel PDF parsing | Raw JSON files |
| 02 | `clean_general.py` | General JSON field cleaning | Stripped JSON files |
| 03 | `process_cn.py` | Chinese paper JSON structuring | Structured Chinese JSON |
| 04 | `process_en.py` | English paper JSON structuring | Structured English JSON |
| 05 | `cn_classify.py` | Hierarchical Chinese paper classification | Classification label CSV |
| 06 | `en_classify.py` | Hierarchical English paper classification | Classification label CSV |
| 07 | `reduced_form_arbitrage.py` | Reduced-form paper arbitrage detection | Arbitrage result CSV |
| 08 | `structural_arbitrage.py` | Structural paper arbitrage detection | Arbitrage result CSV |
| 09 | `paper_nationality.py` | Arbitrage paper nationality identification | Nationality-annotated Excel |
| 10 | `cn_paper_author.py` | Arbitrage paper author extraction | Author information Excel |

---

## Data Flow Overview

```
Raw PDF Files
  │
  ▼  [Step 01] multi_gpu_process.py
  │  Multi-GPU parallel (4× A800), folder-level scheduling, calls mineru CLI
  │
Raw JSON output (*_model.json, containing bbox / score and many intermediate fields)
  │
  ▼  [Step 02] clean_general.py
  │  Recursively extracts type + content, strips all noise fields
  │
Stripped JSON (each node retains only type / content)
  │
  ├──▶  [Step 03] process_cn.py  →  Structured Chinese JSON
  │     Re-organises by section; produces Title + Introductory-Part + page_footnote
  │
  └──▶  [Step 04] process_en.py  →  Structured English JSON
        Uses CSV metadata + three-tier title matching to locate body start;
        produces introductory-part
  │
  ▼  [Step 05] cn_classify.py (Chinese)  +  [Step 06] en_classify.py (English)
  │  Calls Kimi API; six-layer decision-tree classification
  │
Classification label CSV (has_data / has_model / paper_type / measurement …)
  │
  ├──▶  [Step 07] reduced_form_arbitrage.py → Pure Empirical paper pairs only
  │
  └──▶  [Step 08] structural_arbitrage.py   → Structural + Non-Measurement pairs only
  │
Arbitrage detection result CSV (arbitrage / research_question / research_design …)
  │
  ├──▶  [Step 09] paper_nationality.py  →  Extract data-origin country for each paper
  │
  └──▶  [Step 10] cn_paper_author.py    →  Extract author name / institution / postcode / email
  │
Final Excel (arbitrage=TRUE rows + nationality + author information)
```

---

## Step 01 · Multi-GPU Parallel PDF Parsing

**Script:** `multi_gpu_process.py`

Batch-converts a raw PDF directory tree into MinerU-parsed JSON files, fully utilising multiple A800 GPUs to accelerate large-scale processing. This script is the **data entry point** for the entire pipeline.

### Design Rationale

The **folder** is used as the minimum scheduling unit. Folder tasks are dynamically distributed to worker processes, each bound to one GPU. All PDFs within the same folder are processed sequentially by the same GPU, ensuring load balance while avoiding GPU resource contention.

### Key Configuration Parameters

Edit at the top of `main()`:

```python
input_root = '/data/.../en_unzip'    # PDF root directory; script recurses all subdirectories
output_dir = 'en_output'             # MinerU output directory
csv_file   = 'pdf_location_info.csv' # Processing result log file
num_gpus   = 4                       # Number of GPUs (default: 4× A800)
```

The per-folder processing timeout defaults to **3600 seconds (1 hour)**. Adjust the `timeout` parameter in `process_folder()` as needed.

### Running

```bash
python multi_gpu_process.py
```

### Processing Architecture

```
Main process
  ├── Scans input directory → builds folder task queue
  ├── Launches N child processes (each bound to one GPU via CUDA_VISIBLE_DEVICES)
  │     └── Child loops: dequeue task → call mineru CLI → report result
  └── Main reads result queue → writes CSV in real time → prints progress every 10 folders
```

Workers are staggered by **2 seconds** at startup to avoid GPU initialisation conflicts. `Ctrl+C` safely interrupts the run; all results processed so far are retained in the CSV.

> ⚠️ PDFs that fail to parse must be manually converted to PNG before re-processing.

### Output: `pdf_location_info.csv`

| Field | Description |
|---|---|
| `filename` | PDF filename (without extension) |
| `full_path` | Full path to the PDF file |
| `is_corrupted` | `FALSE` = success / `TRUE` = failure |
| `error_message` | Error message on failure (up to 500 characters) |
| `process_time` | Average processing time for the file (seconds) |

---

## Step 02 · General JSON Field Cleaning

**Script:** `clean_general.py`

Iterates over all `*_model.json` files produced by MinerU, **recursively extracts `type` and `content` fields**, strips all irrelevant fields (bbox coordinates, confidence scores, font information, etc.), generates stripped JSON files, and outputs an Excel processing report.

### Cleaning Rules

The core function `clean_json_content()` performs a deep recursive traversal:

- **Dict elements inside a list**: retain only `type` and `content`; skip the element entirely if neither exists (no empty dict is emitted)
- **Nested lists inside a list**: recurse into them
- **Key-value pairs of a dict**: recurse if the value is a list or dict; otherwise retain directly
- Field filtering **only applies to dict elements inside lists**; the keys of dicts themselves are never filtered

### Before / After Comparison

```json
// Before (raw)
{ "type": "text", "content": "body text here", "bbox": [100,200,300,400], "score": 0.98, "page_id": 1 }

// After (output)
{ "type": "text", "content": "body text here" }
```

### Running

```bash
# Basic usage (preserves original directory structure)
python clean_general.py -i ./en_output -o ./cleaned_output

# Flat output — all files placed in one directory
python clean_general.py -i ./input -o ./output --flat

# Custom target file suffix
python clean_general.py -i ./input -o ./output --suffix "_result.json"
```

### Parameter Reference

| Parameter | Short | Required | Default | Description |
|---|---|:---:|---|---|
| `--input` | `-i` | ✅ | — | Input directory path |
| `--output` | `-o` | ✅ | — | Output directory path |
| `--suffix` | `-s` | ❌ | `_model.json` | Target file suffix to match |
| `--flat` | — | ❌ | No | Flat output; do not preserve directory structure |
| `--report` | `-r` | ❌ | Timestamped | Custom report filename |

### Output Notes

- The cleaned filename replaces the matched suffix with `.json`, e.g. `2014_09_12_9_model.json` → `2014_09_12_9.json`
- The Excel report contains two sheets: **Detailed Records** (processing status, path, and error message per file) and **Summary Statistics** (total / succeeded / failed / success rate)

---

## Step 03 · Chinese Paper JSON Structuring

**Script:** `process_cn.py`

Semantically re-organises cleaned Chinese JSON files: merges content by section, detects and flags files that contain no Chinese characters, and produces structurally uniform academic JSON.

### Design Rationale

Converts the raw flat list into a **section-oriented, semantically clear structured JSON**, filtering out noise such as page headers and metadata, while detecting whether each file actually contains Chinese characters and automatically flagging files that may have been misrouted into the Chinese content pool.

### Five-Step Processing Pipeline

**① Flatten Nested Structure**

MinerU output is sometimes a "list of lists" (one array per page). On load, this is automatically flattened into a single-level list, providing a uniform input for all subsequent steps.

**② Type Filtering**

Only the following four types are retained; everything else is discarded:

| Retained Type | Description |
|---|---|
| `title` | Section heading (`*` and `#` characters are stripped) |
| `text` | Body paragraph |
| `equation` | Mathematical formula |
| `page_footnote` | Page footnote |

All content **before the first `title`** is also skipped, avoiding retention of page headers, author information bars, and similar noise.

**③ Merge Adjacent Titles**

If the document begins with two consecutive `title` blocks (no body text between them), they are automatically merged into one, preventing the paper title from being incorrectly split:

```
title: "Deep-Learning-Based"  +  title: "Image Recognition Research"
  →  title: "Deep-Learning-Based Image Recognition Research"
```

**④ Generate Title and Introductory-Part**

The first `title` is promoted to the special type `Title`. All content between `Title` and the third `title` (abstract, keywords, author information, etc.) is merged into a single `Introductory-Part` block. `page_footnote` blocks are retained independently, placed after `Introductory-Part`.

If no third `title` exists, all remaining content is absorbed into `Introductory-Part`.

**⑤ Merge Body by Section**

The remaining content is traversed and adjacent `text` and `equation` blocks are merged into a single `text` block until the next `title` is encountered. `page_footnote` blocks always remain independent and are never merged.

### Output Structure Example

```json
[
  { "type": "Title",             "content": "The Effect of X on Y: Evidence from Z" },
  { "type": "Introductory-Part", "content": "Abstract: This paper… Keywords: … 1. Introduction…" },
  { "type": "title",             "content": "2 Data" },
  { "type": "text",              "content": "The data used in this paper come from the XXX database…" },
  { "type": "page_footnote",     "content": "Author information: …" }
]
```

### Running

```bash
python process_cn.py -i ./cn_output -o ./cn_processed
python process_cn.py -i ./cn_output -o ./cn_processed -c ./report.csv
```

### Key CSV Report Fields

| Field | Description |
|---|---|
| `contains_chinese` | Yes / No |
| `chinese_char_count` | Total number of Chinese characters in the file |
| `is_abnormal` | Files with no Chinese characters are flagged as abnormal and also written to `*_abnormal.csv` |
| `title_count / intro_part_count / section_title_count / text_count` | Block count per type |
| `status` | Success / Failed (with error message on failure) |

---

## Step 04 · English Paper JSON Structuring

**Script:** `process_en.py`

Combines CSV metadata with JSON content and uses a **three-tier title matching strategy** to locate the paper's body start, trims irrelevant content, and produces structurally uniform English academic JSON.

### Design Rationale

English processing is more complex than Chinese: a single PDF may contain concatenated content from multiple papers. The reference title (`Ref_Title`) from the CSV is used to precisely locate the start of the current paper before trimming everything else. The three-tier matching strategy handles wide variation in title formatting.

### Overall Pipeline

```
Read CSV → filter pre-1980 entries
  ↓
Locate JSON file (use first 4 characters of pdf_id as year subdirectory)
  ↓
Clean JSON (type filtering + merge adjacent text/equation blocks by section)
  ↓
Title matching (Ref_Title vs all title blocks inside JSON)
  ├── Match found → trim content + generate introductory-part
  └── No match    → retain full content, skip conversion
  ↓
Save structured JSON + write CSV report
```

### Three-Tier Title Matching Strategy

`Ref_Title` from the CSV is matched against all `title` blocks in the JSON using the tiers below in order; if an earlier tier succeeds, later tiers are not attempted. Text normalisation steps: replace Unicode quotes / superscripts / subscripts → lowercase → NFKC normalisation → strip punctuation → strip whitespace.

| Tier | Method | Rule |
|:---:|---|---|
| ① | Exact match `exact` | Compare normalised strings for full equality |
| ② | Fuzzy match `fuzzy` | `fuzz.ratio` edit-distance similarity ≥ 80 |
| ③ | Subset match `subset` | Containment / length-similar with similarity ≥ 90 / chunk comparison (≥ 80% of chunks match) |

### Match Result Types and Handling

| Result Type | Handling |
|---|---|
| `one_to_one` (unique match) | Use the matched title directly as the content start |
| `one_to_many` (multiple matches) | Use the first match as the start; delete all other duplicate-matched titles |
| `no_match` (no match found) | Retain full content; skip introductory-part conversion |

### Introductory-Part Generation Logic

After trimming, the script searches for a title containing `introduction` or starting with `1` / `1.` / `I` / `I.` as the introduction entry point, and merges `Title → introduction title → first introduction paragraph` into an `introductory-part` block. If none is found, a fallback strategy is applied (target title is selected based on the total number of titles in the document).

### Running

```bash
python process_en.py -i all_pdf_info.csv -j ./en_output -o ./en_processed_json
python process_en.py -i all_pdf_info.csv -j ./en_output -o ./en_processed_json -c ./report.csv
```

### Required Input CSV Columns

| Column | Description |
|---|---|
| `pdf_id` | Document ID; first 4 characters are the year, used to locate the JSON subdirectory |
| `Year` | Publication year; entries before 1980 are automatically filtered out |
| `Ref_Title` | Reference title used for matching against internal JSON title blocks |

---

## Step 05/06 · Hierarchical Paper Classification

**Scripts:** `cn_classify.py` (Chinese) / `en_classify.py` (English parallel)

Calls the Kimi API to automatically classify papers using a **six-layer decision tree**. The classification input is the `introductory-part` field in the JSON file (title + abstract + introduction).

### Differences Between the Two Scripts

| Dimension | `cn_classify.py` (Chinese) | `en_classify.py` (English) |
|---|---|---|
| Target | All Chinese papers | Subset of Chinese papers with English references |
| JSON path column | `cn_json_path` | `en_json_path` |
| Title column | `json_title` | `Ref_Title` |
| Result column prefix | `has_data` / `paper_type` etc. | `en_has_data` / `en_paper_type` etc. |
| Prompt language | Chinese | English |

### Six-Layer Classification Decision Tree

```
Layer 1: has_data & has_model (uses real-world data? / builds a formal model?)
  │
  ├─ Data=Yes & Model=Yes
  │     ↓
  │   Layer 2: is_structural (contains counterfactual analysis or calibration?)
  │     ├─ Yes → paper_type = Structural → Layer 6: measurement judgement
  │     └─ No  → Layer 3: Non-Structural sub-classification
  │                 ↓
  │             Pure Empirical / Pure Theory / Other
  │             (Pure Empirical & Other proceed to Layers 4 and 5)
  │
  ├─ Data=Yes & Model=No → paper_type = Pure Empirical → Layers 4 & 5
  ├─ Data=No  & Model=Yes → paper_type = Pure Theory (end)
  └─ Data=No  & Model=No  → paper_type = Other → Layers 4 & 5

Layer 4: research_article (contains empirical estimation / regression / testing?)
  ↓ (only Research Articles proceed to Layer 5)
Layer 5: variable (title structure: closed analytical framework vs. open exploratory terms)
  Single Explanatory / Multiple Explanators

Layer 6: measurement (Structural papers only)
  Is the research endpoint an isolated numerical indicator?
  Measurement Article / Non-Measurement Article
```

### Layer-by-Layer Criteria

**Layer 1: Data and Model Assessment**

| Field | Yes Definition | No Definition |
|---|---|---|
| `has_data` | Uses real-world data, including empirical analysis, numerical simulation, or CGE models | Does not use real-world data |
| `has_model` | Develops or substantially extends a formal mathematical / theoretical / computational model or statistical test | Does not introduce a new formal model or statistical test |

**Layer 2: Structural vs. Non-Structural** *(triggered only when `has_data=Yes` and `has_model=Yes`)*

| Outcome | Definition |
|---|---|
| `Structural` | The paper contains **counterfactual analysis** or **calibration** |
| `Non-Structural` | The paper contains neither counterfactual analysis nor calibration |

**Layer 5: Number of Explanatory Variables** *(assessed from the paper title structure)*

| Type | Definition |
|---|---|
| `Single Explanatory` | The title explicitly lists all core concepts, forming a closed analytical framework (e.g. "The Effect of X on Y") |
| `Multiple Explanators` | The title uses open-ended exploratory terms (e.g. "Determinants of…", "Factors Affecting…") |

### Resume-from-Checkpoint Mechanism

- **`cn_classify.py`**: saves `temp_classification_result.csv` every 10 records; automatically resumes from the checkpoint on next launch; generates a timestamped final file upon completion
- **`en_classify.py`**: saves `classification_checkpoint_v4.pkl` every 5 records and an intermediate CSV every 20 records; rows that already have valid records (not `ERROR` / `FILE_NOT_FOUND`) are automatically skipped

### Parallel Architecture in `en_classify.py`

10 API keys map to 10 workers; tasks are submitted concurrently via `ThreadPoolExecutor`, with each worker holding an exclusive client instance to avoid interference.

### Key Configuration

```python
# cn_classify.py
client_kimi = OpenAI(api_key="sk-xxx", base_url="https://api.moonshot.cn/v1")
CSV_PATH  = os.path.join(BASE_DIR, "all_Structural_unclassified.csv")

# en_classify.py
API_KEYS   = ["sk-xxx", "sk-yyy", ...]   # Number of keys determines concurrency
INPUT_FILE = "all_Structural.csv"
```

### New Output Columns

| Column | Values |
|---|---|
| `has_data` | `Yes` / `No` / `ERROR` |
| `has_model` | `Yes` / `No` / `ERROR` |
| `paper_type` | `Structural` / `Pure Empirical` / `Pure Theory` / `Other` / `ERROR` / `FILE_NOT_FOUND` / `NO_INTRO` |
| `research_article` | `Research Article` / `Non-Research Article` / empty (not applicable) |
| `variable` | `Single Explanatory` / `Multiple Explanators` / empty (not applicable) |
| `measurement` | `Measurement Article` / `Non-Measurement Article` / empty (not applicable) |

English-version column names are uniformly prefixed with `en_`, e.g. `en_has_data`, `en_paper_type`.

---

## Step 07 · Reduced-Form Arbitrage Detection

**Script:** `reduced_form_arbitrage.py`

Performs **multi-dimensional, two-round cross-validation** on reduced-form (Pure Empirical) Chinese–English paper pairs to identify academic arbitrage. This is the **core analysis module** of the entire pipeline and handles the largest paper category, so it is processed first.

### What Is Academic Arbitrage

> Applying the identification strategy, empirical design, and research question of an English paper to the Chinese context without substantive methodological innovation — that is, directly adopting the causal inference framework of the English paper and merely substituting Chinese data or institutional background for the original.

For reduced-form papers, the arbitrage judgement focuses on the **originality of the identification strategy**: whether the Chinese paper independently proposes a new exogenous variable, natural experiment, or causal identification scheme, or merely substitutes data into the existing English framework.

### Eligibility Criteria (all four columns must be satisfied)

| Column | Condition |
|---|---|
| `paper_type` | Must be `Pure Empirical` |
| `en_paper_type` | Must be `Pure Empirical` |
| `research_article` | Must be `Research Article` |
| `en_research_article` | Must be `Research Article` |

Rows that fail any condition are marked `skipped` with a recorded reason.

### Three-Dimensional Arbitrage Framework

**All three dimensions must be satisfied simultaneously for an arbitrage verdict. If any dimension is `False` or `ERROR`, no arbitrage is declared.**

| Dimension | Reduced-Form Specific Criteria |
|---|---|
| **Research Question (RQ)** | Both papers investigate the same causal question: after removing the "China" label, the core dependent and independent variables are economically equivalent, differing only in data source |
| **Research Design (RD)** | Uses the same identification strategy: same instrumental variable source, same natural experiment logic, or same DID/RDD treatment variable and control group construction — the Chinese paper proposes no independent exogenous identification scheme |
| **Mechanism （Mech）** | A Chinese paper arbitrages an English paper if: it explores the exactly same mechanism in a identical way with nearly same findings. |

### Canonical Arbitrage and Non-Arbitrage Cases

| Scenario | Verdict | Reason |
|---|:---:|---|
| English paper uses a migration wave as IV to estimate labour supply shocks; Chinese paper uses a similar migration variable to estimate China's urbanisation effect | ✅ Arbitrage | IV logic and identification source are substantively identical |
| English paper uses a geographic discontinuity to identify policy effects; Chinese paper uses a completely different policy time-point for DID | ❌ Not Arbitrage | Identification strategies are independently designed with different sources of exogeneity |
| The Chinese paper's core explanatory variable is a uniquely Chinese institutional variable with no English-paper counterpart | ❌ Not Arbitrage | The research question itself has no functionally equivalent mapping in the English paper |

### Two-Round Cross-Validation Mechanism

The API is called **twice** per paper pair (Round 1 + Round 2), and the **intersection** of the two rounds is taken: only dimensions judged `True` in both rounds are counted as `True` in the final verdict. This substantially improves reliability and reduces false positives caused by LLM randomness.

### English-to-English Arbitrage Detection

When a Chinese paper matches multiple English candidate sources, the system automatically runs pairwise arbitrage checks among those English papers and uses the `most_likely_source` field to identify the most likely primary source. The English-to-English arbitrage summary is saved as a separate JSON file:

```
{original_filename}_arbitrage_results_{timestamp}_en_en_arbitrage_summary.json
```

### Running and Configuration

```bash
python reduced_form_arbitrage.py
```

```python
BASE_URL   = "https://api.moonshot.cn/v1"
MODEL      = "kimi-k2-0711-preview"
API_KEYS   = ["sk-xxx", "sk-yyy", ...]   # Number of keys determines concurrency
INPUT_PATH = "path/to/your/input.csv"
```

### New Output Columns

| Column | Description |
|---|---|
| `research_question` | Research question dimension: `True` / `False` / `ERROR` |
| `research_design` | Research design dimension: `True` / `False` / `ERROR` |
| `counterfactual_analysis` | Counterfactual analysis dimension: `True` / `False` / `ERROR` |
| `arbitrage` | Overall arbitrage verdict (True only if all three dimensions are True) |
| `most_likely_source` | Whether this row is the most likely primary arbitrage source |
| `round1_rq` / `round2_rq` | Round 1 / Round 2 research question verdict |
| `round1_rd` / `round2_rd` | Round 1 / Round 2 research design verdict |
| `round1_mech` / `round2_mech` | Round 1 / Round 2 counterfactual analysis verdict |
| `consistency_rq/rd/mech` | Two-round consistency (`consistent` / `inconsistent` / `error`) |
| `en_en_arbitrage_detected` | Whether English-to-English arbitrage was detected |
| `en_en_arbitrage_pairs` | English paper pairs found to be in an arbitrage relationship |
| `skip_reason` | Reason for skipping (e.g. `en_paper_type is not Pure Empirical`) |

### Resume from Checkpoint

A `*_checkpoint.json` file is generated during the run. After an interruption, simply re-run the script; the system automatically detects the checkpoint and resumes, skipping already-processed rows.

---

## Step 08 · Structural Arbitrage Detection

**Script:** `structural_arbitrage.py`

Performs arbitrage detection on structural Chinese–English paper pairs. The logical framework is the same as Step 07, but the judgement criteria and prompt are redesigned for the methodological characteristics of structural model papers.

### Differences from the Reduced-Form Module

Arbitrage in structural papers is harder to detect: the same system of equations and moment-matching strategy can be transplanted directly to data from a different country, appearing formally independent while being methodologically equivalent. The judgement criterion therefore shifts from "is the identification strategy the same?" to "**are the model skeleton and counterfactual experiments functionally equivalent?**"

### Eligibility Criteria (all four columns must be satisfied)

| Column | Condition |
|---|---|
| `paper_type` | Must be `Structural` |
| `en_paper_type` | Must be `Structural` |
| `measurement` | Must be `Non-Measurement Article` |
| `en_measurement` | Must be `Non-Measurement Article` |

### Three-Dimensional Arbitrage Framework

| Dimension | Structural Paper Specific Criteria |
|---|---|
| **Research Question (RQ)** | Via the "mechanism mapping test": after removing Chinese institutional labels (Hukou, subsidies, SOEs), is the causal question identical to the English paper? |
| **Research Design (RD)** | Uses the same model skeleton (functional forms for preferences, technology, and constraints) and the same identification strategy (e.g. same SMM/GMM moment-matching targets) |
| **Counterfactual Analysis (CF)** | Policy simulation shocks target the same economic margin, with functionally equivalent transmission mechanisms and welfare implications — only the policy label is replaced with a Chinese counterpart |

### Running and Configuration

```bash
python structural_arbitrage.py
```

```python
BASE_URL   = "https://api.moonshot.cn/v1"
MODEL      = "kimi-k2-0711-preview"
API_KEYS   = ["sk-xxx", "sk-yyy", ...]
INPUT_PATH = "path/to/your/input.csv"
```

The output column structure is identical to Step 07 and the checkpoint mechanism works the same way; see Step 07 for details.

---

## Step 09 · Arbitrage Paper Nationality Identification

**Script:** `paper_nationality.py`

Automatically extracts the **data-origin country** of Chinese and English papers from arbitrage detection results and writes the information back to the Excel file.

### Design Rationale

This script is a **downstream post-processing module** for arbitrage detection, operating only on rows where `arbitrage == TRUE` to minimise API call overhead. It calls the Kimi API to analyse each paper's `introductory-part` and identify the data-origin country, supporting subsequent analysis of "which countries' English research Chinese papers have transposed to a Chinese context."

### Country Standardisation Rules

Built-in mapping rules ensure consistent output:

| Raw Identification | Standardised Output |
|---|---|
| `United States` / `U.S.` / `America` | `USA` |
| `United Kingdom` / `Great Britain` | `UK` |
| `P.R.C` / `Mainland China` | `China` |
| OECD / EU / Global / cross-country multi-country data | `Multiple Countries` |

### Running and Configuration

```bash
python paper_nationality.py
```

```python
MOONSHOT_API_KEY = "sk-xxx"
INPUT_FILE  = "Arbitrage_results.xlsx"
OUTPUT_FILE = "Arbitrage_results_nationality.xlsx"
BASE_URL    = "https://api.moonshot.cn/v1"
MODEL       = "kimi-k2-0711-preview"
```

Results are auto-saved every 10 rows to prevent data loss on unexpected interruption.

### New Output Columns

| Column | Description | Example Values |
|---|---|---|
| `cn_country` | Data-origin country of the Chinese paper | `China` |
| `en_country` | Data-origin country of the English paper | `USA` / `Multiple Countries` |

### Special Value Reference

| Value | Meaning |
|---|---|
| `Multiple Countries` | Paper uses multi-country data (including OECD, EU, global samples, etc.) |
| `Content Missing` | No `introductory-part` content found in the JSON file |
| `No Content` | JSON path is empty or the file does not exist |
| `API Error` | API call failed; requires manual review |

---

## Step 10 · Arbitrage Paper Author Extraction

**Script:** `cn_paper_author.py`

Extracts author names, institutions, postcodes, and email addresses from the footnotes (`page_footnote`) of Chinese paper JSON files, with support for **multi-key parallel processing** and a **per-path deduplication cache**.

### Design Rationale

This script processes **all rows** (not filtered by `arbitrage` value) for subsequent academic network analysis and institutional distribution statistics. The core design feature is the per-path deduplication cache, which substantially reduces API call costs.

### Core Design: Per-Path Deduplication Cache

A single Chinese paper may appear multiple times in the arbitrage result table (once per matched English paper). The script uses a **per-path lock mechanism** to ensure the API is called at most once per unique path; subsequent rows reuse the cached result:

```
Actual API calls = number of unique cn_json_paths (not total row count)

Example: 500 rows with 312 unique paths
  → At most 312 API calls, saving 37.6% of call costs
```

```
First thread accesses path A → acquires lock → calls API → stores in cache → releases lock
Subsequent threads access path A → acquire lock → read from cache → release lock (no API call)
```

### Running and Configuration

```bash
python cn_paper_author.py
```

```python
API_KEYS    = ["sk-xxx", "sk-yyy", ...]   # More keys → higher concurrency
INPUT_FILE  = "Arbitrage_results.xlsx"
OUTPUT_FILE = "Arbitrage_results_author.xlsx"
```

The console displays at runtime:

```
Total rows: 500 | Unique paths: 312 | Concurrent threads: 14
=> At most 312 API calls; duplicate paths reuse cached results automatically
```

### New Output Column Format

Four columns are generated per author and expand automatically with the number of authors in the paper:

| Column Example | Description |
|---|---|
| `Author_1_Name` | Name of the 1st author |
| `Author_1_Institution` | Institution of the 1st author |
| `Author_1_Code` | Postcode of the 1st author |
| `Author_1_Email` | Email address of the 1st author |
| `Author_N_*` | Same pattern for author N (N = total author count) |

`Not Found` indicates that no footnote content was found in the JSON file, or the LLM was unable to extract author information.

---

## Appendix: Environment Setup & Dependencies

### Core Python Dependencies

```bash
pip install pandas openai tqdm rapidfuzz openpyxl
```

| Package | Purpose | Scripts |
|---|---|---|
| `pandas` | CSV / Excel reading, writing, and data processing | All scripts |
| `openai` | Kimi API client | cn_classify / en_classify / arbitrage / nationality / author |
| `tqdm` | Progress bar display | en_classify / arbitrage / nationality / author |
| `rapidfuzz` | Fuzzy title matching | process_en |
| `openpyxl` | Excel file read/write backend | nationality / author |

### Hardware and System Requirements

| Component | Requirement |
|---|---|
| GPU | 4 × NVIDIA A800 (or adjust `num_gpus` for other configurations) |
| CUDA | Must be correctly configured (`multi_gpu_process.py`) |
| MinerU | The `mineru` CLI tool must be on the system PATH (`multi_gpu_process.py`) |
| Python | 3.8+ |

### API Service Configuration

| Parameter | Default |
|---|---|
| `BASE_URL` | `https://api.moonshot.cn/v1` |
| `MODEL` | `kimi-k2-0711-preview` |
| `API_KEYS` | Configured at the top of each script; 10 keys recommended to maximise concurrent quota |

---

## Appendix: Frequently Asked Questions

| Issue | Solution |
|---|---|
| PDF parsing failure (`is_corrupted=TRUE`) | Convert the failed PDF to PNG and re-parse (see notes in `multi_gpu_process.py`) |
| English JSON has no match (`no_match`) | Check the format of `Ref_Title` in the CSV; manually correct the title and re-run `process_en.py` |
| All classification results are `ERROR` | Verify that the API key is valid; check network connectivity; review console error details |
| Arbitrage detection interrupted and restarted | Simply re-run the script; the system automatically detects the checkpoint and resumes |
| `cn_country` shows `API Error` in the Excel output | Note the row number, verify the corresponding JSON path is correct, or re-run `paper_nationality.py` |
| Author information column shows `Not Found` | The paper's footnote format may be non-standard; inspect the `page_footnote` field in the raw JSON manually |
| Chinese classification result shows `NO_INTRO` | No `introductory-part` field was found in the corresponding JSON; check the structuring output from `process_cn.py` |

---

*Scripts covered: `multi_gpu_process.py` · `clean_general.py` · `process_cn.py` · `process_en.py` · `cn_classify.py` · `en_classify.py` · `reduced_form_arbitrage.py` · `structural_arbitrage.py` · `paper_nationality.py` · `cn_paper_author.py`*

---

## Appendix: Data Table Reference

### Input Data: `all_ref_final.csv`

This is the **initial reference table** for the entire pipeline, recording metadata for all English references to be analysed. Each row represents one English paper cited by one Chinese paper.

| Column | Description |
|---|---|
| `Ref_Title` | English reference title, used by `process_en.py` for title-matching and location |
| `Author` | Author(s) of the English paper |
| `Journal` | Publication journal |
| `Year` | Publication year; used to filter out pre-1980 entries |
| `source` | Original file path of the citing Chinese paper (includes journal, year, and paper title) |
| `title` | Title of the Chinese paper that cites this English reference |
| `id` | Unique identifier of the Chinese paper, formatted as `year_issue_sequence` (e.g. `2008_01_1`) |

A single Chinese paper typically corresponds to multiple rows (one per cited English paper); all rows sharing the same `id` belong to the same Chinese paper.

---

### Output Data: `Arbitrage_Result_Cleaned.xlsx`

This is the **final data table produced after the entire pipeline completes**. It extends `all_ref_final.csv` by appending classification results, arbitrage detection results, nationality information, and author information in sequence. Each row represents a complete analysis record for one Chinese–English paper pair.

**Overview of Key New Columns:**

| Column | Source Step | Description |
|---|---|---|
| `pdf_id` | Step 01 | English paper PDF identifier, used to locate the JSON file |
| `cn_json_path` / `en_json_path` | Step 03/04 | File paths of the structured Chinese and English JSON files |
| `has_data` / `has_model` | Step 05 | Whether the Chinese paper uses real-world data / builds a formal model |
| `paper_type` / `measurement` | Step 05 | Chinese paper type and whether it is a measurement paper |
| `en_has_data` / `en_paper_type` etc. | Step 06 | Corresponding classification fields for the English paper (prefixed with `en_`) |
| `research_question` / `research_design` / `mechanism` / `counterfactual_analysis` | Step 07/08 | Three-dimensional arbitrage verdicts |
| `arbitrage` | Step 07/08 | Overall arbitrage verdict (`TRUE` / `FALSE` / `ERROR` / `skipped`) |
| `most_likely_source` | Step 07/08 | When multiple potential arbitrage sources exist, pairwise comparison selects the most similar source |
| `round1_*` / `round2_*` / `consistency_*` | Step 07/08 | Two-round verification details and consistency flags |
| `cn_country` / `en_country` | Step 09 | Data-origin country of the Chinese and English papers |
| `en_Author` | all_ref_final.csv | Author(s) of the English paper |
| `Author_N_Name` / `Author_N_Institution` / `Author_N_Code` / `Author_N_Email` | Step 10 | Information for the Nth author of the Chinese paper |

**Two Manual Review Fields:**

| Column | Type | Description |
|---|---|---|
| `Author_Match` | Manual review | For paper pairs with `arbitrage=TRUE`, manually verify whether the Chinese and English paper authors overlap. `TRUE` = authors match (shared co-authors or clear association); `FALSE` = authors do not match. Used to determine whether the arbitrage behaviour constitutes author self-reuse. |
| `Edition` | Manual review | When a Chinese paper has multiple potential English arbitrage sources, manually verify whether those sources are different versions of the same paper. `Same` = same version (e.g. working paper vs. published version); `Different` = different versions. Used to distinguish "repeated matching against the same English study" from "simultaneously drawing on multiple distinct English papers." |
