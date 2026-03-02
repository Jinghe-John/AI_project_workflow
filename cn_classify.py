# -*- coding: utf-8 -*-
"""
Paper Classification Pipeline
Reads paper metadata from a CSV file, loads introductory content from JSON files,
and classifies each paper through a multi-step hierarchical API workflow.

Usage:
    Set BASE_DIR, CSV_PATH, and KIMI_API_KEY before running, then:
        python classify_papers.py
"""

import os
import json
import time
import traceback
from datetime import datetime

import pandas as pd
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================
BASE_DIR    = r"/path/to/your/data/directory"
CSV_PATH    = os.path.join(BASE_DIR, "all_Structural_unclassified.csv")
JSON_DIR    = BASE_DIR
KIMI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# ============================================================
# API Client
# ============================================================
client_kimi = OpenAI(
    api_key=KIMI_API_KEY,
    base_url="https://api.moonshot.cn/v1",
)

MODEL_NAME   = "kimi-k2-0711-preview"
TEMPERATURE  = 0
SEED         = 12345
MAX_TOKENS   = 200
RETRY_DELAY  = 2   # seconds between retries
MAX_RETRIES  = 3
API_DELAY    = 2   # seconds between successful API calls
SAVE_EVERY   = 10  # save intermediate results every N processed papers


# ============================================================
# Prompt Builders  (Chinese prompts intentionally preserved)
# ============================================================

def build_data_model_prompt(introductory_part: str) -> str:
    """Build the data-and-model classification prompt."""
    prompt = {
        "persona": "《经济研究》期刊的资深编辑助理",
        "task": "根据提供的论文标题、摘要和引言，回答关于论文方法论的两个独立问题。首先，判断论文是否使用了实证数据。其次，判断论文是否建立了正式模型。",
        "output_format": {
            "description": "基于文本内容，对两个问题分别给出'是'或'否'的分类结果。",
            "classification_definitions": {
                "has_data": {
                    "Yes": "论文使用了真实世界的数据,包含实证分析，数值模拟或CGE(computable general equilibrium)模型",
                    "No": "论文未使用真实世界的数据"
                },
                "has_model": {
                    "Yes": "论文开发或显著扩展了正式的数学模型、理论模型或计算模型/统计检验",
                    "No": "论文未引入新的正式模型或统计检验；可能使用了现有模型或专注于非基于模型的分析"
                }
            },
            "schema": {
                "has_data": "[Yes|No]",
                "has_model": "[Yes|No]"
            }
        },
        "paper_title & paper_abstract & introduction": introductory_part
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_structural_prompt(introductory_part: str) -> str:
    """Build the structural-empirical classification prompt."""
    prompt = {
        "persona": "《经济研究》期刊的资深编辑助理",
        "task": "根据提供的论文信息（标题、摘要和引言），精确判断该论文是否为结构实证论文。请使用下面的详细定义进行准确分类。",
        "output_format": {
            "description": "基于文本中对研究方法的描述（标题、摘要和引言），分类该论文是否为结构实证论文。",
            "classification_definitions": {
                "Structural": "结构实证论文必须包含反事实分析或校准",
                "Non-Structural": "论文不包含反事实分析和校准"
            },
            "schema": {
                "structural_classification": "[Yes|No]"
            }
        },
        "paper_title & paper_abstract & introduction": introductory_part
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_non_structural_prompt(introductory_part: str) -> str:
    """Build the non-structural / has-data-no-model sub-classification prompt."""
    prompt = {
        "persona": "《经济研究》期刊的资深编辑助理",
        "task": "对于非结构实证论文或包含数据但无模型的论文，确定论文的具体类型。请使用下面的详细定义进行准确分类。",
        "output_format": {
            "description": "根据论文中描述的主要贡献，将实证方法分类为以下两个类别之一。",
            "classification_definitions": {
                "Pure Empirical": "涵盖不涉及估计完全指定的结构经济模型的数据实证分析。这一类别的范围从纯粹的描述性工作到因果效应的估计。一端包括引入新数据集、记录程式化事实或报告趋势和相关性而不做因果声明的论文（'描述性'）。另一端包括旨在估计一个变量对另一个变量因果效应的'简化形式'分析，重点关注研究设计和计量经济学技术（如工具变量、双重差分、断点回归和实验）来克服内生性并识别因果联系。",
                "Pure Theory": "专注于经济分析的抽象模型和方法的发展，其中主要贡献是理论框架本身，而不是来自真实世界数据的实证发现。这一类别包括两种主要的工作类型：1）经济理论，从公理假设发展数学模型来表示经济现象，并得出关于理性代理人行为的逻辑命题；2）计量经济理论，发展和分析经济数据的统计方法，提出新的估计量或检验程序并证明其统计性质。虽然可能使用蒙特卡罗模拟来评估这些方法的性能，但这些论文不使用真实世界的数据来检验模型或产生实质性的经济结论。",
                "Other": "不符合上述类别的任何论文。"
            },
            "schema": {
                "classification": "[Pure Empirical|Pure Theory|Other]"
            }
        },
        "paper_title & paper_abstract & introduction": introductory_part
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_explanatory_variables_prompt(title: str) -> str:
    """Build the explanatory-variables classification prompt (title-only)."""
    prompt = {
        "persona": "《经济研究》期刊的资深编辑助理",
        "task": "根据论文标题的结构，确定论文的类型。请使用下面的严格定义进行准确分类。",
        "output_format": {
            "description": "根据论文标题是提出一个封闭的分析框架还是一个开放的因素探索，将论文分类为以下两个类别之一。",
            "classification_definitions": {
                "Single Explanatory": "论文的标题明确列出了所有要分析的核心具体概念，构成一个封闭的分析框架。例如，标题包含'A、B和C'、'X和Y'或'X对Y的影响'。即使标题包含'影响'一词，但如果它明确指出了施加影响的具体变量（如'A对B的影响'），那么它仍然属于此类。",
                "Multiple Explanators": "论文的标题使用了宽泛的、开放式的探索性词语，如'决定因素'、'因素'、'原因'或'影响'（当它没有指明影响来源时），暗示了对一个现象的多种未指明原因的广泛研究。例如，《银行利率的决定因素》或《影响企业创新的因素研究》。"
            },
            "schema": {
                "classification": "[Single Explanatory|Multiple Explanators]"
            }
        },
        "paper_title": title
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_research_classification_prompt(introductory_part: str) -> str:
    """Build the research-article vs non-research-article classification prompt."""
    prompt = {
        "persona": "《经济研究》期刊的资深编辑助理",
        "task": "根据论文中是否包含回归方程，判断其是研究型文章（Research Article）还是非研究型文章（Non-Research Article）。请使用下面的详细定义进行准确分类。",
        "output_format": {
            "description": "根据论文是否包含回归方程，将其分类为以下两个类别之一。",
            "classification_definitions": {
                "Non-Research Article": "论文不包含实证估计/实证回归/实证检测/实证比较。",
                "Research Article": "论文明确包含一个或多个实证估计/实证回归/实证检测/实证比较。"
            },
            "schema": {
                "classification": "[Research Article|Non-Research Article]"
            }
        },
        "paper_title & paper_abstract & introduction": introductory_part
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_measurement_prompt(introductory_part: str) -> str:
    """Build the measurement-article classification prompt."""
    prompt = {
        "persona": "《经济研究》期刊的资深编辑助理",
        "task": "基于'研究终点'将论文机械地划分为'测量型'或'非测量型'。",
        "output_format": {
            "classification_definitions": {
                "Measurement Article": "纯指标测度研究终点仅为一个孤立的数值指标（如：测算某年的贸易限制指数、关税等效值）。判定准则：论文不涉及任何'X对Y的影响'或'因果推断'。如果论文只是单纯算出一个'尺子'本身的数值，即归为此类。",
                "Non-Measurement Article": "影响分析与机制检验只要论文涉及'X对Y的影响/贡献'、'机制分析'、'政策效应评估'或'增长核算'，一律归为此类。判定准则：任何包含'因果推断'、'要素贡献分解（如增长核算）'、'脉冲响应'或'体制对比'的论文，严禁判定为测量型。研究终点是逻辑结论而非孤立数值。"
            },
            "schema": {
                "classification": "[Measurement Article|Non-Measurement Article]"
            }
        },
        "introductory_part": introductory_part
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


# ============================================================
# Generic API Caller
# ============================================================

def call_kimi_api(system_message: str, user_prompt: str) -> str:
    """
    Call the Kimi API with retry logic.

    Returns the raw text content from the model, or raises RuntimeError
    after exhausting all retries.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client_kimi.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                seed=SEED,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            print(f"  API attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(f"All {MAX_RETRIES} API attempts failed.")


def parse_json_response(text: str) -> dict:
    """
    Parse a JSON response from the model.
    Raises json.JSONDecodeError if the text is not valid JSON.
    """
    return json.loads(text)


# ============================================================
# Individual Classification Functions
# ============================================================

def classify_data_model(introductory_part: str) -> tuple[str, str]:
    """
    Determine whether the paper uses real-world data and/or a formal model.

    Returns: (has_data, has_model) — each "Yes", "No", or "ERROR".
    """
    system = "你是一位资深编辑助理。请根据标题、摘要和引言判断论文是否包含数据和模型。仅以有效的JSON格式回复。"
    try:
        text   = call_kimi_api(system, build_data_model_prompt(introductory_part))
        result = parse_json_response(text)
        return result.get("has_data", "No"), result.get("has_model", "No")
    except json.JSONDecodeError:
        print(f"  Warning: could not parse JSON response: {text}")
        return "ERROR", "ERROR"
    except RuntimeError:
        return "ERROR", "ERROR"


def classify_structural(introductory_part: str) -> str:
    """
    Determine whether the paper is a structural-empirical paper.

    Returns: "Yes", "No", or "ERROR".
    """
    system = "你是一位资深编辑助理。请判断这是否为结构实证论文。仅以有效的JSON格式回复。"
    try:
        text   = call_kimi_api(system, build_structural_prompt(introductory_part))
        result = parse_json_response(text)
        return result.get("structural_classification", "No")
    except json.JSONDecodeError:
        # Fallback: keyword scan
        lower = text.lower()
        if "yes" in lower or "是" in text:
            return "Yes"
        return "No"
    except RuntimeError:
        return "ERROR"


def classify_non_structural_type(introductory_part: str) -> str:
    """
    Sub-classify non-structural or data-only papers.

    Returns: "Pure Empirical", "Pure Theory", "Other", or "ERROR".
    """
    system = "你是一位资深编辑助理。请对非结构论文或包含数据但无模型的论文进行类型分类。仅以有效的JSON格式回复。"
    try:
        text   = call_kimi_api(system, build_non_structural_prompt(introductory_part))
        result = parse_json_response(text)
        return result.get("classification", "Other")
    except json.JSONDecodeError:
        lower = text.lower()
        if "pure empirical" in lower:
            return "Pure Empirical"
        if "pure theory" in lower:
            return "Pure Theory"
        return "Other"
    except RuntimeError:
        return "ERROR"


def classify_explanatory_variables(title: str) -> str:
    """
    Classify the paper by the number of explanatory variables in its title.

    Returns: "Single Explanatory", "Multiple Explanators", "Other", or "ERROR".
    """
    system = "你是一位资深编辑助理。请根据论文标题结构对论文进行类型分类。仅以有效的JSON格式回复。"
    try:
        text   = call_kimi_api(system, build_explanatory_variables_prompt(title))
        result = parse_json_response(text)
        return result.get("classification", "Other")
    except json.JSONDecodeError:
        lower = text.lower()
        if "single explanatory" in lower:
            return "Single Explanatory"
        if "multiple explanators" in lower:
            return "Multiple Explanators"
        return "Other"
    except RuntimeError:
        return "ERROR"


def classify_research_article(introductory_part: str) -> str:
    """
    Determine whether the paper is a Research Article or Non-Research Article.

    Returns: "Research Article", "Non-Research Article", "Other", or "ERROR".
    """
    system = "你是一位资深编辑助理。请判断所提供的论文是研究型文章还是非研究型文章。仅以有效的JSON格式回复。"
    try:
        text   = call_kimi_api(system, build_research_classification_prompt(introductory_part))
        result = parse_json_response(text)
        return result.get("classification", "Other")
    except json.JSONDecodeError:
        lower = text.lower()
        if "non-research article" in lower:
            return "Non-Research Article"
        if "research article" in lower:
            return "Research Article"
        return "Other"
    except RuntimeError:
        return "ERROR"


def classify_measurement(introductory_part: str) -> str:
    """
    Determine whether the paper is a Measurement Article or Non-Measurement Article.

    Returns: "Measurement Article", "Non-Measurement Article", "Other", or "ERROR".
    """
    system = "你是一位资深编辑助理。请判断所提供的论文是测量型论文还是非测量型论文。仅以有效的JSON格式回复。"
    try:
        text   = call_kimi_api(system, build_measurement_prompt(introductory_part))
        result = parse_json_response(text)
        return result.get("classification", "Other")
    except json.JSONDecodeError:
        lower = text.lower()
        if "non-measurement article" in lower:
            return "Non-Measurement Article"
        if "measurement article" in lower:
            return "Measurement Article"
        return "Other"
    except RuntimeError:
        return "ERROR"


# ============================================================
# Hierarchical Classification Orchestrator
# ============================================================

def classify_paper_hierarchical(
    title: str, introductory_part: str
) -> tuple[str, str, str, str, str, str]:
    """
    Classify a single paper through the full hierarchical decision tree.

    Returns:
        (has_data, has_model, paper_type, variable, research_article, measurement)
    """
    print("  Starting hierarchical classification...")

    # Step 1: Data & model presence
    has_data, has_model = classify_data_model(introductory_part)
    print(f"    has_data={has_data}, has_model={has_model}")
    time.sleep(1)

    variable       = ""
    research_article = ""
    measurement    = ""

    # Step 2: Branch on data / model presence
    if has_data == "ERROR" or has_model == "ERROR":
        print("    Classification aborted due to API error.")
        return has_data, has_model, "ERROR", variable, research_article, measurement

    if has_data == "Yes" and has_model == "Yes":
        print("    Branch: data=Yes & model=Yes")
        is_structural = classify_structural(introductory_part)
        print(f"    is_structural={is_structural}")
        time.sleep(1)

        if is_structural == "Yes":
            paper_type  = "Structural"
            print("    Paper is Structural — checking Measurement type...")
            measurement = classify_measurement(introductory_part)
            print(f"    measurement={measurement}")
            time.sleep(1)
        else:
            paper_type = classify_non_structural_type(introductory_part)
            print(f"    paper_type={paper_type}")
            if paper_type in ("Pure Empirical", "Other"):
                research_article = classify_research_article(introductory_part)
                print(f"    research_article={research_article}")
                if research_article == "Research Article":
                    variable = classify_explanatory_variables(title)
                    print(f"    variable={variable}")

    elif has_data == "Yes" and has_model == "No":
        print("    Branch: data=Yes & model=No")
        paper_type       = "Pure Empirical"
        research_article = classify_research_article(introductory_part)
        print(f"    research_article={research_article}")
        if research_article == "Research Article":
            variable = classify_explanatory_variables(title)
            print(f"    variable={variable}")

    elif has_data == "No" and has_model == "Yes":
        print("    Branch: data=No & model=Yes")
        paper_type = "Pure Theory"

    else:
        print("    Branch: data=No & model=No")
        paper_type       = "Other"
        research_article = classify_research_article(introductory_part)
        print(f"    research_article={research_article}")
        if research_article == "Research Article":
            variable = classify_explanatory_variables(title)
            print(f"    variable={variable}")

    return has_data, has_model, paper_type, variable, research_article, measurement


# ============================================================
# I/O Helpers
# ============================================================

def read_json_intro(json_path: str) -> str:
    """
    Load a JSON file and return the content of the first 'introductory-part' item.
    Returns an empty string on any error.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data:
            if isinstance(item, dict) and item.get("type") == "introductory-part":
                return item.get("content", "")
    except Exception as exc:
        print(f"  Warning: failed to read JSON file {json_path}: {exc}")
    return ""


def read_csv(csv_path: str) -> pd.DataFrame | None:
    """
    Attempt to read a CSV file trying several common encodings.
    Returns a DataFrame on success, or None if all encodings fail.
    """
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"):
        try:
            print(f"  Trying encoding: {encoding}...")
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"  Successfully read {len(df)} rows with encoding '{encoding}'.")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            print(f"  Error with encoding '{encoding}': {exc}")
    print(f"  ERROR: could not read file with any known encoding: {csv_path}")
    return None


def get_temp_path(csv_path: str) -> str:
    """Return the path used for the incremental checkpoint file."""
    return os.path.join(os.path.dirname(csv_path), "temp_classification_checkpoint.csv")


def is_valid_record(row: pd.Series) -> bool:
    """
    Return True if a row already has a valid, non-error classification.
    """
    value = row.get("paper_type", "")
    if pd.isna(value):
        return False
    return str(value).strip() not in ("", "ERROR", "FILE_NOT_FOUND", "NO_INTRO", "NO_PATH")


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing output columns (initialised to empty string)."""
    for col in ("has_data", "has_model", "paper_type", "research_article", "variable", "measurement"):
        if col not in df.columns:
            df[col] = ""
    return df


def save_checkpoint(df: pd.DataFrame, path: str, label: str = "") -> None:
    """Save the DataFrame to a CSV checkpoint file."""
    df.to_csv(path, index=False, encoding="utf-8-sig")
    suffix = f" ({label})" if label else ""
    print(f"  [Checkpoint saved{suffix}: {path}]")


# ============================================================
# Main Processing Loop
# ============================================================

def process_papers(csv_path: str, json_dir: str) -> None:  # noqa: ARG001  (json_dir kept for API parity)
    """
    Main entry point: load the CSV, classify each paper, and write results.
    Supports resuming from a checkpoint file if a previous run was interrupted.
    """
    temp_path = get_temp_path(csv_path)

    # ── Load data (prefer checkpoint if it exists) ──────────────────────────
    print("Checking for a resume checkpoint...")
    if os.path.exists(temp_path):
        print(f"  Checkpoint found — resuming from: {temp_path}")
        try:
            df = pd.read_csv(temp_path, encoding="utf-8-sig")
            valid_count = df[is_valid_record(df.iloc[0:0])  # just for type-check
                             if False else df.apply(is_valid_record, axis=1)].shape[0]
            print(f"  {len(df)} rows loaded; {valid_count} already classified.")
        except Exception as exc:
            print(f"  Failed to read checkpoint ({exc}). Falling back to original CSV.")
            df = read_csv(csv_path)
            if df is None:
                return
    else:
        print("  No checkpoint found — starting from the original CSV.")
        df = read_csv(csv_path)
        if df is None:
            return

    # ── Validate required columns ────────────────────────────────────────────
    required = ("cn_json_path", "json_title")
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing required column(s): {missing}")
        print(f"  Available columns: {df.columns.tolist()}")
        return

    df = ensure_output_columns(df)

    # ── Build a cache of already-classified papers (keyed by json path) ──────
    paper_cache: dict[str, tuple] = {}
    for _, row in df.iterrows():
        if is_valid_record(row):
            key = str(row.get("cn_json_path", "")).strip()
            if key:
                paper_cache[key] = (
                    row.get("has_data",         ""),
                    row.get("has_model",         ""),
                    row.get("paper_type",        ""),
                    row.get("variable",          ""),
                    row.get("research_article",  ""),
                    row.get("measurement",       ""),
                )
    print(f"  Cache populated with {len(paper_cache)} previously classified paper(s).\n")

    # ── Classification loop ──────────────────────────────────────────────────
    print("Starting classification loop...")
    print("=" * 60)

    counts = {"processed": 0, "skipped": 0, "duplicated": 0, "error": 0}
    total  = len(df)

    for idx, row in df.iterrows():
        json_path = row["cn_json_path"]
        title     = row["json_title"]
        prefix    = f"Row {idx + 1}/{total}"

        # Already has a valid classification — skip
        if is_valid_record(row):
            print(f"{prefix}: already classified [{row.get('paper_type', '')}] — skipping.")
            counts["skipped"] += 1
            continue

        # Validate json_path
        if pd.isna(json_path) or not str(json_path).strip():
            print(f"{prefix}: empty JSON path — marking NO_PATH.")
            df.at[idx, "paper_type"] = "NO_PATH"
            counts["error"] += 1
            continue

        json_path = str(json_path).strip()

        # Duplicate detection: reuse cached result
        if json_path in paper_cache:
            print(f"{prefix}: duplicate detected — reusing cached result for {os.path.basename(json_path)}.")
            has_data, has_model, paper_type, variable, research_article, measurement = paper_cache[json_path]
            df.at[idx, "has_data"]          = has_data
            df.at[idx, "has_model"]         = has_model
            df.at[idx, "paper_type"]        = paper_type
            df.at[idx, "variable"]          = variable
            df.at[idx, "research_article"]  = research_article
            df.at[idx, "measurement"]       = measurement
            counts["duplicated"] += 1
            if (counts["duplicated"] + counts["processed"]) % SAVE_EVERY == 0:
                save_checkpoint(df, temp_path, "periodic")
            continue

        # File existence check
        if not os.path.exists(json_path):
            print(f"{prefix}: file not found — {json_path}")
            df.at[idx, "paper_type"] = "FILE_NOT_FOUND"
            counts["error"] += 1
            continue

        # Prepare inputs
        title = "" if pd.isna(title) else str(title).strip()
        intro = read_json_intro(json_path)
        short_title = title[:50] + "..." if len(title) > 50 else title
        print(f"\n{prefix}: {os.path.basename(json_path)}")
        print(f"  Title: {short_title}")

        if not intro:
            print(f"  Warning: no introductory-part found in JSON — marking NO_INTRO.")
            df.at[idx, "paper_type"] = "NO_INTRO"
            counts["error"] += 1
            continue

        # Classify
        try:
            has_data, has_model, paper_type, variable, research_article, measurement = \
                classify_paper_hierarchical(title, intro)

            df.at[idx, "has_data"]         = has_data
            df.at[idx, "has_model"]        = has_model
            df.at[idx, "paper_type"]       = paper_type
            df.at[idx, "variable"]         = variable
            df.at[idx, "research_article"] = research_article
            df.at[idx, "measurement"]      = measurement

            paper_cache[json_path] = (has_data, has_model, paper_type, variable, research_article, measurement)
            counts["processed"] += 1

            if counts["processed"] % SAVE_EVERY == 0:
                save_checkpoint(df, temp_path, f"{counts['processed']} processed")

            time.sleep(API_DELAY)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user — saving checkpoint...")
            save_checkpoint(df, temp_path, "interrupted")
            print("  Resume by running the script again; it will pick up from the checkpoint.")
            return

        except Exception as exc:
            print(f"  Classification error: {exc}")
            df.at[idx, "paper_type"] = "ERROR"
            counts["error"] += 1
            save_checkpoint(df, temp_path, "after error")

    # ── Save final output ────────────────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name   = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(os.path.dirname(csv_path), f"{base_name}_classified_{timestamp}.csv")

    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\nResults saved to: {output_path}")

        # Clean up checkpoint
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("  Checkpoint file removed.")

        # ── Summary statistics ───────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Classification Summary")
        print("=" * 60)
        print(f"  Total rows          : {total}")
        print(f"  Newly classified    : {counts['processed']}")
        print(f"  Reused from cache   : {counts['duplicated']}")
        print(f"  Skipped (had result): {counts['skipped']}")
        print(f"  Errors / skipped    : {counts['error']}")
        print(f"  Unique papers seen  : {len(paper_cache)}")

        def print_distribution(label: str, column: str) -> None:
            series = df.loc[df[column].astype(str).str.strip() != "", column]
            if series.empty:
                print(f"\n  {label}: (no records)")
                return
            print(f"\n  {label}:")
            for value, cnt in series.value_counts().items():
                print(f"    {value}: {cnt}")

        print_distribution("has_data distribution",        "has_data")
        print_distribution("has_model distribution",       "has_model")
        print_distribution("paper_type distribution",      "paper_type")
        print_distribution("research_article distribution","research_article")
        print_distribution("variable distribution",        "variable")
        print_distribution("measurement distribution",     "measurement")

    except Exception as exc:
        print(f"ERROR: failed to save output file: {exc}")
        backup_path = os.path.join(os.path.dirname(csv_path), f"backup_result_{timestamp}.csv")
        try:
            df.to_csv(backup_path, index=False, encoding="utf-8-sig")
            print(f"  Backup saved to: {backup_path}")
        except Exception:
            print("  Backup save also failed.")


# ============================================================
# Entry Point
# ============================================================

def main() -> None:
    print("=" * 60)
    print("Paper Classification Pipeline")
    print("=" * 60)
    print(f"  CSV path      : {CSV_PATH}")
    print(f"  Checkpoint    : {get_temp_path(CSV_PATH)}")
    print(f"  JSON paths are read directly from the 'cn_json_path' column.")
    print()

    if not os.path.exists(CSV_PATH) and not os.path.exists(get_temp_path(CSV_PATH)):
        print(f"ERROR: neither the CSV file nor a checkpoint file was found.")
        print(f"  CSV path : {CSV_PATH}")
        return

    try:
        process_papers(CSV_PATH, JSON_DIR)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Pipeline finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
