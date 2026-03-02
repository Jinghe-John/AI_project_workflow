# 大规模学术文档智能处理与跨语言内容分析流水线

> **Large-Scale Academic Document Intelligence & Cross-Lingual Content Analysis Pipeline**

---


## 目录

1. [系统概览](#系统概览)
2. [数据流向总览](#数据流向总览)
3. [Step 01 · 多GPU并行PDF解析 `multi_gpu_process.py`](#step-01--多gpu并行-pdf-解析)
4. [Step 02 · 通用JSON字段清洗 `clean_general.py`](#step-02--通用-json-字段清洗)
5. [Step 03 · 中文文档结构化处理 `process_cn.py`](#step-03--中文文档-json-结构化处理)
6. [Step 04 · 英文文档结构化处理 `process_en.py`](#step-04--英文文档-json-结构化处理)
7. [Step 05/06 · 文档分层自动分类 `cn_classify.py` / `en_classify.py`](#step-0506--文档分层自动分类)
8. [Step 07 · 实证类文档跨语言相似性检测 `reduced_form_similarity.py`](#step-07--实证类文档跨语言相似性检测)
9. [Step 08 · 结构类文档跨语言相似性检测 `structural_similarity.py`](#step-08--结构类文档跨语言相似性检测)
10. [Step 09 · 文档地域来源识别 `doc_geo_tagging.py`](#step-09--文档地域来源识别)
11. [Step 10 · 文档作者信息提取 `cn_doc_author.py`](#step-10--文档作者信息提取)
12. [附录：环境配置与依赖](#附录环境配置与依赖总览)
13. [附录：常见问题](#附录常见问题)
14. [附录：数据表格说明](#附录数据表格说明)

---

## 系统概览

本系统是一套面向**大规模结构化学术文档**的端到端自动化处理与跨语言内容分析流水线。系统从原始 PDF 出发，经过多 GPU 并行解析、JSON 清洗与结构化、基于大语言模型（LLM）的自动分类，最终完成中英文文档对的多维度语义相似性分析，并提取文档的地域来源与作者信息。

整套流程由 **10 个 Python 脚本**按顺序协同完成，覆盖从数据预处理到结果后处理的全生命周期，具备断点续传、多 Key 并发、路径级去重缓存等生产级工程能力。

| 步骤 | 脚本文件 | 核心功能 | 输出 |
|:---:|---|---|---|
| 01 | `multi_gpu_process.py` | 多GPU并行PDF解析 | JSON 原始文件 |
| 02 | `clean_general.py` | 通用JSON字段清洗 | 精简JSON文件 |
| 03 | `process_cn.py` | 中文文档结构化处理 | 结构化中文JSON |
| 04 | `process_en.py` | 英文文档结构化处理 | 结构化英文JSON |
| 05 | `cn_classify.py` | 中文文档分层分类 | 分类标签CSV |
| 06 | `en_classify.py` | 英文文档分层分类 | 分类标签CSV |
| 07 | `reduced_form_similarity.py` | 实证类文档相似性检测 | 检测结果CSV |
| 08 | `structural_similarity.py` | 结构类文档相似性检测 | 检测结果CSV |
| 09 | `doc_geo_tagging.py` | 文档地域来源识别 | 地域标注Excel |
| 10 | `cn_doc_author.py` | 文档作者信息提取 | 作者信息Excel |

---

## 数据流向总览

```
原始 PDF 文件
  │
  ▼  [Step 01] multi_gpu_process.py
  │  多GPU并行，4× A800，以文件夹为调度单位，调用 MinerU CLI
  │
JSON 原始输出（含 *_model.json，包含 bbox / score 等大量中间字段）
  │
  ▼  [Step 02] clean_general.py
  │  递归提取 type + content，剔除所有噪声字段
  │
精简 JSON（每个节点仅保留 type / content 两个字段）
  │
  ├──▶  [Step 03] process_cn.py  →  结构化中文JSON
  │     按章节重组，生成 Title + Introductory-Part + page_footnote
  │
  └──▶  [Step 04] process_en.py  →  结构化英文JSON
        结合 CSV 元信息，三级标题匹配定位正文起点，生成 introductory-part
  │
  ▼  [Step 05] cn_classify.py（中文）  +  [Step 06] en_classify.py（英文）
  │  调用 LLM API，六层决策树分类
  │
分类标签 CSV（has_data / has_model / doc_type / measurement 等）
  │
  ├──▶  [Step 07] reduced_form_similarity.py → 处理 Pure Empirical 文档对
  │
  └──▶  [Step 08] structural_similarity.py   → 处理 Structural + Non-Measurement 文档对
  │
相似性检测结果 CSV（similarity / research_question / research_design 等）
  │
  ├──▶  [Step 09] doc_geo_tagging.py  →  提取中英文文档数据来源地域
  │
  └──▶  [Step 10] cn_doc_author.py    →  提取中文文档作者姓名/机构/邮编/邮箱
  │
最终 Excel（高相似度条目 + 地域标注 + 作者信息）
```

---

## Step 01 · 多GPU并行 PDF 解析

**脚本：** `multi_gpu_process.py`

将原始 PDF 目录树批量转换为 MinerU 解析的 JSON 文件，充分利用多张 A800 显卡加速大规模处理。本脚本是整个流水线的**数据入口**。

### 设计思路

以**文件夹**为最小调度单位，将文件夹任务动态分配给各 GPU 对应的工作进程。同一文件夹内的 PDF 由同一 GPU 顺序处理，既保证负载均衡，又避免 GPU 资源竞争。

### 关键配置参数

在 `main()` 函数顶部修改：

```python
input_root = '/data/.../en_unzip'    # PDF 根目录，脚本递归扫描所有子文件夹
output_dir = 'en_output'             # MinerU 输出目录
csv_file   = 'pdf_location_info.csv' # 处理结果记录文件
num_gpus   = 4                       # 使用 GPU 数量（默认 4× A800）
```

单文件夹处理超时上限默认为 **3600 秒（1小时）**，可在 `process_folder()` 中修改 `timeout` 参数。

### 运行

```bash
python multi_gpu_process.py
```

### 处理架构

```
主进程
  ├── 扫描输入目录 → 构建文件夹任务队列
  ├── 启动 N 个子进程（每个通过 CUDA_VISIBLE_DEVICES 绑定一张GPU）
  │     └── 子进程循环取任务 → 调用 MinerU CLI → 上报结果
  └── 主进程读取结果队列 → 实时写入CSV → 每10个文件夹打印进度
```

启动时各进程错开 **2秒** 以避免 GPU 初始化阶段的资源竞争。`Ctrl+C` 可安全中断，已处理结果均保留在 CSV 中。

> ⚠️ 处理失败的 PDF 文件需手动转换为 PNG 后重新解析。

### 输出：`pdf_location_info.csv`

| 字段 | 说明 |
|---|---|
| `filename` | PDF 文件名（不含扩展名） |
| `full_path` | PDF 完整路径 |
| `is_corrupted` | `FALSE` = 成功 / `TRUE` = 失败 |
| `error_message` | 失败时的错误信息（最多 500 字符） |
| `process_time` | 该文件的平均处理耗时（秒） |

---

## Step 02 · 通用 JSON 字段清洗

**脚本：** `clean_general.py`

批量遍历 MinerU 输出的 `*_model.json` 文件，**递归提取 `type` 与 `content` 字段**，剔除所有无关字段（bbox 坐标、置信分、字体信息等），生成精简 JSON 并输出 Excel 处理报告。

### 清洗规则

核心函数 `clean_json_content()` 对 JSON 进行深度递归：

- **列表中的字典元素**：只保留 `type` 和 `content`，若两者均不存在则跳过（不输出空字典）
- **列表中的嵌套列表**：继续递归处理
- **字典的键值对**：若值为列表或字典则递归，其他类型直接保留
- 字段过滤**只发生在列表内的字典元素**层面，字典本身的键不过滤

### 清洗前后对比

```json
// 清洗前（原始）
{ "type": "text", "content": "正文内容", "bbox": [100,200,300,400], "score": 0.98, "page_id": 1 }

// 清洗后（输出）
{ "type": "text", "content": "正文内容" }
```

### 运行

```bash
# 基本用法（保持原目录结构）
python clean_general.py -i ./en_output -o ./cleaned_output

# 平铺输出，所有文件放在同一目录
python clean_general.py -i ./input -o ./output --flat

# 自定义目标文件后缀
python clean_general.py -i ./input -o ./output --suffix "_result.json"
```

### 参数说明

| 参数 | 简写 | 必填 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `--input` | `-i` | ✅ | — | 输入目录路径 |
| `--output` | `-o` | ✅ | — | 输出目录路径 |
| `--suffix` | `-s` | ❌ | `_model.json` | 目标文件后缀 |
| `--flat` | — | ❌ | 否 | 平铺输出，不保持原目录结构 |
| `--report` | `-r` | ❌ | 含时间戳 | 自定义报告文件名 |

### 输出说明

- 清洗后文件名将后缀替换为 `.json`，例如：`2014_09_12_9_model.json` → `2014_09_12_9.json`
- Excel 报告包含**详细记录**（每个文件的处理状态、路径、错误信息）与**统计摘要**（总数/成功数/失败数/成功率）两个 Sheet

---

## Step 03 · 中文文档 JSON 结构化处理

**脚本：** `process_cn.py`

对清洗后的中文 JSON 文件进行语义重组：按章节合并内容，识别并标记无中文字符的异常文件，生成结构统一的标准化学术 JSON。

### 设计思路

将原始扁平列表转换为**以章节为单位、语义清晰的结构化 JSON**，过滤页眉、元数据等噪声，同时检测文件是否含有中文字符，自动标记可能存在语言错配的异常文件。

### 五步处理流程

**① 展平嵌套结构**

MinerU 输出有时为"列表的列表"（每页一个数组），读取时自动展平为单层列表，统一后续处理入口。

**② 类型过滤**

只保留以下四种类型，其余全部丢弃：

| 保留类型 | 说明 |
|---|---|
| `title` | 章节标题（清除其中的 `*` 和 `#` 符号） |
| `text` | 正文段落 |
| `equation` | 数学公式 |
| `page_footnote` | 页脚注释 |

同时**跳过第一个 `title` 出现之前的所有内容**，避免保留页眉、作者信息栏等干扰项。

**③ 合并相邻标题**

若文档开头存在两个连续 `title`（中间无正文），自动将其合并为一个，避免文档标题被错误拆分：

```
title: "基于深度学习的"  +  title: "图像识别研究"
  →  title: "基于深度学习的 图像识别研究"
```

**④ 生成 Title 与 Introductory-Part**

将第一个 `title` 提升为特殊类型 `Title`，将 `Title` 到第三个 `title` 之间的所有内容（摘要、关键词、作者信息等）合并为一个 `Introductory-Part` 块。`page_footnote` 独立保留，置于 `Introductory-Part` 之后。

若文档不存在第三个 `title`，则将所有剩余内容都归入 `Introductory-Part`。

**⑤ 按章节合并正文**

遍历剩余内容，将相邻的 `text` 和 `equation` 块合并为一个 `text` 块，直到遇到下一个 `title` 为止。`page_footnote` 始终独立存在，不参与合并。

### 输出结构示例

```json
[
  { "type": "Title",             "content": "文档标题" },
  { "type": "Introductory-Part", "content": "摘要：本文... 关键词：... 1. 引言..." },
  { "type": "title",             "content": "2 数据与方法" },
  { "type": "text",              "content": "本研究的数据来自XXX数据库..." },
  { "type": "page_footnote",     "content": "作者信息：..." }
]
```

### 运行

```bash
python process_cn.py -i ./cn_output -o ./cn_processed
python process_cn.py -i ./cn_output -o ./cn_processed -c ./report.csv
```

### CSV 报告关键字段

| 字段 | 说明 |
|---|---|
| `是否包含中文` | 是 / 否 |
| `中文字符数` | 文件中中文字符总数 |
| `是否异常文件` | 无中文字符则标记为异常，同步写入 `*_abnormal.csv` |
| `Title数 / Introductory-Part数 / title数 / text数` | 各类型块的数量统计 |
| `处理状态` | 成功 / 失败（失败时附错误信息） |

---

## Step 04 · 英文文档 JSON 结构化处理

**脚本：** `process_en.py`

结合 CSV 元信息与 JSON 内容，通过**三级标题匹配策略**定位文档正文起点，裁剪无关内容，生成结构统一的英文标准化 JSON。

### 设计思路

英文处理比中文复杂：同一 PDF 可能包含多篇文档的拼接内容。需借助 CSV 中的参考标题（`Ref_Title`）精准定位当前文档的起始位置，然后截断其他内容。三级匹配策略应对标题格式差异大的挑战。

### 总体流程

```
读取 CSV → 过滤 1980 年前文献
  ↓
定位 JSON 文件（按 doc_id 年份前4位找子目录）
  ↓
清洗 JSON（类型过滤 + 相邻 text/equation 按章节合并）
  ↓
标题匹配（Ref_Title vs JSON 内部所有 title）
  ├── 匹配成功 → 裁剪内容 + 生成 introductory-part
  └── 匹配失败 → 保留全部内容，跳过转换
  ↓
保存结构化 JSON + 写入 CSV 报告
```

### 三级标题匹配策略

将 CSV 中的 `Ref_Title` 与 JSON 内部所有 `title` 依次尝试，前一种成功则不再尝试后续。文本标准化步骤：Unicode 引号/上下标替换 → 转小写 → NFKC 归一化 → 去除标点 → 去除空格。

| 级别 | 方法 | 规则 |
|:---:|---|---|
| ① | 精确匹配 `exact` | 标准化处理后直接比较字符串是否完全相等 |
| ② | 模糊匹配 `fuzzy` | 编辑距离相似度（`fuzz.ratio`）≥ 80 则命中 |
| ③ | 子集匹配 `subset` | 包含关系 / 长度相近且相似度 ≥ 90 / 分块比对（80% 块命中） |

### 匹配结果类型与处理策略

| 结果类型 | 处理方式 |
|---|---|
| `one_to_one`（唯一匹配） | 直接以命中 title 为内容起始位置 |
| `one_to_many`（多处命中） | 取第一个为起始，删除其余重复命中的 title |
| `no_match`（无匹配） | 保留全部内容，跳过 introductory-part 转换 |

### introductory-part 生成逻辑

裁剪完成后，查找含 `introduction` 或以 `1` / `1.` / `I` / `I.` 开头的 title 作为引言入口，将 `Title → 引言 title → 引言第一段正文` 合并为 `introductory-part` 块。若未找到，启用备用方案（按文档 title 总数选定目标 title）。

### 运行

```bash
python process_en.py -i all_doc_info.csv -j ./en_output -o ./en_processed_json
python process_en.py -i all_doc_info.csv -j ./en_output -o ./en_processed_json -c ./report.csv
```

### 输入 CSV 必要列

| 列名 | 说明 |
|---|---|
| `doc_id` | 文献ID，前4位为年份，用于定位 JSON 子目录 |
| `Year` | 发表年份，自动过滤 1980 年前文献 |
| `Ref_Title` | 参考标题，用于与 JSON 内部 title 进行匹配 |

---

## Step 05/06 · 文档分层自动分类

**脚本：** `cn_classify.py`（中文版）/ `en_classify.py`（英文并行版）

调用 LLM API，通过**六层决策树**对文档进行自动分类。分类依据为 JSON 文件中的 `introductory-part` 字段（标题 + 摘要 + 引言）。

### 两个脚本的区别

| 维度 | `cn_classify.py`（中文版） | `en_classify.py`（英文版） |
|---|---|---|
| 处理对象 | 中文文档全量 | 含英文对应版本的中文文档子集 |
| JSON 路径列 | `cn_json_path` | `en_json_path` |
| 标题列 | `json_title` | `Ref_Title` |
| 结果列前缀 | `has_data` / `doc_type` 等 | `en_has_data` / `en_doc_type` 等 |
| Prompt 语言 | 中文 | 英文 |

### 六层分类决策树

```
第一层：has_data & has_model（是否使用真实数据 / 是否建立正式模型）
  │
  ├─ 有数据 & 有模型
  │     ↓
  │   第二层：is_structural（是否含反事实分析或校准）
  │     ├─ Yes → doc_type = Structural → 第六层：measurement 判断
  │     └─ No  → 第三层：Non-Structural 细分
  │                 ↓
  │             Pure Empirical / Pure Theory / Other
  │             （Pure Empirical & Other 进入第四、五层）
  │
  ├─ 有数据 & 无模型 → doc_type = Pure Empirical → 第四、五层
  ├─ 无数据 & 有模型 → doc_type = Pure Theory（结束）
  └─ 无数据 & 无模型 → doc_type = Other → 第四、五层

第四层：research_article（是否包含实证估计/回归/检测）
  ↓（仅 Research Article 进入第五层）
第五层：variable（标题结构：封闭分析框架 vs 开放式探索词）
  Single Explanatory / Multiple Explanators

第六层：measurement（仅 Structural 触发）
  研究终点是否为孤立数值指标
  Measurement Article / Non-Measurement Article
```

### 各层判定标准

**第一层：数据与模型判断**

| 字段 | Yes 定义 | No 定义 |
|---|---|---|
| `has_data` | 使用真实世界数据，含实证分析、数值模拟或 CGE 模型 | 未使用真实世界数据 |
| `has_model` | 开发或显著扩展了正式数学/理论/计算模型或统计检验 | 未引入新的正式模型或统计检验 |

**第二层：结构实证判断**（仅 `has_data=Yes & has_model=Yes` 时触发）

| 结果 | 定义 |
|---|---|
| `Structural` | 文档包含**反事实分析**或**校准** |
| `Non-Structural` | 文档不包含反事实分析和校准 |

**第五层：解释变量数量**（基于文档标题结构判断）

| 类型 | 定义 |
|---|---|
| `Single Explanatory` | 标题明确列出所有核心概念，构成封闭分析框架（如"X 对 Y 的影响"） |
| `Multiple Explanators` | 标题使用开放式探索性词语（如"决定因素"、"Determinants"、"Factors"） |

### 断点续传机制

- **`cn_classify.py`**：每处理10条保存 `temp_classification_result.csv`，下次启动自动从断点恢复；全部完成后生成带时间戳最终文件
- **`en_classify.py`**：每处理5条保存 `classification_checkpoint.pkl`，每20条额外保存中间 CSV；已有有效记录的行（非 `ERROR`/`FILE_NOT_FOUND`）自动跳过

### 英文版并行架构（`en_classify.py`）

10 个 API Key 对应 10 个 Worker，通过 `ThreadPoolExecutor` 并发提交任务，每个 Worker 独占一个 Client 实例互不干扰。

### 关键配置

```python
# cn_classify.py
client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.example.com/v1")
CSV_PATH  = os.path.join(BASE_DIR, "all_docs_unclassified.csv")

# en_classify.py
API_KEYS   = ["key-1", "key-2", ...]   # Key 数量决定并发线程数
INPUT_FILE = "all_docs.csv"
```

### 输出新增列

| 列名 | 说明 |
|---|---|
| `has_data` | `Yes` / `No` / `ERROR` |
| `has_model` | `Yes` / `No` / `ERROR` |
| `doc_type` | `Structural` / `Pure Empirical` / `Pure Theory` / `Other` / `ERROR` / `FILE_NOT_FOUND` / `NO_INTRO` |
| `research_article` | `Research Article` / `Non-Research Article` / 空（不适用） |
| `variable` | `Single Explanatory` / `Multiple Explanators` / 空（不适用） |
| `measurement` | `Measurement Article` / `Non-Measurement Article` / 空（不适用） |

英文版列名统一加 `en_` 前缀，如 `en_has_data`、`en_doc_type` 等。

---

## Step 07 · 实证类文档跨语言相似性检测

**脚本：** `reduced_form_similarity.py`

对实证类（Reduced-Form / Pure Empirical）中英文文档对进行**多维度、双轮交叉验证**，评估跨语言内容相似程度。这是整个流水线的**核心分析模块**，也是数量最多的文档类型，因此优先处理。

### 适用范围（四列同时满足才进入分析）

| 列名 | 条件 |
|---|---|
| `doc_type` | 必须为 `Pure Empirical` |
| `en_doc_type` | 必须为 `Pure Empirical` |
| `research_article` | 必须为 `Research Article` |
| `en_research_article` | 必须为 `Research Article` |

不满足任一条件的行将被标记为 `skipped` 并记录跳过原因。

### 三维度相似性判定框架

**三个维度同时满足才判定为高度相似；任一维度为 `False` 或 `ERROR`，均不判定为高度相似。**

| 维度 | 判定标准 |
|---|---|
| **研究问题 (RQ)** | 中英文文档聚焦同一因果关系：核心的被解释变量与解释变量在语义上高度等价，研究框架基本一致 |
| **研究设计 (RD)** | 采用相同或高度类似的识别策略：工具变量逻辑相同、自然实验设计相同，或 DID/RDD 的处理变量与对照组构造逻辑相同 |
| **机制分析 (Mech)** | 机制解释框架功能等价，传导路径逻辑实质相同 |

### 典型案例对照

| 情形 | 判定 | 原因 |
|---|:---:|---|
| 英文用移民潮作 IV 估计劳动力供给冲击；中文用同类迁移变量、同一 IV 逻辑进行估计 | ✅ 高度相似 | IV 逻辑与识别来源实质相同 |
| 英文用边界断点识别政策效应；中文用完全不同的政策时间节点做 DID | ❌ 差异显著 | 识别策略独立设计，外生性来源不同 |
| 中文文档的核心解释变量来自中文文档独有的制度变量（无英文对应物） | ❌ 差异显著 | 研究问题本身无法在英文文档中找到功能等价的映射 |

### 双轮交叉验证机制

对每对文档调用 API **两次**（Round 1 + Round 2），取两轮结果的**交集**：只有两轮均判为 `True` 的维度，最终才计为 `True`。显著提升判断可靠性，减少 LLM 随机性导致的误判。

### 候选源文档两两比对

当一篇中文文档匹配到多个英文候选来源时，系统自动对这些英文文档两两进行相似性检测，并通过 `most_likely_source` 字段标识相似程度最高的主要来源。两两比对汇总结果另存为独立 JSON 文件：

```
{原文件名}_similarity_results_{时间戳}_en_en_summary.json
```

### 运行与配置

```bash
python reduced_form_similarity.py
```

```python
BASE_URL   = "https://api.example.com/v1"
MODEL      = "your-model-name"
API_KEYS   = ["key-1", "key-2", ...]   # Key 数量决定并发线程数
input_path = r"C:\path\to\your\input.csv"
```

### 输出新增列

| 列名 | 说明 |
|---|---|
| `research_question` | 研究问题维度：`True` / `False` / `ERROR` |
| `research_design` | 研究设计维度：`True` / `False` / `ERROR` |
| `mechanism` | 机制分析维度：`True` / `False` / `ERROR` |
| `high_similarity` | 综合相似性判断（三维度均 True 才为 True） |
| `most_likely_source` | 相似程度最高的主要英文来源 |
| `round1_rq` / `round2_rq` | 第一/二轮研究问题判断 |
| `round1_rd` / `round2_rd` | 第一/二轮研究设计判断 |
| `round1_mech` / `round2_mech` | 第一/二轮机制分析判断 |
| `consistency_rq/rd/mech` | 两轮一致性（`consistent` / `inconsistent` / `error`） |
| `en_en_similarity_detected` | 是否检测到英文文档间高度相似 |
| `en_en_similarity_pairs` | 存在高度相似关系的英文文档对 |
| `skip_reason` | 跳过原因（如 `en_doc_type 为非Pure Empirical`） |

### 断点续传

运行过程中生成 `*_checkpoint.json`。中断后直接重新运行脚本，系统自动检测并从断点恢复，跳过已处理行。

---

## Step 08 · 结构类文档跨语言相似性检测

**脚本：** `structural_similarity.py`

对结构类文档中英文对进行相似性检测，逻辑框架与 Step 07 一致，但针对结构模型文档的方法论特点重新设计了判定标准与 Prompt。

### 与实证类检测模块的区别

结构类文档的相似性更隐蔽：同一套方程组和矩匹配策略可直接应用于不同数据集，在形式上看似独立研究，但在方法论层面实质等价。因此判定标准从"识别策略是否相同"转向"**模型骨架与反事实实验是否功能等价**"。

### 适用范围（四列同时满足才进入分析）

| 列名 | 条件 |
|---|---|
| `doc_type` | 必须为 `Structural` |
| `en_doc_type` | 必须为 `Structural` |
| `measurement` | 必须为 `Non-Measurement Article` |
| `en_measurement` | 必须为 `Non-Measurement Article` |

### 三维度相似性判定框架

| 维度 | 结构类文档专项判定标准 |
|---|---|
| **研究问题 (RQ)** | 通过"机制映射测试"：去除特定制度标签后，因果问题是否与英文文档完全相同 |
| **研究设计 (RD)** | 采用相同的模型骨架（偏好、技术、约束的函数形式）与相同的识别策略（如相同的 SMM/GMM 矩匹配目标） |
| **反事实分析 (CF)** | 政策模拟冲击针对同一经济边际，传导机制与福利含义功能等价 |

### 运行与配置

```bash
python structural_similarity.py
```

```python
BASE_URL   = "https://api.example.com/v1"
MODEL      = "your-model-name"
API_KEYS   = ["key-1", "key-2", ...]
input_path = r"C:\path\to\your\input.csv"
```

输出列结构与 Step 07 完全相同，断点续传机制亦一致，此处不再赘述。

---

## Step 09 · 文档地域来源识别

**脚本：** `doc_geo_tagging.py`

从相似性检测结果中，自动提取中英文文档**数据来源地域**信息，回写至 Excel。

### 设计思路

本脚本是相似性检测的**下游后处理模块**，仅处理 `high_similarity == TRUE` 的条目，节省 API 调用开销。通过调用 LLM API 分析文档 `introductory-part`，从中识别数据来源地域，用于后续地域分布分析。

### 地域标准化规则

内置映射规则确保输出标准统一：

| 原始识别 | 标准化输出 |
|---|---|
| `United States` / `U.S.` / `America` | `USA` |
| `United Kingdom` / `Great Britain` | `UK` |
| `P.R.C` / `Mainland China` | `China` |
| OECD / EU / Global / Cross-country 多国数据 | `Multiple Countries` |

### 运行与配置

```bash
python doc_geo_tagging.py
```

```python
API_KEY     = "YOUR_API_KEY"
INPUT_FILE  = "similarity_results.xlsx"
OUTPUT_FILE = "similarity_results_geo.xlsx"
BASE_URL    = "https://api.example.com/v1"
MODEL       = "your-model-name"
```

每处理 10 条自动保存一次，防止意外中断丢失结果。

### 输出新增列

| 新增列 | 说明 | 示例值 |
|---|---|---|
| `cn_country` | 中文文档的数据来源地域 | `China` |
| `en_country` | 英文文档的数据来源地域 | `USA` / `Multiple Countries` |

### 特殊值说明

| 值 | 含义 |
|---|---|
| `Multiple Countries` | 文档使用多国/地区数据（含 OECD、EU、全球样本等） |
| `Content Missing` | JSON 文件中未找到 `introductory-part` 内容 |
| `No Content` | JSON 路径为空或文件不存在 |
| `API Error` | API 调用失败，需人工核查 |

---

## Step 10 · 文档作者信息提取

**脚本：** `cn_doc_author.py`

从中文文档 JSON 的脚注（`page_footnote`）中提取作者姓名、机构、邮编与邮箱，支持**多Key并行**与**路径级去重缓存**。

### 设计思路

本脚本处理**所有行**（不按相似性结果筛选），用于后续作者网络分析与机构分布统计。核心设计亮点是路径级去重缓存机制，大幅降低 API 调用成本。

### 核心设计：路径级去重缓存

一份中文文档可能在结果表中出现多次（对应多篇英文文档）。本脚本通过 **per-path 锁机制**确保每个路径全程只调用一次 API，后续行直接复用缓存结果：

```
实际 API 调用次数 = 不重复的 cn_json_path 数量（而非表格总行数）

示例：500 行数据中有 312 个不重复路径
  → 最多打 312 次 API，节省 37.6% 调用成本
```

```
第一个线程访问路径 A → 拿锁 → 调用 API → 存入缓存 → 释放锁
后续线程访问路径 A  → 拿锁 → 直接读缓存 → 释放锁（无 API 调用）
```

### 运行与配置

```bash
python cn_doc_author.py
```

```python
API_KEYS    = ["key-1", "key-2", ...]   # Key 越多，并发越高
INPUT_FILE  = "similarity_results.xlsx"
OUTPUT_FILE = "similarity_results_author.xlsx"
```

运行时控制台会显示：

```
总行数: 500 | 不重复的路径数: 312 | 并发线程: 14
=> 实际最多打 312 次 API，重复路径会自动复用结果
```

### 输出新增列格式

每位作者生成独立的四列，依文档作者数量自动扩展：

| 列名示例 | 说明 |
|---|---|
| `Author_1_Name` | 第 1 位作者姓名 |
| `Author_1_Institution` | 第 1 位作者所属机构 |
| `Author_1_Code` | 第 1 位作者邮编 |
| `Author_1_Email` | 第 1 位作者邮箱 |
| `Author_N_*` | 依此类推，N = 文档作者总数 |

`Not Found` 表示 JSON 文件中未找到脚注内容，或 LLM 未能解析出作者信息。

---

## 附录：环境配置与依赖总览

### Python 基础依赖

```bash
pip install pandas openai tqdm rapidfuzz openpyxl
```

| 包名 | 用途 | 涉及脚本 |
|---|---|---|
| `pandas` | CSV/Excel 读写与数据处理 | 全部脚本 |
| `openai` | LLM API 客户端 | cn_classify / en_classify / similarity / geo_tagging / author |
| `tqdm` | 进度条显示 | en_classify / similarity / geo_tagging / author |
| `rapidfuzz` | 模糊标题匹配 | process_en |
| `openpyxl` | Excel 文件读写后端 | geo_tagging / author |

### 硬件与系统要求

| 组件 | 要求 |
|---|---|
| GPU | 4 × NVIDIA A800（或修改 `num_gpus` 适配其他配置） |
| CUDA 环境 | 需正常配置（`multi_gpu_process.py`） |
| MinerU | 命令行工具 `mineru` 需在 PATH 中（`multi_gpu_process.py`） |
| Python | 3.8+ |

### API 服务配置

| 参数 | 说明 |
|---|---|
| `BASE_URL` | LLM API 接入地址，在各脚本顶部配置 |
| `MODEL` | 使用的模型名称 |
| `API_KEYS` | 在各脚本顶部配置，建议配置 10 个以充分利用并发额度 |

---

## 附录：常见问题

| 问题 | 解决方案 |
|---|---|
| PDF 解析失败（`is_corrupted=TRUE`） | 将失败 PDF 转换为 PNG 后重新解析（见 `multi_gpu_process.py` 注意事项） |
| 英文 JSON 无匹配（`no_match`） | 检查 CSV 中 `Ref_Title` 格式；可手动修正标题后重跑 `process_en.py` |
| 分类结果全为 `ERROR` | 检查 API Key 是否有效；检查网络连接；查看控制台错误详情 |
| 相似性检测中断后重启 | 直接重新运行脚本，系统自动检测检查点并从断点恢复 |
| Excel 输出中 `cn_country` 为 `API Error` | 记录行号，核查对应 JSON 路径是否正确，或重新运行 `doc_geo_tagging.py` |
| 作者信息列为 `Not Found` | 该文档脚注格式可能不标准；可查看原始 JSON 中的 `page_footnote` 字段人工核对 |
| 中文分类结果出现 `NO_INTRO` | 对应 JSON 文件中未找到 `introductory-part` 字段，需检查 `process_cn.py` 的结构化处理结果 |

---

*文档覆盖脚本：`multi_gpu_process.py` · `clean_general.py` · `process_cn.py` · `process_en.py` · `cn_classify.py` · `en_classify.py` · `reduced_form_similarity.py` · `structural_similarity.py` · `doc_geo_tagging.py` · `cn_doc_author.py`*

---

## 附录：数据表格说明

### 输入数据：`all_ref_final.csv`

这是整个流水线的**初始文献表格**，记录了所有待分析的英文参考文献元信息，每行代表一篇中文文档所引用的一篇英文文献。

| 列名 | 说明 |
|---|---|
| `Ref_Title` | 英文参考文献标题，供 `process_en.py` 进行标题匹配定位 |
| `Author` | 英文文献作者 |
| `Journal` | 发表期刊 |
| `Year` | 发表年份，自动过滤 1980 年前文献 |
| `source` | 中文文档的原始文件路径（含期刊、年份、文档标题） |
| `title` | 引用该英文文献的中文文档标题 |
| `id` | 中文文档唯一编号，格式为 `年份_期号_文章序号`（如 `2008_01_1`） |

同一篇中文文档通常对应多行（每引用一篇英文文献即一行），`id` 相同的行均属于同一篇中文文档。

---

### 输出数据：`Similarity_Result_Cleaned.xlsx`

这是流水线**全流程处理完成后的最终数据表格**，在 `all_ref_final.csv` 的基础上依次追加了分类结果、相似性检测结果、地域信息与作者信息，每行代表一对中英文文档的完整分析记录。

**核心新增列概览：**

| 列名 | 来源步骤 | 说明 |
|---|---|---|
| `doc_id` | Step 01 | 英文文献 PDF 编号，用于定位 JSON 文件 |
| `cn_json_path` / `en_json_path` | Step 03/04 | 中英文文档结构化 JSON 的文件路径 |
| `has_data` / `has_model` | Step 05 | 中文文档是否使用真实数据 / 建立正式模型 |
| `doc_type` / `measurement` | Step 05 | 中文文档类型与是否为测量类 |
| `en_has_data` / `en_doc_type` 等 | Step 06 | 英文文档对应分类字段（前缀 `en_`） |
| `research_question` / `research_design` / `mechanism` / `counterfactual_analysis` | Step 07/08 | 三维度相似性判定结果 |
| `high_similarity` | Step 07/08 | 综合相似性判断（`TRUE` / `FALSE` / `ERROR` / `skipped`） |
| `most_likely_source` | Step 07/08 | 相似程度最高的主要英文来源（多候选时通过两两比对确定） |
| `round1_*` / `round2_*` / `consistency_*` | Step 07/08 | 双轮验证详情与一致性标记 |
| `cn_country` / `en_country` | Step 09 | 中英文文档数据来源地域 |
| `en_Author` | all_ref_final.csv | 英文文献作者 |
| `Author_N_Name` / `Author_N_Institution` / `Author_N_Code` / `Author_N_Email` | Step 10 | 中文文档第 N 位作者信息 |

**两个人工核查字段：**

| 列名 | 类型 | 说明 |
|---|---|---|
| `Author_Match` | 手工核查 | 对高相似度文档对，人工核查中英文文档作者是否存在重合。`TRUE` = 存在共同作者或明显关联，`FALSE` = 作者不一致。 |
| `Edition` | 手工核查 | 当一篇中文文档存在多个潜在英文来源时，人工核查这些英文来源是否为同一篇文章的不同版本。`Same` = 版本一致（如工作论文与正式发表版），`Different` = 不同文章。 |
