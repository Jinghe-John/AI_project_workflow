# 学术套利检测系统 · 端到端技术文档

> **Academic Arbitrage Detection System — End-to-End Pipeline Documentation**

---
**其他语言版本: [English](README.md), [中文](README_cn.md).**
## 目录

1. [系统概览](#系统概览)
2. [数据流向总览](#数据流向总览)
3. [Step 01 · 多GPU并行PDF解析 `multi_gpu_process.py`](#step-01--多gpu并行-pdf-解析)
4. [Step 02 · 通用JSON字段清洗 `clean_general.py`](#step-02--通用-json-字段清洗)
5. [Step 03 · 中文论文结构化处理 `process_cn.py`](#step-03--中文论文-json-结构化处理)
6. [Step 04 · 英文论文结构化处理 `process_en.py`](#step-04--英文论文-json-结构化处理)
7. [Step 05/06 · 论文分层自动分类 `cn_classify.py` / `en_classify.py`](#step-0506--论文分层自动分类)
8. [Step 07 · 简约式论文套利检测 `reduced_form_arbitrage.py`](#step-07--简约式论文套利检测)
9. [Step 08 · 结构论文套利检测 `structural_arbitrage.py`](#step-08--结构论文套利检测)
10. [Step 09 · 套利论文国别识别 `paper_nationality.py`](#step-09--套利论文国别识别)
11. [Step 10 · 套利论文作者提取 `cn_paper_author.py`](#step-10--套利论文作者信息提取)
12. [附录：环境配置与依赖](#附录环境配置与依赖总览)
13. [附录：常见问题](#附录常见问题)
14. [附录：数据表格说明](#附录数据表格说明)

---

## 系统概览

本系统是一套面向**结构经济学论文**的端到端学术套利自动检测流水线。系统从原始 PDF 出发，经过多 GPU 并行解析、JSON 清洗与结构化、基于大语言模型的自动分类，最终完成中英文论文对的套利行为识别，并提取套利论文的国别与作者信息。

整套流程由 **10 个 Python 脚本**按顺序协同完成，覆盖从数据预处理到结果后处理的全生命周期。

| 步骤 | 脚本文件 | 核心功能 | 输出 |
|:---:|---|---|---|
| 01 | `multi_gpu_process.py` | 多GPU并行PDF解析 | JSON 原始文件 |
| 02 | `clean_general.py` | 通用JSON字段清洗 | 精简JSON文件 |
| 03 | `process_cn.py` | 中文论文结构化处理 | 结构化中文JSON |
| 04 | `process_en.py` | 英文论文结构化处理 | 结构化英文JSON |
| 05 | `cn_classify.py` | 中文论文分层分类 | 分类标签CSV |
| 06 | `en_classify.py` | 英文论文分层分类 | 分类标签CSV |
| 07 | `reduced_form_arbitrage.py` | 简约式论文套利检测 | 套利结果CSV |
| 08 | `structural_arbitrage.py` | 结构论文套利检测 | 套利结果CSV |
| 09 | `paper_nationality.py` | 套利论文国别识别 | 国别标注Excel |
| 10 | `cn_paper_author.py` | 套利论文作者信息提取 | 作者信息Excel |

---

## 数据流向总览

```
原始 PDF 文件
  │
  ▼  [Step 01] multi_gpu_process.py
  │  多GPU并行，4× A800，以文件夹为调度单位，调用 mineru CLI
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
  │  调用 Kimi API，六层决策树分类
  │
分类标签 CSV（has_data / has_model / paper_type / measurement 等）
  │
  ├──▶  [Step 07] reduced_form_arbitrage.py → 仅处理 Pure Empirical 论文对
  │
  └──▶  [Step 08] structural_arbitrage.py   → 仅处理 Structural + Non-Measurement 论文对
  │
套利检测结果 CSV（arbitrage / research_question / research_design 等）
  │
  ├──▶  [Step 09] paper_nationality.py  →  提取中英文论文数据来源国别
  │
  └──▶  [Step 10] cn_paper_author.py    →  提取中文论文作者姓名/机构/邮编/邮箱
  │
最终 Excel（arbitrage=TRUE 条目 + 国别 + 作者信息）
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
input_root = '/data/.../en_unzip'   # PDF 根目录，脚本递归扫描所有子文件夹
output_dir = 'en_output'            # MinerU 输出目录
csv_file   = 'pdf_location_info.csv'# 处理结果记录文件
num_gpus   = 4                      # 使用 GPU 数量（默认 4× A800）
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
  │     └── 子进程循环取任务 → 调用 mineru CLI → 上报结果
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
{ "type": "text", "content": "这是正文", "bbox": [100,200,300,400], "score": 0.98, "page_id": 1 }

// 清洗后（输出）
{ "type": "text", "content": "这是正文" }
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

## Step 03 · 中文论文 JSON 结构化处理

**脚本：** `process_cn.py`

对清洗后的中文 JSON 文件进行语义重组：按章节合并内容，识别并标记无中文字符的异常文件，生成结构统一的学术 JSON。

### 设计思路

将原始扁平列表转换为**以章节为单位、语义清晰的结构化 JSON**，过滤页眉、元数据等噪声，同时检测文件是否含有中文字符，自动标记可能误入英文内容区的异常文件。

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

若文档开头存在两个连续 `title`（中间无正文），自动将其合并为一个，避免论文标题被错误拆分：

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
  { "type": "Title",             "content": "X对Y的影响：基于Z的证据" },
  { "type": "Introductory-Part", "content": "摘要：本文... 关键词：... 1.引言..." },
  { "type": "title",             "content": "2 数据" },
  { "type": "text",              "content": "本文的数据来自XXX数据库..." }，
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

## Step 04 · 英文论文 JSON 结构化处理

**脚本：** `process_en.py`

结合 CSV 元信息与 JSON 内容，通过**三级标题匹配策略**定位论文正文起点，裁剪无关内容，生成结构统一的英文学术 JSON。

### 设计思路

英文处理比中文复杂：同一 PDF 可能包含多篇论文的拼接内容。需借助 CSV 中的参考标题（`Ref_Title`）精准定位当前论文的起始位置，然后截断其他内容。三级匹配策略应对标题格式差异大的挑战。

### 总体流程

```
读取 CSV → 过滤 1980 年前文献
  ↓
定位 JSON 文件（按 pdf_id 年份前4位找子目录）
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
python process_en.py -i all_pdf_info.csv -j ./en_output -o ./en_processed_json
python process_en.py -i all_pdf_info.csv -j ./en_output -o ./en_processed_json -c ./report.csv
```

### 输入 CSV 必要列

| 列名 | 说明 |
|---|---|
| `pdf_id` | 文献ID，前4位为年份，用于定位 JSON 子目录 |
| `Year` | 发表年份，自动过滤 1980 年前文献 |
| `Ref_Title` | 参考标题，用于与 JSON 内部 title 进行匹配 |

---

## Step 05/06 · 论文分层自动分类

**脚本：** `cn_classify.py`（中文版）/ `en_classify.py`（英文并行版）

调用 Kimi API，通过**六层决策树**对论文进行自动分类。分类依据为 JSON 文件中的 `introductory-part` 字段（标题 + 摘要 + 引言）。

### 两个脚本的区别

| 维度 | `cn_classify.py`（中文版） | `en_classify.py`（英文版） |
|---|---|---|
| 处理对象 | 中文论文全量 | 中文论文中含英文引用的子集 |
| JSON 路径列 | `cn_json_path` | `en_json_path` |
| 标题列 | `json_title` | `Ref_Title` |
| 结果列前缀 | `has_data` / `paper_type` 等 | `en_has_data` / `en_paper_type` 等 |
| Prompt 语言 | 中文 | 英文） |

### 六层分类决策树

```
第一层：has_data & has_model（是否使用真实数据 / 是否建立正式模型）
  │
  ├─ 有数据 & 有模型
  │     ↓
  │   第二层：is_structural（是否含反事实分析或校准）
  │     ├─ Yes → paper_type = Structural → 第六层：measurement 判断
  │     └─ No  → 第三层：Non-Structural 细分
  │                 ↓
  │             Pure Empirical / Pure Theory / Other
  │             （Pure Empirical & Other 进入第四、五层）
  │
  ├─ 有数据 & 无模型 → paper_type = Pure Empirical → 第四、五层
  ├─ 无数据 & 有模型 → paper_type = Pure Theory（结束）
  └─ 无数据 & 无模型 → paper_type = Other → 第四、五层

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
| `Structural` | 论文包含**反事实分析**或**校准** |
| `Non-Structural` | 论文不包含反事实分析和校准 |

**第五层：解释变量数量**（基于论文标题结构判断）

| 类型 | 定义 |
|---|---|
| `Single Explanatory` | 标题明确列出所有核心概念，构成封闭分析框架（如"X 对 Y 的影响"） |
| `Multiple Explanators` | 标题使用开放式探索性词语（如"决定因素"、"Determinants"、"Factors"） |

### 断点续传机制

- **`cn_classify.py`**：每处理10条保存 `temp_classification_result.csv`，下次启动自动从断点恢复；全部完成后生成带时间戳最终文件
- **`en_classify.py`**：每处理5条保存 `classification_checkpoint_v4.pkl`，每20条额外保存中间 CSV；已有有效记录的行（非 `ERROR`/`FILE_NOT_FOUND`）自动跳过

### 英文版并行架构（`en_classify.py`）

10 个 API Key 对应 10 个 Worker，通过 `ThreadPoolExecutor` 并发提交任务，每个 Worker 独占一个 Client 实例互不干扰。

### 关键配置

```python
# cn_classify.py
client_kimi = OpenAI(api_key="sk-xxx", base_url="https://api.moonshot.cn/v1")
CSV_PATH  = os.path.join(BASE_DIR, "all_Structural_unclassified.csv")

# en_classify.py
API_KEYS   = ["sk-xxx", "sk-yyy", ...]   # Key 数量决定并发线程数
INPUT_FILE = "all_Structural.csv"
```

### 输出新增列

| 列名 | 说明 |
|---|---|
| `has_data` | `Yes` / `No` / `ERROR` |
| `has_model` | `Yes` / `No` / `ERROR` |
| `paper_type` | `Structural` / `Pure Empirical` / `Pure Theory` / `Other` / `ERROR` / `FILE_NOT_FOUND` / `NO_INTRO` |
| `research_article` | `Research Article` / `Non-Research Article` / 空（不适用） |
| `variable` | `Single Explanatory` / `Multiple Explanators` / 空（不适用） |
| `measurement` | `Measurement Article` / `Non-Measurement Article` / 空（不适用） |

英文版列名统一加 `en_` 前缀，如 `en_has_data`、`en_paper_type` 等。

---

## Step 07 · 简约式论文套利检测

**脚本：** `reduced_form_arbitrage.py`

对简约式（Reduced-Form / Pure Empirical）论文中英文对进行**多维度、双轮交叉验证**，识别学术套利行为。这是整个流水线的**核心分析模块**，也是数量最多的论文类型，因此优先处理。

### 什么是学术套利

> 将英文论文的识别策略、实证设计与研究问题套用于中国情境，而未做实质性的方法论创新——即直接沿用英文论文的因果推断框架，仅将研究对象替换为中国数据或中国制度背景。

简约式论文的套利判定聚焦于**识别策略的原创性**：中文论文是否独立提出了新的外生变量、自然实验或因果识别方案，还是仅在英文论文现成框架上做数据替换。

### 适用范围（四列同时满足才进入分析）

| 列名 | 条件 |
|---|---|
| `paper_type` | 必须为 `Pure Empirical` |
| `en_paper_type` | 必须为 `Pure Empirical` |
| `research_article` | 必须为 `Research Article` |
| `en_research_article` | 必须为 `Research Article` |

不满足任一条件的行将被标记为 `skipped` 并记录跳过原因。

### 三维度套利判定框架

**三个维度同时满足才判定为套利；任一维度为 `False` 或 `ERROR`，均不判定为套利。**

| 维度 | 简约式论文专项判定标准 |
|---|---|
| **研究问题 (RQ)** | 中英文论文研究的是同一个因果问题：去掉"中国"标签后，核心的被解释变量与解释变量在经济含义上完全等价，仅数据来源不同 |
| **研究设计 (RD)** | 采用相同的识别策略：工具变量来源相同、自然实验逻辑相同，或 DID/RDD 设计的处理变量与对照组构造逻辑相同，中文论文未提出独立的外生识别方案 |
| **机制 (Mech)** | 机制解释与英文论文功能等价，仅将政策背景替换为中国对应物，传导路径判断实质相同 |

### 套利与非套利典型案例

| 情形 | 判定 | 原因 |
|---|:---:|---|
| 英文用移民潮作 IV 估计劳动力供给冲击；中文用同类迁移变量估计中国城镇化影响 | ✅ 套利 | IV 逻辑与识别来源实质相同 |
| 英文用边界断点识别政策效应；中文用完全不同的政策时间节点做 DID | ❌ 非套利 | 识别策略独立设计，外生性来源不同 |
| 中文论文的核心解释变量来自中国特有的制度变量（无英文对应物） | ❌ 非套利 | 研究问题本身无法在英文论文中找到功能等价的映射 |

### 双轮交叉验证机制

对每对论文调用 API **两次**（Round 1 + Round 2），取两轮结果的**交集**：只有两轮均判为 `True` 的维度，最终才计为 `True`。显著提升判断可靠性，减少 LLM 随机性导致的误判。

### 英文论文间套利检测

当一篇中文论文匹配到多个英文候选来源时，系统自动对这些英文论文两两进行套利关系检测，并通过 `most_likely_source` 字段标识最可能的主要套利来源。英文间套利汇总结果另存为独立 JSON 文件：

```
{原文件名}_arbitrage_results_{时间戳}_en_en_arbitrage_summary.json
```

### 运行与配置

```bash
python reduced_form_arbitrage.py
```

```python
BASE_URL   = "https://api.moonshot.cn/v1"
MODEL      = "kimi-k2-0711-preview"
API_KEYS   = ["sk-xxx", "sk-yyy", ...]   # Key 数量决定并发线程数
input_path = r"C:\path\to\your\input.csv"
```

### 输出新增列

| 列名 | 说明 |
|---|---|
| `research_question` | 研究问题维度：`True` / `False` / `ERROR` |
| `research_design` | 研究设计维度：`True` / `False` / `ERROR` |
| `counterfactual_analysis` | 反事实分析维度：`True` / `False` / `ERROR` |
| `arbitrage` | 综合套利判断（三维度均 True 才为 True） |
| `most_likely_source` | 是否为最可能的主要套利来源 |
| `round1_rq` / `round2_rq` | 第一/二轮研究问题判断 |
| `round1_rd` / `round2_rd` | 第一/二轮研究设计判断 |
| `round1_mech` / `round2_mech` | 第一/二轮反事实分析判断 |
| `consistency_rq/rd/mech` | 两轮一致性（`consistent` / `inconsistent` / `error`） |
| `en_en_arbitrage_detected` | 是否检测到英文论文间套利 |
| `en_en_arbitrage_pairs` | 存在套利关系的英文论文对 |
| `skip_reason` | 跳过原因（如 `en_paper_type 为非Pure Empirical`） |

### 断点续传

运行过程中生成 `*_checkpoint.json`。中断后直接重新运行脚本，系统自动检测并从断点恢复，跳过已处理行。

---

## Step 08 · 结构论文套利检测

**脚本：** `structural_arbitrage.py`

对结构论文中英文对进行套利检测，逻辑框架与 Step 07 一致，但针对结构模型论文的方法论特点重新设计了判定标准与 Prompt。

### 与简约式检测模块的区别

结构论文的套利隐蔽性更强：同一套方程组和矩匹配策略可直接移植至不同国家数据，在形式上看似独立研究，但在方法论层面实质等价。因此判定标准从"识别策略是否相同"转向"**模型骨架与反事实实验是否功能等价**"。

### 适用范围（四列同时满足才进入分析）

| 列名 | 条件 |
|---|---|
| `paper_type` | 必须为 `Structural` |
| `en_paper_type` | 必须为 `Structural` |
| `measurement` | 必须为 `Non-Measurement Article` |
| `en_measurement` | 必须为 `Non-Measurement Article` |

### 三维度套利判定框架

| 维度 | 结构论文专项判定标准 |
|---|---|
| **研究问题 (RQ)** | 通过"机制映射测试"：去掉中国制度标签（户籍、补贴、国企）后，因果问题是否与英文论文完全相同 |
| **研究设计 (RD)** | 采用相同的模型骨架（偏好、技术、约束的函数形式）与相同的识别策略（如相同的 SMM/GMM 矩匹配目标） |
| **反事实分析 (CF)** | 政策模拟冲击针对同一经济边际，传导机制与福利含义功能等价，仅将政策标签替换为中国对应物 |

### 运行与配置

```bash
python structural_arbitrage.py
```

```python
BASE_URL   = "https://api.moonshot.cn/v1"
MODEL      = "kimi-k2-0711-preview"
API_KEYS   = ["sk-xxx", "sk-yyy", ...]
input_path = r"C:\path\to\your\input.csv"
```

输出列结构与 Step 07 完全相同，断点续传机制亦一致，此处不再赘述。

---

## Step 09 · 套利论文国别识别

**脚本：** `paper_nationality.py`

从套利检测结果中，自动提取中英文论文**数据来源国别**信息，回写至 Excel。

### 设计思路

本脚本是套利检测的**下游后处理模块**，仅处理 `arbitrage == TRUE` 的条目，节省 API 调用开销。通过调用 Kimi API 分析论文 `introductory-part`，从中识别数据来源国别，用于后续分析"中文论文将哪些国家的英文研究搬运至中国情境"。

### 国别标准化规则

内置映射规则确保输出标准统一：

| 原始识别 | 标准化输出 |
|---|---|
| `United States` / `U.S.` / `America` | `USA` |
| `United Kingdom` / `Great Britain` | `UK` |
| `P.R.C` / `Mainland China` | `China` |
| OECD / EU / Global / Cross-country 多国数据 | `Multiple Countries` |

### 运行与配置

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

每处理 10 条自动保存一次，防止意外中断丢失结果。

### 输出新增列

| 新增列 | 说明 | 示例值 |
|---|---|---|
| `cn_country` | 中文论文的数据来源国 | `China` |
| `en_country` | 英文论文的数据来源国 | `USA` / `Multiple Countries` |

### 特殊值说明

| 值 | 含义 |
|---|---|
| `Multiple Countries` | 论文使用多国数据（含 OECD、EU、全球样本等） |
| `Content Missing` | JSON 文件中未找到 `introductory-part` 内容 |
| `No Content` | JSON 路径为空或文件不存在 |
| `API Error` | API 调用失败，需人工核查 |

---

## Step 10 · 套利论文作者信息提取

**脚本：** `cn_paper_author.py`

从中文论文 JSON 的脚注（`page_footnote`）中提取作者姓名、机构、邮编与邮箱，支持**多Key并行**与**路径级去重缓存**。

### 设计思路

本脚本处理**所有行**（不按 `arbitrage` 值筛选），用于后续学术网络分析与机构分布统计。核心设计亮点是路径级去重缓存机制，大幅降低 API 调用成本。

### 核心设计：路径级去重缓存

一份中文论文可能在套利结果表中出现多次（对应多篇英文论文）。本脚本通过 **per-path 锁机制**确保每个路径全程只调用一次 API，后续行直接复用缓存结果：

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
python cn_paper_author.py
```

```python
API_KEYS    = ["sk-xxx", "sk-yyy", ...]   # Key 越多，并发越高
INPUT_FILE  = "Arbitrage_results.xlsx"
OUTPUT_FILE = "Arbitrage_results_author.xlsx"
```

运行时控制台会显示：

```
总行数: 500 | 不重复的路径数: 312 | 并发线程: 14
=> 实际最多打 312 次 API，重复路径会自动复用结果
```

### 输出新增列格式

每位作者生成独立的四列，依论文作者数量自动扩展：

| 列名示例 | 说明 |
|---|---|
| `Author_1_Name` | 第 1 位作者姓名 |
| `Author_1_Institution` | 第 1 位作者所属机构 |
| `Author_1_Code` | 第 1 位作者邮编 |
| `Author_1_Email` | 第 1 位作者邮箱 |
| `Author_N_*` | 依此类推，N = 论文作者总数 |

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
| `openai` | Kimi API 客户端 | cn_classify / en_classify / arbitrage / nationality / author |
| `tqdm` | 进度条显示 | en_classify / arbitrage / nationality / author |
| `rapidfuzz` | 模糊标题匹配 | process_en |
| `openpyxl` | Excel 文件读写后端 | nationality / author |

### 硬件与系统要求

| 组件 | 要求 |
|---|---|
| GPU | 4 × NVIDIA A800（或修改 `num_gpus` 适配其他配置） |
| CUDA 环境 | 需正常配置（`multi_gpu_process.py`） |
| MinerU | 命令行工具 `mineru` 需在 PATH 中（`multi_gpu_process.py`） |
| Python | 3.8+ |

### API 服务配置

| 参数 | 默认值 |
|---|---|
| `BASE_URL` | `https://api.moonshot.cn/v1` |
| `MODEL` | `kimi-k2-0711-preview` |
| `API_KEYS` | 在各脚本顶部配置，建议配置 10 个以充分利用并发额度 |

---

## 附录：常见问题

| 问题 | 解决方案 |
|---|---|
| PDF 解析失败（`is_corrupted=TRUE`） | 将失败 PDF 转换为 PNG 后重新解析（见 `multi_gpu_process.py` 注意事项） |
| 英文 JSON 无匹配（`no_match`） | 检查 CSV 中 `Ref_Title` 格式；可手动修正标题后重跑 `process_en.py` |
| 分类结果全为 `ERROR` | 检查 API Key 是否有效；检查网络连接；查看控制台错误详情 |
| 套利检测中断后重启 | 直接重新运行脚本，系统自动检测检查点并从断点恢复 |
| Excel 输出中 `cn_country` 为 `API Error` | 记录行号，核查对应 JSON 路径是否正确，或重新运行 `paper_nationality.py` |
| 作者信息列为 `Not Found` | 该论文脚注格式可能不标准；可查看原始 JSON 中的 `page_footnote` 字段人工核对 |
| 中文分类结果出现 `NO_INTRO` | 对应 JSON 文件中未找到 `introductory-part` 字段，需检查 `process_cn.py` 的结构化处理结果 |

---

*文档覆盖脚本：`multi_gpu_process.py` · `clean_general.py` · `process_cn.py` · `process_en.py` · `cn_classify.py` · `en_classify.py` · `reduced_form_arbitrage.py` · `structural_arbitrage.py` · `paper_nationality.py` · `cn_paper_author.py`*

---

## 附录：数据表格说明

### 输入数据：`all_ref_final.csv`

这是整个流水线的**初始文献表格**，记录了所有待分析的英文参考文献元信息，每行代表一篇中文论文所引用的一篇英文文献。

| 列名 | 说明 |
|---|---|
| `Ref_Title` | 英文参考文献标题，供 `process_en.py` 进行标题匹配定位 |
| `Author` | 英文文献作者 |
| `Journal` | 发表期刊 |
| `Year` | 发表年份，用于过滤 1980 年前文献 |
| `source` | 中文论文的原始文件路径（含期刊、年份、论文标题） |
| `title` | 引用该英文文献的中文论文标题 |
| `id` | 中文论文唯一编号，格式为 `年份_期号_文章序号`（如 `2008_01_1`） |

同一篇中文论文通常对应多行（每引用一篇英文文献即一行），`id` 相同的行均属于同一篇中文论文。

---

## 附录：数据表格说明

### 输入数据：`all_ref_final.csv`

这是整个流水线的**初始文献表格**，记录了所有待分析的英文参考文献元信息，每行代表一篇中文论文所引用的一篇英文文献。

| 列名 | 说明 |
|---|---|
| `Ref_Title` | 英文参考文献标题，供 `process_en.py` 进行标题匹配定位 |
| `Author` | 英文文献作者 |
| `Journal` | 发表期刊 |
| `Year` | 发表年份，用于过滤 1980 年前文献 |
| `source` | 中文论文的原始文件路径（含期刊、年份、论文标题） |
| `title` | 引用该英文文献的中文论文标题 |
| `id` | 中文论文唯一编号，格式为 `年份_期号_文章序号`（如 `2008_01_1`） |

同一篇中文论文通常对应多行（每引用一篇英文文献即一行），`id` 相同的行均属于同一篇中文论文。

---

### 输出数据：`Arbitrage_Result_Cleaned.xlsx`

这是流水线**全流程处理完成后的最终数据表格**，在 `all_ref_final.csv` 的基础上依次追加了分类结果、套利检测结果、国别信息与作者信息，每行代表一对中英文论文的完整分析记录。

**核心新增列概览：**

| 列名 | 来源步骤 | 说明 |
|---|---|---|
| `pdf_id` | Step 01 | 英文文献 PDF 编号，用于定位 JSON 文件 |
| `cn_json_path` / `en_json_path` | Step 03/04 | 中英文论文结构化 JSON 的文件路径 |
| `has_data` / `has_model` | Step 05 | 中文论文是否使用真实数据 / 建立正式模型 |
| `paper_type` / `measurement` | Step 05 | 中文论文类型与是否为测量类 |
| `en_has_data` / `en_paper_type` 等 | Step 06 | 英文论文对应分类字段（前缀 `en_`） |
| `research_question` / `research_design` /`mechanism` /  `counterfactual_analysis` | Step 07/08 | 三维度套利判定结果 |
| `arbitrage` | Step 07/08 | 综合套利判断（`TRUE` / `FALSE` / `ERROR` / `skipped`） |
| `most_likely_source` | Step 07/08 | 如果存在有多个潜在套利对象，则通过两两对比的方式选出相似程度最高的套利源 |
| `round1_*` / `round2_*` / `consistency_*` | Step 07/08 | 双轮验证详情与一致性标记 |
| `cn_country` / `en_country` | Step 09 | 中英文论文数据来源国别 |
| `en_Author` | all_ref_final.csv | 英文文献作者 |
| `Author_N_Name` / `Author_N_Institution` / `Author_N_Code` / `Author_N_Email` | Step 10 | 中文论文第 N 位作者信息 |

**两个人工核查字段：**

| 列名 | 类型 | 说明 |
|---|---|---|
| `Author_Match` | 手工核查 | 对套利检测为 `TRUE` 的论文对，人工核查中文论文作者与英文论文作者是否存在重合。`TRUE` = 作者一致（存在共同作者或明显关联），`FALSE` = 作者不一致。用于辅助判断套利行为是否属于作者自我复用。 |
| `Edition` | 手工核查 | 当一篇中文论文存在多个潜在英文套利来源时，人工核查这些英文来源是否为同一篇文章的不同版本。`Same` = 版本一致（如工作论文与正式发表版），`Different` = 版本不一致。用于区分"针对同一英文研究的重复匹配"与"同时借鉴多篇不同英文文献"两种情形。 |

