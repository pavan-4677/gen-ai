# Intelligent Dataset Diagnosis and Quality Analysis System

**B.Tech Final Year Artificial Intelligence & Machine Learning (AI & ML) Capstone Project**  
*An End-to-End Enterprise-Grade Web Application for Automated Data Profiling, IQR Anomaly Detection, Explainable Quality Scoring, Generative AI Remediation, and Compliance Reporting.*

---

## 1. Project Abstract

In high-stakes machine learning and predictive analytics pipelines, data quality degradation—manifesting as missing entries, duplicate records, unstandardized text, semantic anomalies, and extreme statistical outliers—frequently compromises model convergence, inflates generalization error, and leads to data leakage. Traditional data profiling tools are often purely diagnostic, producing static summaries without actionable remediation strategies or natural-language explanations.

The **Intelligent Dataset Diagnosis and Quality Analysis System** bridges this gap by combining classical statistical profiling with state-of-the-art Generative AI. Built on a modular Python architecture using **Streamlit**, **Pandas**, **SciPy**, **Scikit-learn**, and **Plotly**, the platform ingests heterogeneous datasets (`.csv`, `.tsv`, `.xlsx`, `.xls`), performs an automated 10-point structural and statistical diagnosis, calculates an explainable **4-Dimensional Data Quality Score (Completeness, Uniqueness, Validity, Consistency)**, and leverages a context-grounded **Generative AI Assistant** (powered by OpenAI or an offline deterministic fallback engine) to explain detected anomalies and recommend optimal preprocessing actions. Furthermore, an interactive **Data Cleaning Studio** empowers practitioners to perform non-destructive imputation, IQR-based Winsorization, deduplication, and schema conversions with an audit trail, before-and-after score diffing, and export to **PDF**, **JSON**, and cleaned **CSV/Excel** formats.

---

## 2. Problem Statement

Modern data science and machine learning workflows face several critical data quality challenges:

1. **Silent Pipeline Failures**: Raw datasets collected from webhooks, IoT devices, or manual forms routinely contain hidden dirty data (e.g., negative ages, sentinel values like `?` or `NULL` disguised as strings, untrimmed whitespace, and duplicate events).
2. **Lack of Explainability in Profiling**: Conventional libraries (such as raw pandas `.describe()`) output dense statistical tables without explaining *why* certain values constitute critical problems or *how* they impact downstream model families (e.g., tree-based vs. gradient-based models).
3. **Risk of Hallucination in Generic AI**: Naive application of LLMs to tabular data either fails due to context token limits when raw datasets are fed, or hallucinates fictitious statistics.
4. **Destructive or Inflexible Cleaning**: Standard cleaning scripts often mutate original datasets destructively, lack granular undo capabilities, and fail to track changes for data governance and auditability.

---

## 3. Project Objectives

- **Automated Multi-Format Ingestion**: Ingest CSV, TSV, and Excel files with automatic encoding sniffing (UTF-8, Latin-1, CP1252), delimiter detection, and file size validation.
- **10-Point Comprehensive Diagnosis**:
  1. Missing Value & Sparsity Detection
  2. Duplicate Record Detection (exact and subset keys)
  3. Data Type & Schema Mismatch Inference
  4. Statistical Outlier Detection using the Interquartile Range (IQR 1.5x) method
  5. Sentinel / Placeholder String Token Detection (`?`, `null`, `unknown`, `-999`)
  6. Domain Constraint Validation (e.g., impossible negative ages or salaries)
  7. Numerical Distribution Analysis (Skewness, Kurtosis, Zero counts)
  8. Categorical Distribution & Cardinality Profiling
  9. Constant (Zero Variance) Column Detection
  10. Text Consistency & Capitalization Standardization Checks
- **Transparent 4D Quality Scoring Engine**: Formulate an explainable 0–100 quality score spanning Completeness, Uniqueness, Validity, and Consistency with user-configurable weights.
- **Context-Grounded Generative AI Assistant**: Integrate LLM reasoning using structured metadata summaries (no raw data leakage) with prompt-injection defense and an offline fallback engine.
- **Non-Destructive Data Cleaning Studio**: Provide interactive imputation, Winsorization (capping), outlier removal, casing standardization, and schema conversion with undo history and diff tracking.
- **Multi-Format Audit Export**: Generate publication-ready PDF audit reports, machine-readable JSON files, and cleaned CSV/Excel files.

---

## 4. Existing System vs. Proposed System

| Feature / Dimension | Existing System (Pandas / Basic Profilers) | Proposed Intelligent Diagnosis System |
| :--- | :--- | :--- |
| **Ingestion** | Manual coding required per file format/encoding | Automated format, encoding, and delimiter detection |
| **Outlier Detection** | Manual calculation of statistical bounds | Automated IQR 1.5x with interactive row inspection |
| **Quality Scoring** | Ad-hoc or absent; no standardized score | Transparent 4D formula (0–100 scale) with dimension breakdown |
| **AI Integration** | None (purely static summary tables) | Context-grounded Generative AI + interactive chat assistant |
| **Offline Reliability** | N/A | Deterministic fallback engine generates full insights without API keys |
| **Data Remediation** | Hardcoded scripts; destructive mutation | Interactive Studio with non-destructive state, undo, & diffs |
| **Audit Reporting** | Raw console prints or basic HTML | Executive PDF Audit Report, JSON metadata, Cleaned CSV/Excel |
| **UI / UX** | Default CLI or plain tabular notebooks | Dark charcoal / navy SaaS UI with interactive Plotly charts |

---

## 5. System Architecture & Processing Pipeline

```
                                 ┌───────────────────────────┐
                                 │   User Upload (CSV/XLSX)  │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │    DataLoader Module      │
                                 │  - Encoding Sniffing      │
                                 │  - Extension Validation   │
                                 │  - Size & Schema Guards   │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │  QualityAnalyzer Module   │
                                 │  - 10 Diagnostic Checks   │
                                 │  - IQR Outlier Detector   │
                                 │  - DataProfiler Module    │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │   QualityScoringEngine    │
                                 │  - Completeness Score     │
                                 │  - Uniqueness Score       │
                                 │  - Validity Score         │
                                 │  - Consistency Score      │
                                 └─────────────┬─────────────┘
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         │                                           │
                         ▼                                           ▼
          ┌─────────────────────────────┐             ┌─────────────────────────────┐
          │     Plotly Dashboard        │             │      AIService Module       │
          │  - Score Gauge & Cards      │             │  - Structured Summarizer    │
          │  - Missing Values Bar Chart │             │  - Prompt Injection Defense │
          │  - IQR Outliers Bar Chart   │             │  - OpenAI API / Compatible  │
          │  - Type Composition Pie     │             │  - Offline Fallback Engine  │
          │  - Correlation Heatmap      │             │  - Interactive Q&A Chatbot  │
          └──────────────┬──────────────┘             └──────────────┬──────────────┘
                         │                                           │
                         └─────────────────────┬─────────────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │   Data Cleaning Studio    │
                                 │  - Non-Destructive State  │
                                 │  - Imputation & Capping   │
                                 │  - Deduplication & Types  │
                                 │  - Undo & Audit History   │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │   ReportGenerator Module  │
                                 │  - PDF Audit Document     │
                                 │  - JSON Structured Audit  │
                                 │  - Cleaned CSV & Excel    │
                                 └───────────────────────────┘
```

---

## 6. Technology Stack

- **Frontend & Interface**: [Streamlit](https://streamlit.io/) (v1.66+), Custom CSS (Dark Charcoal & Slate Blue theme), [Plotly](https://plotly.com/python/) (v7.1+).
- **Data Processing & Analytics**: [Pandas](https://pandas.pydata.org/) (v3.1+), [NumPy](https://numpy.org/) (v2.5+), [SciPy](https://scipy.org/) (v1.18+), [Scikit-learn](https://scikit-learn.org/) (v1.9+), [OpenPyXL](https://openpyxl.readthedocs.io/) (v3.1+).
- **Generative AI & LLM**: OpenAI API / OpenAI-Compatible endpoints (`gpt-4o-mini`, `gpt-4o`, custom base URLs) via [Requests](https://requests.readthedocs.io/) with structured JSON schemas and deterministic offline fallback.
- **Reporting & Compliance**: [ReportLab](https://www.reportlab.com/) (v5.1+) for vector PDF generation, native JSON serialization, and UTF-8 CSV exports.
- **Configuration & Environment**: [python-dotenv](https://github.com/theskumar/python-dotenv) for secret management.
- **Quality Assurance**: [Pytest](https://docs.pytest.org/) (v9.2+) for unit and integration testing.

---

## 7. Project Folder Structure

```text
intelligent-dataset-diagnosis/
│
├── app.py                      # Main entry point and router
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation
├── .env.example                # Sample environment variables template
├── .gitignore                  # Git ignore rules
│
├── pages/
│   ├── 1_Home.py               # Overview, features, and sample benchmark loader
│   ├── 2_Upload_Dataset.py     # Ingestion, validation, preview, and diagnosis trigger
│   ├── 3_Dataset_Diagnosis.py  # 10-point diagnostic checks with record inspection
│   ├── 4_Quality_Dashboard.py  # Interactive Plotly analytics and quality gauge
│   ├── 5_GenAI_Assistant.py    # LLM report synthesis and interactive dataset Q&A chat
│   ├── 6_Data_Cleaning_Studio.py # Interactive non-destructive remediation with undo
│   ├── 7_Quality_Report.py     # Before/After comparison and PDF/JSON/CSV downloads
│   └── 8_Settings.py           # API keys, scoring weights, IQR thresholds, and reset
│
├── src/
│   ├── __init__.py             # Package marker
│   ├── utils.py                # Styling injection, session state, UI card helpers
│   ├── data_loader.py          # Secure ingestion, encoding sniffer, metadata calculation
│   ├── data_profiler.py        # Numerical, categorical, and correlation profiler
│   ├── outlier_detector.py     # IQR 1.5x and Z-score outlier detection algorithms
│   ├── quality_scoring.py      # Explainable 4D quality scoring engine (0-100)
│   ├── quality_analyzer.py     # Master diagnostic orchestrator
│   ├── data_cleaner.py         # Non-destructive cleaning engine with audit logging
│   ├── ai_service.py           # LLM API caller, prompt sanitizer, & offline fallback engine
│   └── report_generator.py     # PDF generation, JSON exporter, and Excel/CSV writers
│
├── assets/
│   ├── styles.css              # Custom SaaS dashboard stylesheet (dark navy theme)
│   └── logo.png                # System emblem / branded logo
│
├── sample_data/
│   ├── sample_dataset.csv      # Customer demographics benchmark (dirty test data)
│   └── customer_churn_dirty.csv # Telecom churn quality benchmark
│
└── tests/
    ├── __init__.py             # Test package marker
    ├── test_data_loader.py     # Tests for CSV/Excel ingestion, empty files, size limits
    ├── test_outlier_detector.py # Tests for IQR mathematics and row indexing
    ├── test_quality_scoring.py # Tests for 4D scoring dimensions and custom weights
    ├── test_quality_analyzer.py # Tests for the complete 10-point analysis pipeline
    ├── test_data_cleaner.py    # Tests for imputation, deduplication, capping, and casting
    ├── test_ai_service.py       # Tests for structured prompt synthesis, fallback, and Q&A
    └── test_report_generator.py # Tests for PDF generation and JSON export
```

---

## 8. Data Quality Scoring Methodology

The quality engine calculates an explainable score on a continuous **0–100 scale** using four fundamental dimensions:

$$\text{Overall Quality Score} = w_c \cdot C + w_u \cdot U + w_v \cdot V + w_s \cdot S$$

Where:
- **$C$ (Completeness)**:
  $$C = 100 \times \left(1 - \frac{\text{Missing Cells}}{\text{Total Cells}}\right)$$
- **$U$ (Uniqueness)**:
  $$U = 100 \times \left(1 - \frac{\text{Duplicate Rows}}{\text{Total Rows}}\right)$$
- **$V$ (Validity)**:
  Evaluates domain constraint compliance (negative values in strictly non-negative columns such as age/salary) and identifies string placeholder tokens (`?`, `na`, `null`, `unknown`, `-999`).
  $$V = 100 \times \left(1 - \frac{2 \times \text{Invalid Cells}}{\text{Total Cells}}\right) \quad (\text{clamped to } [0, 100])$$
- **$S$ (Consistency)**:
  Evaluates text casing variance (e.g. `'Male'` vs `'male'`), untrimmed leading/trailing whitespace, and statistical outliers exceeding IQR boundaries.
  $$S = 100 \times \left(1 - \frac{1.5 \times \text{Inconsistency Penalty Points}}{\text{Total Cells}}\right) \quad (\text{clamped to } [0, 100])$$

### Grade Classification
- **$\ge 90$**: Grade **A (Excellent)** — Ready for modeling and production inference.
- **$75 - 89$**: Grade **B (Good)** — Minor hygiene cleaning recommended.
- **$50 - 74$**: Grade **C (Fair)** — Moderate cleaning required to prevent bias.
- **$< 50$**: Grade **D (Poor)** — Critical data anomalies; high risk of model failure.

*Limitations Disclaimer: This score represents an automated statistical heuristic. It complements, but does not replace, organizational domain governance or semantic truth verification.*

---

## 9. Installation & Setup Guide

### Prerequisites
- Python 3.10 to Python 3.14 installed on your workstation.
- Windows PowerShell, Command Prompt, or Linux/macOS terminal.

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd "c:\Users\pavan\OneDrive\Desktop\gen ai project"
```

### Step 2: Create and Activate a Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(On Linux/macOS: `source .venv/bin/activate`)*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional for Live LLM)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Edit `.env` and add your OpenAI API key:
```ini
OPENAI_API_KEY=sk-proj-your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
```
> **Note on Offline Mode:** If you do not provide an API key, the application **automatically uses its internal Deterministic Diagnostic Engine**. All statistical profiling, IQR detection, interactive charts, data cleaning, and PDF report downloads function 100% offline!

---

## 10. How to Run the Application

Execute the Streamlit application using PowerShell:
```powershell
streamlit run app.py
```
Or directly using Python module syntax:
```powershell
python -m streamlit run app.py
```

The system will automatically open your default browser at:
`http://localhost:8501`

---

## 11. Step-by-Step User Walkthrough

1. **Home / Overview (`1_Home.py`)**:
   - Review the system architecture and processing workflow.
   - Click **"Load Customer Demographics Sample Dataset (Instant Demo)"** to instantly load test data into memory.
2. **Dataset Upload (`2_Upload_Dataset.py`)**:
   - Drag and drop your own CSV, TSV, or XLSX spreadsheet, or pick a sample dataset.
   - Verify file metadata (rows, columns, file size, memory footprint).
   - Click **"Start Full Dataset Diagnosis & Quality Analysis"**.
3. **Dataset Diagnosis (`3_Dataset_Diagnosis.py`)**:
   - Inspect the 10 diagnostic categories: Missing values, duplicates, IQR outliers, data types, cardinality, and consistency issues.
   - Expand the record inspection panels to examine exact affected rows.
4. **Data Quality Dashboard (`4_Quality_Dashboard.py`)**:
   - View the interactive Plotly quality gauge, dimensional breakdown cards, missing values bar chart, IQR outlier frequencies, type distribution pie, and correlation heatmap.
   - Explore individual numerical distributions with histograms and box plots.
5. **Generative AI Assistant (`5_GenAI_Assistant.py`)**:
   - Generate an in-depth AI report detailing root causes, ML readiness, and cleaning roadmap.
   - Chat interactively with the assistant using preset chips or custom prompts.
6. **Data Cleaning Studio (`6_Data_Cleaning_Studio.py`)**:
   - Execute non-destructive transformations: mean/median/mode imputation, deduplication, IQR Winsorization, casing standardization, and sentinel replacement.
   - Review before-and-after metric diffs.
   - Use **"Undo"** or **"Reset"** at any time.
7. **AI Data Quality Report (`7_Quality_Report.py`)**:
   - Review Before vs After quality scores and dimensional delta gains.
   - Click **"Download PDF Audit Report"** to export an executive PDF document.
   - Download the JSON audit report or cleaned CSV/Excel files.
8. **Settings (`8_Settings.py`)**:
   - Enter your OpenAI API key, configure custom models/proxies, adjust scoring formula weights, tune IQR thresholds, or reset session memory.

---

## 12. Verification & Automated Test Suite

A comprehensive test suite of 34 unit and integration tests is included. To execute the tests:

```powershell
python -m pytest tests -v
```

### Test Coverage Highlights:
- `test_data_loader.py`: CSV, Excel, BytesIO, empty file rejection, size boundary tests.
- `test_outlier_detector.py`: IQR 1.5x formulas, upper/lower bounds, row slicing, Z-score.
- `test_quality_scoring.py`: Completeness, Uniqueness, Validity, Consistency, custom weights, empty datasets.
- `test_quality_analyzer.py`: Complete 10-point analysis integration.
- `test_data_cleaner.py`: Imputation, deduplication, Winsorization, text normalization, schema casting.
- `test_ai_service.py`: Structured summary compactness, prompt injection filtering, offline fallback synthesis, chat Q&A.
- `test_report_generator.py`: PDF document compilation, JSON serialization, CSV & Excel exports.

---

## 13. Security and Privacy Principles

- **Zero Raw Data Transmission**: The Generative AI service never sends complete raw datasets over the network. Only anonymized statistical aggregations, missing percentages, and sanitized column names are shared.
- **Prompt Injection Hardening**: All column names and user text entries are filtered against injection patterns before inclusion in prompts.
- **Volatile In-Memory Processing**: Uploaded files and clean snapshots reside strictly in RAM during session runtime; no user data is stored on disk without explicit export.
- **Credential Protection**: API keys are masked in the UI and never included in logs or generated PDF/JSON reports.

---

## 14. Future Enhancements

- Integration with YData Profiling for automated synthetic data generation.
- Support for distributed datasets via Apache Arrow / Polars for gigabyte-scale files.
- Automated ML model baseline training (e.g. LightGBM / XGBoost) directly on the cleaned dataset to measure accuracy improvements before and after remediation.
- Enterprise SSO and Role-Based Access Control (RBAC).

---

## 15. Authors & Acknowledgments

- **Project Title**: Intelligent Dataset Diagnosis and Quality Analysis System
- **Degree**: Bachelor of Technology (B.Tech) in Artificial Intelligence & Machine Learning
- **Domain**: Generative AI, Machine Learning, Data Quality Engineering
- **Supervision & Guidance**: Department of Artificial Intelligence and Machine Learning

