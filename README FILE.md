# EvidenceRank

## Project Files

```text
EvidenceRank/
│
├── main.py
│
├── data/
│   └── raw/
│       ├── papers.json
│       └── judgments.json
│
└── src/
    ├── app.py
    ├── bm25.py
    ├── collect_corpus.py
    ├── comparison.py
    ├── evaluate.py
    ├── evaluation.py
    ├── evidence_rank.py
    ├── index.py
    ├── judgments.py
    ├── load_data.py
    ├── preprocessing.py
    ├── retrieval.py
    ├── search_engine.py
    └── zone_search.py
```

> `__pycache__` files are generated automatically by Python and are not source files.

---

## Requirements

Python 3.10 or newer is recommended.

Install the required packages:

```bash
pip install numpy scikit-learn rank-bm25 PySide6
```

---

# Main Application

## Compile

Python does not require compilation to run the project.

To check/compile all Python source files:

```bash
python -m py_compile main.py
python -m py_compile src/app.py
python -m py_compile src/bm25.py
python -m py_compile src/collect_corpus.py
python -m py_compile src/comparison.py
python -m py_compile src/evaluate.py
python -m py_compile src/evaluation.py
python -m py_compile src/evidence_rank.py
python -m py_compile src/index.py
python -m py_compile src/judgments.py
python -m py_compile src/load_data.py
python -m py_compile src/preprocessing.py
python -m py_compile src/retrieval.py
python -m py_compile src/search_engine.py
python -m py_compile src/zone_search.py
```

If there is no error, the files compile successfully.

## Run the application

Run from the **EvidenceRank project root**:

```bash
python main.py
```

---

# Source Files and Run Commands

## 1. `main.py`

Main entry point for the desktop application.

It:

- Adds the `src` folder to Python's path.
- Imports `EvidenceRankApp`.
- Creates the PySide6 application.
- Opens the EvidenceRank window.

### Run

```bash
python main.py
```

---

## 2. `src/app.py`

Contains the PySide6 graphical user interface.

It loads:

```text
data/raw/papers.json
```

and provides the search interface and paper-result display.

### Run

Normally run through:

```bash
python main.py
```

---

## 3. `src/preprocessing.py`

Contains the text preprocessing function.

The preprocessing:

```text
lowercase
    ↓
remove special characters
    ↓
split into tokens
```

### Run

```bash
python src/preprocessing.py
```

---

## 4. `src/load_data.py`

Loads:

```text
data/raw/papers.json
```

and applies the preprocessing function to paper titles and abstracts.

### Run

```bash
python src/load_data.py
```

---

## 5. `src/retrieval.py`

Implements the basic TF-IDF retrieval system.

It uses:

```text
TfidfVectorizer
cosine_similarity
```

### Run

```bash
python src/retrieval.py
```

Then enter a search query when prompted.

Example:

```text
air pollution health
```

---

## 6. `src/bm25.py`

Implements BM25 retrieval using:

```text
rank_bm25
```

### Run

```bash
python src/bm25.py
```

Then enter a search query.

---

## 7. `src/zone_search.py`

Implements title/abstract zone-based retrieval.

Weights:

```text
Title     = 70%
Abstract = 30%
```

Formula:

```text
Zone Score =
0.70 × Title Score +
0.30 × Abstract Score
```

### Run

```bash
python src/zone_search.py
```

---

## 8. `src/search_engine.py`

Contains the main EvidenceRank search engine.

It combines:

```text
TF-IDF
BM25
Title/Abstract zone scoring
Citation authority
Recency
Phrase matching
```

### Run

```bash
python src/search_engine.py
```

If the file is being used as a module by the GUI, it can be imported instead of run directly.

---

## 9. `src/evidence_rank.py`

Contains EvidenceRank ranking functionality used by the project.

### Compile

```bash
python -m py_compile src/evidence_rank.py
```

### Run

```bash
python src/evidence_rank.py
```

---

## 10. `src/index.py`

Contains indexing functionality used for preparing/searching the paper corpus.

### Compile

```bash
python -m py_compile src/index.py
```

### Run

```bash
python src/index.py
```

---

## 11. `src/collect_corpus.py`

Collects research papers from the OpenAlex API.

The configured search queries are:

```text
air pollution health
climate change human health
respiratory disease environment
cardiovascular disease air pollution
```

The collected corpus is saved to:

```text
data/raw/papers.json
```

### Run

```bash
python src/collect_corpus.py
```

Internet access is required.

---

## 12. `src/judgments.py`

Creates the relevance-judgment file used for evaluation.

Output:

```text
data/raw/judgments.json
```

### Run

```bash
python src/judgments.py
```

---

## 13. `src/evaluate.py`

Evaluates the EvidenceRank retrieval system using the judged queries.

### Run

```bash
python src/evaluate.py
```

---

## 14. `src/evaluation.py`

Evaluates the TF-IDF baseline.

### Run

```bash
python src/evaluation.py
```

---

## 15. `src/comparison.py`

Compares the stored evaluation results of:

```text
TF-IDF baseline
vs
EvidenceRank
```

### Run

```bash
python src/comparison.py
```

---

# Data Files

## `data/raw/papers.json`

Main research-paper corpus.

The application and retrieval modules read this file.

Each paper contains fields such as:

```json
{
  "id": "...",
  "title": "...",
  "abstract": "...",
  "year": 2024,
  "citations": 10,
  "source": "OpenAlex",
  "url": "..."
}
```

---

## `data/raw/judgments.json`

Contains relevance judgments used by the evaluation scripts.

It is used to calculate retrieval performance.

---

# Recommended Execution Order

If starting with a fresh project:

### 1. Install dependencies

```bash
pip install numpy scikit-learn rank-bm25 PySide6
```

### 2. Collect the corpus

```bash
python src/collect_corpus.py
```

### 3. Create judgments

```bash
python src/judgments.py
```

### 4. Run the main application

```bash
python main.py
```

### 5. Run EvidenceRank evaluation

```bash
python src/evaluate.py
```

### 6. Run TF-IDF evaluation

```bash
python src/evaluation.py
```

### 7. Compare results

```bash
python src/comparison.py
```

---

# Quick Run

If `papers.json` and `judgments.json` are already present:

```bash
pip install numpy scikit-learn rank-bm25 PySide6
python main.py
```

---

# Windows Command Summary

Run these commands from the project folder:

```cmd
cd EvidenceRank

pip install numpy scikit-learn rank-bm25 PySide6

python main.py
```

For the individual modules:

```cmd
python src/preprocessing.py
python src/load_data.py
python src/retrieval.py
python src/bm25.py
python src/zone_search.py
python src/search_engine.py
python src/evidence_rank.py
python src/index.py
python src/collect_corpus.py
python src/judgments.py
python src/evaluate.py
python src/evaluation.py
python src/comparison.py
```

---

# Python Compilation Check

To compile/check every source file at once:

```cmd
python -m compileall main.py src
```

Successful compilation creates `__pycache__` folders automatically.

To remove generated cache files on Windows PowerShell:

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```
