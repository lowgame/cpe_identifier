# CPE-Identifier: Project Guide & Progress Tracker

> Based on: "CPE-Identifier: Automated CPE identification and naming using deep learning and NLP for enhanced CVE/NVD labeling"
> Paper: https://arxiv.org/html/2405.13568v1

---

## Project Overview

**Goal:** Extract CPE (Common Platform Enumeration) entities from CVE (Common Vulnerabilities and Exposures) summaries using Named Entity Recognition (NER) deep learning models.

**Problem:** NVD takes an average of 35 days to label CVEs with CPE metadata, creating dangerous vulnerability windows for organizations.

**Solution:** Automated NER pipeline that identifies 5 entity types from raw CVE text:
- `VENDOR` — software/hardware vendor (e.g., "Microsoft", "Apache")
- `PRODUCT` — specific product name (e.g., "Windows 10", "Log4j")
- `VERSION` — version number (e.g., "2.14.1", "1903")
- `UPDATE` — update/patch identifier (e.g., "SP1", "Update 3")
- `EDITION` — edition specification (e.g., "Enterprise", "x64")

**Best reported performance (BERT):** F1=95.48%, Accuracy=99.13%, Precision=94.83%, Recall=96.14%

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INPUT: CVE Text                    │
│  "Apache Log4j 2.14.1 allows remote code execution" │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Stage 1: Data Pipeline                  │
│  NVD API Fetcher → Preprocessor → BIO Annotator     │
│  Data Augmentor (DistilRoBERTa) → Merged Corpus      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Stage 2: NER Models                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │  BERT   │  │  XLNet  │  │  GPT-2  │             │
│  │  (NER)  │  │  (NER)  │  │  (NER)  │             │
│  └─────────┘  └─────────┘  └─────────┘             │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Stage 3: Output & GUI                   │
│  Entity Extraction → CPE Format → Streamlit App      │
└─────────────────────────────────────────────────────┘
```

---

## BIO Tagging Scheme

Each token gets one of 11 tags:
```
O        — Outside any entity
B-VENDOR — Beginning of vendor entity
I-VENDOR — Inside vendor entity
B-PRODUCT
I-PRODUCT
B-VERSION
I-VERSION
B-UPDATE
I-UPDATE
B-EDITION
I-EDITION
```

Example:
```
Apache   → B-VENDOR
Log4j    → B-PRODUCT
2.14.1   → B-VERSION
allows   → O
remote   → O
code     → O
execution → O
```

---

## Dataset Strategy

| Dataset | Source | Description |
|---------|--------|-------------|
| D1 | GitHub (2010-2013) | Annotated cybersecurity texts |
| D2 | NVD API (1999-2021) | Raw unlabeled CVE summaries |
| D3 | BERT Auto-annotator | Auto-labeled CVE data |
| D4 | DistilRoBERTa Augmentor | Synthetic augmented samples |
| D5 | Merged Final | Combined training corpus (361,472 sentences) |

---

## Project Structure

```
cpe_identifier/
├── progress.md              ← This file
├── requirements.txt         ← Python dependencies
├── README.md                ← User-facing documentation
├── setup.py                 ← Package setup
├── config.yaml              ← Global configuration
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nvd_fetcher.py   ← Fetch CVE data from NVD API 2.0
│   │   ├── preprocessor.py  ← Text cleaning, tokenization, padding
│   │   ├── annotator.py     ← BIO tagging and auto-annotation
│   │   ├── augmentor.py     ← DistilRoBERTa data augmentation
│   │   └── dataset.py       ← PyTorch Dataset classes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py    ← Abstract base NER model
│   │   ├── bert_ner.py      ← BERT NER (best performer)
│   │   ├── xlnet_ner.py     ← XLNet NER
│   │   └── gpt2_ner.py      ← GPT-2 NER
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       ← Training loop (20 epochs)
│   │   └── evaluator.py     ← Precision/Recall/F1/Accuracy
│   └── inference/
│       ├── __init__.py
│       └── predictor.py     ← End-to-end inference pipeline
├── app/
│   └── streamlit_app.py     ← Streamlit web GUI
├── scripts/
│   ├── download_data.py     ← Download and prep datasets
│   ├── train.py             ← Training entry point
│   └── evaluate.py          ← Evaluation entry point
├── data/
│   ├── raw/                 ← Raw CVE data from NVD
│   ├── processed/           ← Tokenized, cleaned data
│   └── annotated/           ← BIO-tagged training data
├── models/                  ← Saved checkpoints
│   ├── bert_ner/
│   ├── xlnet_ner/
│   └── gpt2_ner/
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_inference.py
```

---

## TODO List

### Phase 1: Project Setup ✅ COMPLETE
- [x] Fetch and analyze paper
- [x] Create progress.md
- [x] Create directory structure
- [x] Write requirements.txt
- [x] Write config.yaml
- [x] Write setup.py

### Phase 2: Data Pipeline ✅ COMPLETE
- [x] **NVD Fetcher** (`src/data/nvd_fetcher.py`)
  - [x] NVD API 2.0 integration
  - [x] Rate limiting and pagination
  - [x] CVE text extraction and normalization
  - [x] Local cache (JSONL) to avoid re-fetching
- [x] **Preprocessor** (`src/data/preprocessor.py`)
  - [x] Text cleaning (HTML entities, unicode, whitespace)
  - [x] NLTK sentence tokenization
  - [x] Padding/truncating to 128 tokens
  - [x] Subword-to-BIO tag alignment (first subword rule)
  - [x] `parse_cpe_string` — CPE 2.3 → entity dict
  - [x] `entities_to_cpe` — entity dict → CPE 2.3 string
- [x] **Annotator** (`src/data/annotator.py`)
  - [x] BIO tagging scheme (11 tags, 5 entity types)
  - [x] CPE-to-entity mapping from NVD configurations
  - [x] Case-insensitive multi-word entity matching
  - [x] Label statistics and CoNLL/JSONL I/O
- [x] **Augmentor** (`src/data/augmentor.py`)
  - [x] DistilRoBERTa masked language modeling
  - [x] Entity-aware (only masks O-labeled tokens)
  - [x] Configurable ratio and num_augmented_per_sample
  - [x] Synonym fallback (no-model mode)
- [x] **Dataset** (`src/data/dataset.py`)
  - [x] PyTorch Dataset from BIO file or JSONL
  - [x] DataLoader with NER collate function
  - [x] 80/10/10 train/val/test split

### Phase 3: Models ✅ COMPLETE
- [x] **Base Model** (`src/models/base_model.py`)
  - [x] Abstract NER interface with save/load
  - [x] Shared compute_loss (CrossEntropy, IGNORE_LABEL_ID)
  - [x] count_parameters utility
- [x] **BERT NER** (`src/models/bert_ner.py`) — best model
  - [x] bert-base-uncased + token classification head
  - [x] Paper target: F1=95.48%, Accuracy=99.13%
- [x] **XLNet NER** (`src/models/xlnet_ner.py`)
  - [x] xlnet-base-cased backbone
- [x] **GPT-2 NER** (`src/models/gpt2_ner.py`)
  - [x] gpt2 backbone + pad token fix

### Phase 4: Training Pipeline ✅ COMPLETE
- [x] **Trainer** (`src/training/trainer.py`)
  - [x] 20-epoch loop with AdamW
  - [x] Linear warmup + decay schedule
  - [x] Gradient clipping (max_norm=1.0)
  - [x] Best-model checkpoint by validation F1
  - [x] TensorBoard logging (optional, graceful fallback)
  - [x] Early stopping with configurable patience
- [x] **Evaluator** (`src/training/evaluator.py`)
  - [x] seqeval entity-level F1, Precision, Recall, Accuracy
  - [x] Token-level accuracy
  - [x] Per-entity-type breakdown (NERMetrics dataclass)
  - [x] Decode batch logits → BIO label strings

### Phase 5: Inference ✅ COMPLETE
- [x] **Predictor** (`src/inference/predictor.py`)
  - [x] Load model from checkpoint (BERT/XLNet/GPT-2)
  - [x] Raw text → BIO labels → entity spans → CPE string
  - [x] PredictionResult dataclass with confidence score
  - [x] Batch prediction support
  - [x] Factory: `CPEPredictor.from_checkpoint()`

### Phase 6: Scripts & Entry Points ✅ COMPLETE
- [x] `scripts/download_data.py` — NVD fetch + BIO annotation
- [x] `scripts/train.py` — full CLI training pipeline
- [x] `scripts/evaluate.py` — test evaluation + demo mode

### Phase 7: Streamlit GUI ✅ COMPLETE
- [x] Single CVE text prediction with HTML entity highlighting
- [x] Color-coded entity spans (5 colors, one per entity type)
- [x] CPE string display (monospace terminal style)
- [x] Model selection dropdown (BERT/XLNet/GPT-2)
- [x] Batch CVE processing (CSV upload or paste)
- [x] Entity frequency bar chart (Plotly)
- [x] Demo tab (no model required — uses CPE metadata)
- [x] About tab with paper summary and quick-start guide

### Phase 8: Testing ✅ COMPLETE
- [x] `tests/test_data.py` — 21 tests: preprocessor, annotator, NVD fetcher
- [x] `tests/test_models.py` — 12 tests: base model, evaluator, metrics
- [x] `tests/test_inference.py` — integration tests (tokenizer-dependent)
- [x] All 33 non-transformers-dependent tests pass ✅
- [x] Bug fixed: `entities_to_cpe` key case normalization (VENDOR→vendor)
- [x] Bug fixed: lazy imports in `__init__.py` files (no eager transformers load)
- [x] Bug fixed: optional tensorboard (graceful no-op fallback)

---

## Known Limitations / Future Work
- [ ] `tests/test_inference.py` requires `transformers` installed (skip otherwise)
- [ ] Auto-annotation pipeline (BERT annotates unlabeled CVEs) not yet integrated
- [ ] Multi-GPU / distributed training support
- [ ] WandB integration for experiment tracking
- [ ] Docker / deployment container
- [ ] REST API endpoint (FastAPI) for production use
- [ ] CRF layer on top of BERT head for improved sequence coherence

---

## Progress Log

| Date | Task | Status |
|------|------|--------|
| 2026-02-28 | Paper analysis & project planning | ✅ Done |
| 2026-02-28 | Project structure & progress.md | ✅ Done |
| 2026-02-28 | Data pipeline (NVD fetcher, preprocessor, annotator, augmentor, dataset) | ✅ Done |
| 2026-02-28 | NER models (BERT, XLNet, GPT-2, base model) | ✅ Done |
| 2026-02-28 | Training pipeline (trainer, evaluator, metrics) | ✅ Done |
| 2026-02-28 | Inference pipeline (CPEPredictor, PredictionResult) | ✅ Done |
| 2026-02-28 | CLI scripts (download_data, train, evaluate) | ✅ Done |
| 2026-02-28 | Streamlit web GUI (single CVE, batch, demo, about) | ✅ Done |
| 2026-02-28 | Tests (33 passing), bug fixes, documentation | ✅ Done |

---

## Key References

- NVD API 2.0: https://nvd.nist.gov/developers/vulnerabilities
- CPE specification: https://csrc.nist.gov/projects/security-content-automation-protocol/specifications/cpe
- BERT: https://arxiv.org/abs/1810.04805
- XLNet: https://arxiv.org/abs/1906.08237
- seqeval: https://github.com/chakki-works/seqeval
