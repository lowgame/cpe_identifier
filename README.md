# CPE-Identifier

Automated **CPE (Common Platform Enumeration)** extraction from **CVE (Common Vulnerabilities and Exposures)** summaries using NER deep learning models.

Based on: [CPE-Identifier: Automated CPE identification and naming using deep learning and NLP for enhanced CVE/NVD labeling](https://arxiv.org/abs/2405.13568)

```
Hu, Wanyu, and Vrizlynn L. L. Thing. “CPE-Identifier: Automated CPE Identification and CVE Summaries Annotation with Deep Learning and NLP.” *arXiv preprint* arXiv:2405.13568, May 22, 2024. https://arxiv.org/abs/2405.13568
```


## Performance (BERT — best model)

| Metric | Score |
|--------|-------|
| F1 | **95.48%** |
| Accuracy | **99.13%** |
| Precision | **94.83%** |
| Recall | **96.14%** |

## What it does

Extracts 5 CPE entity types from raw CVE text:

```
Apache Log4j 2.14.1 allows remote code execution via LDAP endpoints.
│      │     │
VENDOR PRODUCT VERSION
→ cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Download CVE data from NVD API (2020-2022)
python scripts/download_data.py --start-year 2020 --end-year 2022

# 2. Train BERT NER model (20 epochs, best checkpoint saved)
python scripts/train.py --model bert --data data/annotated/cves_2020_2022.bio

# 3. Evaluate on test set
python scripts/evaluate.py \
  --model bert \
  --checkpoint models/bert_ner/best \
  --data data/annotated/test.bio

# 4. Single CVE prediction
python scripts/evaluate.py \
  --model bert \
  --checkpoint models/bert_ner/best \
  --text "Apache Log4j 2.14.1 allows remote code execution"

# 5. Launch web GUI
streamlit run app/streamlit_app.py
```

## Project Structure

```
cpe_identifier/
├── src/
│   ├── data/          ← NVD fetcher, preprocessor, annotator, augmentor, dataset
│   ├── models/        ← BERT, XLNet, GPT-2 NER models
│   ├── training/      ← Trainer, evaluator
│   └── inference/     ← CPEPredictor (end-to-end)
├── app/               ← Streamlit web GUI
├── scripts/           ← download_data.py, train.py, evaluate.py
├── tests/             ← pytest unit + integration tests
├── data/              ← raw/, processed/, annotated/
├── models/            ← saved checkpoints (bert_ner/, xlnet_ner/, gpt2_ner/)
├── config.yaml        ← global configuration
└── progress.md        ← project guide & todo tracker
```

## Models

| Model | HuggingFace ID | Paper F1 |
|-------|---------------|----------|
| BERT  | bert-base-uncased | **95.48%** |
| XLNet | xlnet-base-cased | 94.21% |
| GPT-2 | gpt2 | 91.43% |

## API Key (optional but recommended)

NVD limits unauthenticated requests to ~5 req/30s. With a free API key you get ~50 req/30s:

```bash
export NVD_API_KEY=your-key-here
python scripts/download_data.py ...
```

Get a key at: https://nvd.nist.gov/developers/request-an-api-key

## Run Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=html
```

## Architecture

```
CVE Text
  ↓ clean + tokenize (128 tokens)
BIO Annotator (CPE metadata → word labels)
  ↓
NER Model (BERT/XLNet/GPT-2) fine-tuned on 361,472 sentences
  ↓
Entity Spans → CPE 2.3 String
```

## Data Pipeline

```
NVD API → raw CVEs (D2)
         ↓
BERT auto-annotator → D3
         ↓
DistilRoBERTa augmentor → D4
         ↓
Merged corpus D5 (361,472 sentences)
```
