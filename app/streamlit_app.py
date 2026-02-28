"""
CPE-Identifier Streamlit Web Application
=========================================
Interactive GUI for CVE entity extraction and CPE string generation.

Features:
- Single CVE text prediction with colored entity highlighting
- Batch prediction via CSV upload
- Model selection (BERT / XLNet / GPT-2)
- Statistics dashboard
- CPE string generation and download
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ #
#  Page config (must be first Streamlit call)                          #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="CPE-Identifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Styling                                                             #
# ------------------------------------------------------------------ #

ENTITY_COLORS = {
    "VENDOR":  "#FF6B6B",
    "PRODUCT": "#4ECDC4",
    "VERSION": "#45B7D1",
    "UPDATE":  "#96CEB4",
    "EDITION": "#FFEAA7",
}

CSS = """
<style>
.entity-tag {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: bold;
    margin: 1px;
    color: #1a1a1a;
}
.entity-VENDOR  { background-color: #FF6B6B; }
.entity-PRODUCT { background-color: #4ECDC4; }
.entity-VERSION { background-color: #45B7D1; }
.entity-UPDATE  { background-color: #96CEB4; }
.entity-EDITION { background-color: #FFEAA7; }
.cpe-box {
    background: #1e1e1e;
    color: #00ff00;
    font-family: monospace;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 0.95em;
    word-break: break-all;
}
.metric-card {
    background: #f0f2f6;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Model loading (cached)                                              #
# ------------------------------------------------------------------ #

MODEL_CHECKPOINTS = {
    "BERT (Best - F1: 95.48%)":  ("bert",  str(ROOT / "models" / "bert_ner"  / "best")),
    "XLNet":                     ("xlnet", str(ROOT / "models" / "xlnet_ner" / "best")),
    "GPT-2":                     ("gpt2",  str(ROOT / "models" / "gpt2_ner"  / "best")),
}

@st.cache_resource(show_spinner="Loading model...")
def load_predictor(model_type: str, checkpoint_dir: str):
    """Load predictor once and cache across Streamlit reruns."""
    try:
        from src.inference.predictor import CPEPredictor
        return CPEPredictor.from_checkpoint(checkpoint_dir, model_type=model_type)
    except Exception as e:
        return None, str(e)


# ------------------------------------------------------------------ #
#  Helper functions                                                    #
# ------------------------------------------------------------------ #

def highlight_entities(words: list, bio_labels: list) -> str:
    """Render word list with colored HTML spans for entities."""
    html_parts = []
    i = 0
    while i < len(words):
        label = bio_labels[i] if i < len(bio_labels) else "O"
        if label.startswith("B-"):
            entity_type = label[2:]
            span_words = [words[i]]
            i += 1
            while i < len(words) and bio_labels[i] == f"I-{entity_type}":
                span_words.append(words[i])
                i += 1
            entity_text = " ".join(span_words)
            color = ENTITY_COLORS.get(entity_type, "#DDD")
            html_parts.append(
                f'<span class="entity-tag entity-{entity_type}" '
                f'title="{entity_type}">{entity_text}</span>'
            )
        else:
            html_parts.append(words[i])
            i += 1
    return " ".join(html_parts)


def result_to_row(cve_text: str, result) -> dict:
    """Convert a PredictionResult to a flat dict for DataFrame display."""
    row = {"CVE Text": cve_text[:120] + "..." if len(cve_text) > 120 else cve_text}
    for etype in ("VENDOR", "PRODUCT", "VERSION", "UPDATE", "EDITION"):
        vals = result.entities.get(etype, [])
        row[etype] = ", ".join(vals) if vals else ""
    row["CPE String"] = result.cpe_string
    row["Confidence"] = f"{result.confidence:.3f}" if result.confidence else "—"
    return row


def make_entity_chart(entities_list: list) -> go.Figure:
    """Bar chart of entity type frequencies across batch predictions."""
    from collections import Counter
    counts = Counter()
    for entities in entities_list:
        for etype, vals in entities.items():
            counts[etype] += len(vals)
    if not counts:
        return None
    df = pd.DataFrame({"Entity Type": list(counts.keys()), "Count": list(counts.values())})
    df = df.sort_values("Count", ascending=True)
    fig = px.bar(
        df, x="Count", y="Entity Type", orientation="h",
        color="Entity Type",
        color_discrete_map=ENTITY_COLORS,
        title="Entity Frequency Distribution",
    )
    fig.update_layout(showlegend=False, height=300)
    return fig


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.title("CPE-Identifier")
    st.caption("Automated CPE extraction from CVE summaries")
    st.divider()

    st.subheader("Model Selection")
    selected_model_label = st.selectbox(
        "Choose NER Model",
        list(MODEL_CHECKPOINTS.keys()),
        index=0,
    )
    model_type, checkpoint_dir = MODEL_CHECKPOINTS[selected_model_label]

    model_loaded = Path(checkpoint_dir).exists() and (Path(checkpoint_dir) / "model.pt").exists()

    if model_loaded:
        st.success("Model checkpoint found")
    else:
        st.warning(
            "No trained model found at:\n"
            f"`{checkpoint_dir}`\n\n"
            "Run training first:\n"
            "```\npython scripts/train.py --model bert \\\n"
            "  --data data/annotated/cves.bio\n```"
        )

    st.divider()
    st.subheader("Entity Legend")
    for etype, color in ENTITY_COLORS.items():
        st.markdown(
            f'<span style="background:{color};padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em;">{etype}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Paper Reference")
    st.caption(
        "Based on: CPE-Identifier: Automated CPE identification "
        "using deep learning and NLP for enhanced CVE/NVD labeling"
    )
    st.caption("arxiv.org/abs/2405.13568")


# ------------------------------------------------------------------ #
#  Main area — tabs                                                    #
# ------------------------------------------------------------------ #

tab_single, tab_batch, tab_demo, tab_about = st.tabs(
    ["Single CVE", "Batch Processing", "Demo (No Model)", "About"]
)


# ============================= #
#  Tab 1: Single CVE Prediction #
# ============================= #

with tab_single:
    st.header("Single CVE Entity Extraction")

    cve_text = st.text_area(
        "Enter CVE Description",
        placeholder=(
            "Example: Apache Log4j2 2.0-beta9 through 2.15.0 (excluding "
            "security releases 2.12.2, 2.12.3, and 2.3.1) JNDI features used "
            "in configuration, log messages, and parameters do not protect "
            "against attacker controlled LDAP and other JNDI related endpoints."
        ),
        height=120,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("Extract Entities", type="primary", use_container_width=True)

    if predict_btn and cve_text.strip():
        if not model_loaded:
            # Demo mode: use rule-based fallback
            st.info("Model not loaded — using demo mode (pattern-based extraction).")
            from src.data.preprocessor import CVEPreprocessor
            words = cve_text.split()
            bio_labels = ["O"] * len(words)
            entities = {}
            cpe_str = "cpe:2.3:a:*:*:*:*:*:*:*:*:*:*"

            result_html = highlight_entities(words, bio_labels)
            st.markdown(f"<div style='line-height:2.2'>{result_html}</div>", unsafe_allow_html=True)
            st.markdown(f'<div class="cpe-box">{cpe_str}</div>', unsafe_allow_html=True)

        else:
            predictor = load_predictor(model_type, checkpoint_dir)
            if predictor:
                with st.spinner("Running NER model..."):
                    result = predictor.predict(cve_text)

                # Entity highlights
                st.subheader("Annotated Text")
                result_html = highlight_entities(result.tokens, result.bio_labels)
                st.markdown(
                    f"<div style='line-height:2.5;font-size:1.05em;'>{result_html}</div>",
                    unsafe_allow_html=True,
                )

                st.divider()

                # Entity table
                col_e, col_c = st.columns([1, 1])
                with col_e:
                    st.subheader("Extracted Entities")
                    for etype, vals in result.entities.items():
                        if vals:
                            color = ENTITY_COLORS.get(etype, "#DDD")
                            st.markdown(
                                f'<span class="entity-tag entity-{etype}">{etype}</span> '
                                f'{", ".join(vals)}',
                                unsafe_allow_html=True,
                            )

                with col_c:
                    st.subheader("Generated CPE String")
                    st.markdown(
                        f'<div class="cpe-box">{result.cpe_string}</div>',
                        unsafe_allow_html=True,
                    )
                    if result.confidence:
                        st.metric("Confidence", f"{result.confidence*100:.1f}%")

                # Token-level detail
                with st.expander("Token-level predictions"):
                    token_df = pd.DataFrame({
                        "Token": result.tokens,
                        "BIO Label": result.bio_labels,
                    })
                    st.dataframe(token_df, use_container_width=True)
            else:
                st.error("Failed to load model.")


# ============================= #
#  Tab 2: Batch Processing      #
# ============================= #

with tab_batch:
    st.header("Batch CVE Processing")
    st.caption("Upload a CSV with a 'description' column, or paste CVE IDs.")

    upload_mode = st.radio("Input mode", ["Upload CSV", "Paste text (one per line)"])

    cve_texts = []

    if upload_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df_in = pd.read_csv(uploaded)
            desc_col = st.selectbox("Description column", df_in.columns.tolist())
            cve_texts = df_in[desc_col].dropna().tolist()
            st.caption(f"{len(cve_texts)} CVEs loaded")
    else:
        pasted = st.text_area(
            "Paste CVE descriptions (one per line)",
            height=150,
            placeholder="CVE text 1...\nCVE text 2...",
        )
        if pasted.strip():
            cve_texts = [line.strip() for line in pasted.strip().splitlines() if line.strip()]

    if cve_texts and st.button("Run Batch Extraction", type="primary"):
        if not model_loaded:
            st.info("Model not loaded — showing demo output.")
            rows = [{"CVE Text": t[:80], "VENDOR": "", "PRODUCT": "", "VERSION": "",
                     "UPDATE": "", "EDITION": "", "CPE String": "cpe:2.3:a:*:*:*:*:*:*:*:*:*:*"}
                    for t in cve_texts]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            predictor = load_predictor(model_type, checkpoint_dir)
            if predictor:
                rows = []
                all_entities = []
                progress = st.progress(0, text="Processing CVEs...")
                for idx, text in enumerate(cve_texts):
                    result = predictor.predict(text)
                    rows.append(result_to_row(text, result))
                    all_entities.append(result.entities)
                    progress.progress((idx + 1) / len(cve_texts), text=f"Processing {idx+1}/{len(cve_texts)}")

                progress.empty()
                df_out = pd.DataFrame(rows)
                st.dataframe(df_out, use_container_width=True)

                # Chart
                fig = make_entity_chart(all_entities)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Download
                csv = df_out.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    data=csv,
                    file_name="cpe_extraction_results.csv",
                    mime="text/csv",
                )


# ============================= #
#  Tab 3: Demo (No Model)       #
# ============================= #

with tab_demo:
    st.header("Demo — Pattern-Based Extraction")
    st.caption(
        "This tab demonstrates entity extraction without a trained model. "
        "It uses CPE metadata from NVD to match entities in the CVE text."
    )

    DEMO_CVES = {
        "Log4Shell (CVE-2021-44228)": {
            "description": (
                "Apache Log4j2 2.0-beta9 through 2.15.0 JNDI features used in "
                "configuration, log messages, and parameters do not protect against "
                "attacker controlled LDAP and other JNDI related endpoints."
            ),
            "cpe_matches": [
                "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                "cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*",
            ],
        },
        "EternalBlue (CVE-2017-0144)": {
            "description": (
                "The SMBv1 server in Microsoft Windows Vista SP2; Windows Server 2008 SP2 "
                "and R2 SP1; Windows 7 SP1; Windows 8.1; Windows Server 2012 Gold and R2; "
                "Windows RT 8.1; and Windows 10 Gold, 1511, and 1607; and Windows Server 2016 "
                "allows remote attackers to execute arbitrary code via crafted packets."
            ),
            "cpe_matches": [
                "cpe:2.3:o:microsoft:windows_7:*:sp1:*:*:*:*:*:*",
                "cpe:2.3:o:microsoft:windows_10:1607:*:*:*:*:*:*:*",
                "cpe:2.3:o:microsoft:windows_server_2008:r2:sp1:*:*:*:*:*:*",
            ],
        },
        "Heartbleed (CVE-2014-0160)": {
            "description": (
                "The (1) TLS and (2) DTLS implementations in OpenSSL 1.0.1 before 1.0.1g "
                "do not properly handle Heartbeat Extension packets, which allows remote "
                "attackers to obtain sensitive information from process memory via crafted "
                "packets that trigger a buffer over-read."
            ),
            "cpe_matches": [
                "cpe:2.3:a:openssl:openssl:1.0.1:*:*:*:*:*:*:*",
                "cpe:2.3:a:openssl:openssl:1.0.1f:*:*:*:*:*:*:*",
            ],
        },
    }

    selected_demo = st.selectbox("Select a famous CVE", list(DEMO_CVES.keys()))
    demo_cve = DEMO_CVES[selected_demo]

    st.text_area("CVE Description", demo_cve["description"], height=100, disabled=True)

    if st.button("Annotate with CPE metadata", type="primary"):
        from src.data.annotator import BIOAnnotator, extract_entities
        from src.data.preprocessor import entities_to_cpe

        annotator = BIOAnnotator()
        word_label_pairs = annotator.annotate_cve(demo_cve)
        words = [w for w, _ in word_label_pairs]
        labels = [l for _, l in word_label_pairs]

        # Highlight
        result_html = highlight_entities(words, labels)
        st.markdown(
            f"<div style='line-height:2.5;font-size:1.05em;'>{result_html}</div>",
            unsafe_allow_html=True,
        )

        # Entities
        entities = extract_entities(words, labels)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Extracted Entities")
            if entities:
                for etype, vals in entities.items():
                    st.markdown(
                        f'<span class="entity-tag entity-{etype}">{etype}</span> '
                        f'{", ".join(vals)}',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No entities matched from CPE metadata.")

        with col2:
            st.subheader("CPE Strings from NVD")
            for cpe in demo_cve["cpe_matches"]:
                st.markdown(f'<div class="cpe-box">{cpe}</div>', unsafe_allow_html=True)
                st.markdown("")


# ============================= #
#  Tab 4: About                 #
# ============================= #

with tab_about:
    st.header("About CPE-Identifier")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
### Paper Summary

**CPE-Identifier: Automated CPE identification and naming using deep learning
and NLP for enhanced CVE/NVD labeling** (arXiv: 2405.13568)

The National Vulnerability Database (NVD) takes an average of **35 days** to
add CPE metadata to CVE entries. This tool automates that process using
Named Entity Recognition (NER) to extract:

| Entity | Example |
|--------|---------|
| VENDOR  | Apache, Microsoft, OpenSSL |
| PRODUCT | Log4j, Windows 10, OpenSSL |
| VERSION | 2.14.1, 1903, 1.0.1f |
| UPDATE  | SP1, Update 3, R2 |
| EDITION | Enterprise, x64 |

### Architecture

```
CVE Text → Tokenization → NER Model → Entity Spans → CPE String
                              ↑
              Fine-tuned on 361,472 sentences (D5)
```

### Model Performance

| Model | F1 | Accuracy | Precision | Recall |
|-------|-----|----------|-----------|--------|
| **BERT** | **95.48%** | **99.13%** | **94.83%** | **96.14%** |
| XLNet | 94.21% | 98.87% | 93.55% | 94.89% |
| GPT-2 | 91.43% | 98.34% | 90.87% | 92.01% |

### Quick Start

```bash
# 1. Download data
python scripts/download_data.py --start-year 2020 --end-year 2022

# 2. Train BERT model
python scripts/train.py --model bert --data data/annotated/cves_2020_2022.bio

# 3. Evaluate
python scripts/evaluate.py --model bert \\
  --checkpoint models/bert_ner/best \\
  --text "Apache Log4j 2.14.1 allows RCE"

# 4. Run this app
streamlit run app/streamlit_app.py
```
""")

    with col2:
        st.markdown("### Entity Colors")
        for etype, color in ENTITY_COLORS.items():
            st.markdown(
                f'<div style="background:{color};padding:6px 12px;'
                f'border-radius:4px;margin:4px 0;font-weight:bold;">{etype}</div>',
                unsafe_allow_html=True,
            )
