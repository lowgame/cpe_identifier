#!/usr/bin/env python3
"""
Model Evaluation Script
========================
Evaluate a trained NER model on a test dataset.

Usage:
    python scripts/evaluate.py --model bert --checkpoint models/bert_ner/best --data data/annotated/test.bio
    python scripts/evaluate.py --model bert --checkpoint models/bert_ner/best --text "Apache Log4j 2.14.1 allows RCE"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

from src.data.dataset import NERDataset, DataLoader
from src.inference.predictor import CPEPredictor
from src.training.evaluator import NERMetricComputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CPE NER model")
    parser.add_argument("--model",      choices=["bert", "xlnet", "gpt2"], default="bert")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--data",       type=str, default=None,
                        help="Path to BIO/JSONL test data file")
    parser.add_argument("--text",       type=str, default=None,
                        help="Single CVE text to predict (demo mode)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output",     type=str, default=None,
                        help="Save evaluation results to JSON file")
    return parser.parse_args()


def evaluate_on_file(predictor: CPEPredictor, data_path: str, batch_size: int) -> dict:
    """Full evaluation on annotated test file."""
    pretrained_map = {
        "bert":  "bert-base-uncased",
        "xlnet": "xlnet-base-cased",
        "gpt2":  "gpt2",
    }
    from src.data.annotator import BIOAnnotator
    annotator = BIOAnnotator()
    sequences = annotator.load_bio_file(data_path)

    all_preds = []
    all_refs = []

    for seq in sequences:
        if not seq:
            continue
        words = [w for w, _ in seq]
        ref_labels = [l for _, l in seq]
        text = " ".join(words)
        result = predictor.predict(text)
        pred_labels = result.bio_labels

        # Align lengths (truncation may differ)
        min_len = min(len(pred_labels), len(ref_labels))
        all_preds.append(pred_labels[:min_len])
        all_refs.append(ref_labels[:min_len])

    computer = NERMetricComputer()
    detailed = computer.compute_detailed(all_preds, all_refs)
    return {
        "f1": detailed.f1,
        "precision": detailed.precision,
        "recall": detailed.recall,
        "accuracy": detailed.accuracy,
        "per_entity": detailed.per_entity,
    }


def demo_predict(predictor: CPEPredictor, text: str) -> None:
    """Single text prediction demo."""
    result = predictor.predict(text)
    print(f"\n{'='*60}")
    print(f"CVE Text: {text}")
    print(f"{'='*60}")
    print("\nToken-level predictions:")
    for word, label in zip(result.tokens, result.bio_labels):
        marker = f"  [{label}]" if label != "O" else ""
        print(f"  {word:20s}{marker}")
    print(f"\nExtracted Entities:")
    for etype, vals in result.entities.items():
        print(f"  {etype:10s}: {', '.join(vals)}")
    print(f"\nGenerated CPE: {result.cpe_string}")
    print(f"Confidence:    {result.confidence:.3f}" if result.confidence else "")


def main():
    args = parse_args()

    logger.info("Loading predictor from %s ...", args.checkpoint)
    predictor = CPEPredictor.from_checkpoint(
        checkpoint_dir=args.checkpoint,
        model_type=args.model,
    )

    if args.text:
        demo_predict(predictor, args.text)
        return

    if args.data:
        logger.info("Evaluating on %s ...", args.data)
        results = evaluate_on_file(predictor, args.data, args.batch_size)

        print(f"\n{'='*50}")
        print(f"Evaluation Results ({args.model.upper()})")
        print(f"{'='*50}")
        print(f"F1:        {results['f1']:.4f} ({results['f1']*100:.2f}%)")
        print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
        print(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")

        if results["per_entity"]:
            print(f"\nPer-Entity:")
            for entity, scores in results["per_entity"].items():
                print(
                    f"  {entity:12s} P={scores.get('precision',0):.3f} "
                    f"R={scores.get('recall',0):.3f} "
                    f"F1={scores.get('f1-score',0):.3f}"
                )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Results saved to %s", args.output)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
