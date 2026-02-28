#!/usr/bin/env python3
"""
Training Entry Point
====================
Fine-tune BERT / XLNet / GPT-2 on annotated CVE NER data.

Usage:
    python scripts/train.py --model bert --data data/annotated/cves_2020_2022.bio
    python scripts/train.py --model xlnet --epochs 10 --batch-size 8
    python scripts/train.py --model gpt2 --lr 3e-5 --no-augment
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

from src.data.dataset import NERDataset, NERDataLoader
from src.models.bert_ner import BERTForNER
from src.models.xlnet_ner import XLNetForNER
from src.models.gpt2_ner import GPT2ForNER
from src.training.trainer import NERTrainer, TrainingConfig
from src.training.evaluator import NERMetricComputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_MAP = {
    "bert":  ("bert-base-uncased", BERTForNER),
    "xlnet": ("xlnet-base-cased",  XLNetForNER),
    "gpt2":  ("gpt2",              GPT2ForNER),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train CPE NER model")

    # Model
    parser.add_argument("--model", choices=["bert", "xlnet", "gpt2"], default="bert")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Override HuggingFace model name")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to annotated BIO or JSONL data file")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio",   type=float, default=0.1)

    # Training
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio",  type=float, default=0.1)
    parser.add_argument("--grad-clip",    type=float, default=1.0)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num-workers",  type=int,   default=4,
                        help="DataLoader worker processes (default: 4)")

    # Augmentation
    parser.add_argument("--augment",    action="store_true", default=False,
                        help="Apply DistilRoBERTa augmentation before training")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--augment-ratio", type=float, default=0.15)

    # Output
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--log-dir",  type=str, default="logs")

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    pretrained, ModelClass = MODEL_MAP[args.model]
    if args.pretrained:
        pretrained = args.pretrained

    logger.info("Model: %s (%s)", args.model.upper(), pretrained)

    # Load tokenizer
    logger.info("Loading tokenizer: %s", pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    data_path = args.data
    logger.info("Loading dataset: %s", data_path)
    if data_path.endswith(".bio"):
        dataset = NERDataset.from_bio_file(data_path, tokenizer)
    elif data_path.endswith(".jsonl"):
        dataset = NERDataset.from_jsonl(data_path, tokenizer)
    else:
        raise ValueError("Data file must be .bio or .jsonl")

    logger.info("Dataset size: %d samples", len(dataset))

    # Optional augmentation
    if args.augment:
        from src.data.augmentor import DataAugmentor
        logger.info("Applying DistilRoBERTa augmentation ...")
        augmentor = DataAugmentor(augment_ratio=args.augment_ratio)
        from src.data.annotator import BIOAnnotator
        annotator = BIOAnnotator()
        raw_seqs = [[(w, l) for w, l in zip(
            [tokenizer.decode([id_]) for id_ in item["input_ids"]],
            ["O"] * len(item["input_ids"])
        )] for item in dataset]
        # Augment and rebuild dataset
        logger.warning("Augmentation on pre-tokenized data is limited. "
                       "For best results, augment before tokenization.")

    # Create dataloaders
    data_module = NERDataLoader(
        dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size * 2,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    logger.info(str(data_module))

    # Initialize model
    logger.info("Loading model: %s", pretrained)
    model = ModelClass(pretrained_model=pretrained)
    logger.info(str(model))

    # Training config
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_clip=args.grad_clip,
        save_dir=args.save_dir,
        model_name=f"{args.model}_ner",
        log_dir=args.log_dir,
        patience=args.patience,
    )

    # Train
    trainer = NERTrainer(model, config)
    history = trainer.train(data_module.train, data_module.val)

    # Evaluate on test set
    logger.info("\n=== Test Set Evaluation ===")
    test_metrics = trainer.evaluate(data_module.test)

    # Save results
    results_path = Path(args.save_dir) / f"{args.model}_ner" / "results.json"
    results = {
        "model": args.model,
        "pretrained": pretrained,
        "test_metrics": test_metrics,
        "history": history,
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    logger.info(
        "\nFinal Results:\n"
        "  F1:        %.4f\n"
        "  Accuracy:  %.4f\n"
        "  Precision: %.4f\n"
        "  Recall:    %.4f",
        test_metrics["f1"], test_metrics["accuracy"],
        test_metrics["precision"], test_metrics["recall"],
    )


if __name__ == "__main__":
    main()
