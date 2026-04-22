#!/usr/bin/env python3
"""
CLI for the Intent Classifier.

Usage:
  python main.py                          # interactive REPL
  python main.py "find all .env files"   # single prediction
  python main.py --dataset my_data.json  # train on custom data then REPL
  python main.py --save model.joblib     # train and save model
  python main.py --load model.joblib     # load saved model then REPL
"""

import argparse
import sys
from classifier import build_classifier, IntentClassifier

RANK_LABELS = {1: "trivial", 2: "simple", 3: "moderate", 4: "complex", 5: "advanced"}
CATEGORY_ICONS = {
    "SEARCH":   "🔍",
    "CREATE":   "✏️ ",
    "QUESTION": "❓",
    "UNKNOWN":  "❔",
}


def fmt_result(result: dict) -> str:
    icon  = CATEGORY_ICONS.get(result["label"], "  ")
    rank  = result["rank"]
    conf  = result["confidence"] * 100
    rdesc = RANK_LABELS.get(rank, "")
    bars  = "█" * rank + "░" * (5 - rank)

    lines = [
        f"\n  {icon} {result['label']}  |  rank {rank}/5 [{bars}] {rdesc}  |  confidence {conf:.1f}%",
        "  scores: " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(result["scores"].items())),
    ]
    return "\n".join(lines)


def repl(clf: IntentClassifier):
    print("\n── Intent Classifier REPL (Ctrl+C or 'exit' to quit) ──")
    while True:
        try:
            text = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if text.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break
        if not text:
            continue
        result = clf.predict(text)
        print(fmt_result(result))


def main():
    parser = argparse.ArgumentParser(description="Intent Classifier for Terminal Agent")
    parser.add_argument("query",        nargs="?",      help="Single query to classify")
    parser.add_argument("--dataset",    metavar="FILE", help="Path to custom JSON dataset")
    parser.add_argument("--save",       metavar="FILE", help="Save trained model to file")
    parser.add_argument("--load",       metavar="FILE", help="Load pre-trained model from file")
    args = parser.parse_args()

    # Build / load model
    if args.load:
        clf = IntentClassifier.load(args.load)
        print(f"Model loaded from {args.load}")
    else:
        clf = build_classifier(args.dataset)

    if args.save:
        clf.save(args.save)

    # Single prediction or REPL
    if args.query:
        result = clf.predict(args.query)
        print(fmt_result(result))
    else:
        repl(clf)


if __name__ == "__main__":
    main()