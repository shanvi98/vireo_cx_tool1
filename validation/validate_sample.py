"""
validate_sample.py - how often is the classifier wrong?
--------------------------------------------------------
Scores the classifier against validation/gold_labels.csv: 120 tickets whose
full customer message and agent note were read and labelled by hand with
  true_issue       one label from llm_classify.ISSUES
  true_category    the category that issue belongs to
  hardware_defect  Y / N / U (U = genuinely ambiguous, e.g. "connection
                   drops" - excluded from defect precision/recall)

Two splits, and the difference matters:
  dev      80 tickets. The keyword rules were written AFTER reading these,
           so rule-based scores on dev are optimistic.
  holdout  40 tickets labelled before the rules were run on them. This is
           the honest number for the rule-based classifier.
The LLM prompt never saw either split, so for the LLM both splits are fair.

Also reported, as a baseline: how often the intake bot's category tag agrees
with the hand label. If the classifier can't beat the tag, it isn't earning
its keep.

Usage:
    python validation/validate_sample.py                 # rule-based
    python validation/validate_sample.py --llm gemini    # also score the LLM
Writes validation/results.md.
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
from clean_and_join import build_clean_dataset  # noqa: E402
import llm_classify  # noqa: E402


def score(df: pd.DataFrame, pred_issue: str, pred_defect: str) -> dict:
    pred_cat = df[pred_issue].map(llm_classify.ISSUE_CATEGORY)
    d = df[df["hardware_defect"] != "U"]
    truth = d["hardware_defect"] == "Y"
    pred = d[pred_defect].astype(bool)
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    return {
        "n": len(df),
        "issue_acc": (df[pred_issue] == df["true_issue"]).mean(),
        "category_acc": (pred_cat == df["true_category"]).mean(),
        "defect_n": len(d),
        "defect_acc": (truth == pred).mean(),
        "defect_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "defect_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "tp": tp, "fp": fp, "fn": fn,
    }


def fmt(s: dict) -> str:
    return (f"| {s['n']} | {s['issue_acc']:.0%} | {s['category_acc']:.0%} | {s['defect_acc']:.0%} "
            f"| {s['defect_precision']:.0%} ({s['tp']}/{s['tp'] + s['fp']}) "
            f"| {s['defect_recall']:.0%} ({s['tp']}/{s['tp'] + s['fn']}) |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["groq", "gemini"], default=None)
    args = parser.parse_args()

    gold = pd.read_csv(os.path.join(HERE, "gold_labels.csv"))
    df = build_clean_dataset()
    g = gold.merge(df, on="ticket_id", how="left", validate="one_to_one")
    rows = g.to_dict("records")

    llm_classify.USE_LLM = False
    rb = llm_classify.classify_many(rows)
    g["rb_issue"] = [r["issue"] for r in rb]
    g["rb_defect"] = [r["potential_hardware_defect"] for r in rb]
    # Baseline: the intake tag, read as "is it a hardware category?"
    g["tag_defect"] = g["category"].isin(["Charging & Battery", "Audio Quality", "Warranty & Repair"])

    systems = [("Rule-based", "rb_issue", "rb_defect")]
    if args.llm:
        llm_classify.USE_LLM, llm_classify.LLM_PROVIDER = True, args.llm
        out = llm_classify.classify_many(rows)
        g["llm_issue"] = [r["issue"] for r in out]
        g["llm_defect"] = [r["potential_hardware_defect"] for r in out]
        g["llm_source"] = [r["source"] for r in out]
        systems.append((f"LLM ({llm_classify.MODELS[args.llm]})", "llm_issue", "llm_defect"))

    lines = ["# Classifier validation", "",
             "Gold set: 120 tickets labelled by reading the full message and agent note "
             "(see `gold_labels.csv` and the docstring of `validate_sample.py` for the rules).", "",
             "| System | Split | n | Issue exact | Category | Defect acc. | Defect precision | Defect recall |",
             "|---|---|---|---|---|---|---|---|"]
    for split in ["dev", "holdout"]:
        s = g[g["split"] == split]
        tag_cat_acc = (s["category"] == s["true_category"]).mean()
        tag = score(s.assign(tag_issue="other"), "tag_issue", "tag_defect")
        lines.append(f"| Intake tag (baseline) | {split} | {len(s)} | - | {tag_cat_acc:.0%} | {tag['defect_acc']:.0%} "
                     f"| {tag['defect_precision']:.0%} | {tag['defect_recall']:.0%} |")
        for name, ic, dc in systems:
            lines.append(f"| {name} | {split} " + fmt(score(s, ic, dc)))
    if args.llm:
        st = llm_classify.STATS
        lines += ["", f"LLM calls: {st['llm_ok']} live, {st['llm_cached']} from cache, "
                      f"{st['llm_failed_fell_back']} failed and fell back to rules."]

    lines += ["", "## Where each system is wrong", ""]
    for name, ic, dc in systems:
        wrong = g[(g[ic] != g["true_issue"]) | ((g["hardware_defect"] != "U") & (g[dc] != (g["hardware_defect"] == "Y")))]
        lines += [f"### {name}: {len(wrong)} of {len(g)} tickets have an issue or defect error", "",
                  "| Ticket | Split | Tag | Gold issue | Predicted | Gold defect | Pred defect |", "|---|---|---|---|---|---|---|"]
        for _, r in wrong.iterrows():
            lines.append(f"| {r.ticket_id} | {r.split} | {r.category} | {r.true_issue} | {r[ic]} "
                         f"| {r.hardware_defect} | {'Y' if r[dc] else 'N'} |")
        lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(HERE, "results.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report.split("## Where")[0])
    print(f"Full error list written to {os.path.join(HERE, 'results.md')}")


if __name__ == "__main__":
    main()
