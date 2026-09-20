"""
run_pipeline.py
----------------
Entry point. Cleans and joins the five source files, classifies every ticket
(keyword rules by default; an LLM for the most recent weeks with --llm),
builds the weekly digest, the agent view and the goal tracker, and writes
web/dashboard_data.json for the static page to read.

Usage:
    python run_pipeline.py                        # offline, no API key needed
    python run_pipeline.py --llm gemini           # LLM on the last 4 weeks (cached)
    python run_pipeline.py --llm gemini --llm-weeks 12
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))
from clean_and_join import build_clean_dataset  # noqa: E402
import llm_classify  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "web", "dashboard_data.json")

# support-policy.pdf §6 (and Neha, in the email thread): Tier 2 cases are
# multi-touch and measured on resolution days, never on tickets closed.
TIER2_TEAM = "Escalations & Warranty"
HARDWARE_TAGS = {"Charging & Battery", "Audio Quality", "Warranty & Repair"}

# The business goal. Status chasers = contacts whose only purpose is "where is
# my order / pickup / refund / repair?". They exist because the customer had
# no update, so proactive status messages can remove them from the queue.
CHASER_ISSUES = [
    "order not delivered or delayed",
    "return pickup not done",
    "refund not received",
    "repair or warranty claim follow-up",
]
GOAL_TARGET_SHARE = 0.18  # from ~27%: a one-third cut. A judgment call, see README.
LEADERBOARD_WEEKS = 12


def week_start(ts: pd.Series) -> pd.Series:
    """Monday-start weeks. (v1 used 'W-MON', which is weeks *ending* Monday,
    so every 'week' silently started on a Tuesday.)"""
    return ts.dt.to_period("W-SUN").dt.start_time


def classify_all(df: pd.DataFrame, llm: str, llm_weeks: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    rows = df.to_dict("records")
    llm_classify.USE_LLM = False
    results = llm_classify.classify_many(rows)
    if llm:
        # Only recent tickets go to the model: that is all a weekly digest
        # needs, and it keeps the bill predictable. Results are cached, so
        # next week only the new week's tickets are billed.
        since = (cutoff - pd.Timedelta(weeks=llm_weeks)).normalize()
        idx = [i for i, r in enumerate(rows) if r["created_at"] >= since]
        llm_classify.USE_LLM, llm_classify.LLM_PROVIDER = True, llm
        for i, res in zip(idx, llm_classify.classify_many([rows[i] for i in idx])):
            results[i] = res
    out = df.copy()
    for key in ["issue", "customer_sentiment", "potential_hardware_defect", "defect_reason", "source"]:
        out[key] = [r.get(key) for r in results]
    out["issue_category"] = out["issue"].map(llm_classify.ISSUE_CATEGORY)
    out["is_chaser"] = out["issue"].isin(CHASER_ISSUES)
    return out


def build_weekly_digest(df: pd.DataFrame, cutoff: pd.Timestamp) -> list:
    d = df.copy()
    d["week"] = week_start(d["created_at"])
    weeks = sorted(d["week"].unique())
    rows, prev_counts = [], {}
    for wk in weeks:
        w = d[d["week"] == wk]
        counts = w["issue"].value_counts()
        top = []
        for issue, n in counts.head(8).items():
            wi = w[w["issue"] == issue]
            top.append({
                "issue": issue,
                "tickets": int(n),
                "prev_week": int(prev_counts.get(issue, 0)),
                "hardware": bool(llm_classify.ISSUE_IS_HARDWARE[issue]),
                "top_product": wi["product_name"].mode().iat[0] if wi["product_name"].notna().any() else None,
            })
        # Defects the intake tag missed: the text says hardware fault, the tag says something else.
        hidden = w[w["potential_hardware_defect"] & ~w["category"].isin(HARDWARE_TAGS)]
        angry = w[w["customer_sentiment"] == "angry"].sort_values("created_at", ascending=False)
        rows.append({
            "week": wk.strftime("%Y-%m-%d"),
            "complete": bool(wk + pd.Timedelta(days=7) <= cutoff.normalize() + pd.Timedelta(days=1)),
            "total_tickets": int(len(w)),
            "service_cost_inr": float(w["service_cost_inr"].sum()),
            "chaser_share": float(w["is_chaser"].mean()),
            "same_issue_repeat_rate": float(w["is_same_issue_repeat"].mean()),
            "angry_share": float((w["customer_sentiment"] == "angry").mean()),
            "defect_tickets": int(w["potential_hardware_defect"].sum()),
            "hidden_defects": [
                {"ticket_id": r.ticket_id, "tag": r.category, "issue": r.issue, "product": r.product_name}
                for r in hidden.itertuples()
            ][:5],
            "top_issues": top,
            "quotes": [
                {"ticket_id": r.ticket_id, "issue": r.issue,
                 "text": " ".join(str(r.customer_message).split())[:220]}
                for r in angry.head(3).itertuples()
            ],
            "classified_by": w["source"].value_counts().to_dict(),
        })
        prev_counts = counts.to_dict()
    rows.sort(key=lambda r: r["week"], reverse=True)
    return rows


def add_bounce_flag(df: pd.DataFrame) -> pd.DataFrame:
    """bounced = the customer came back about the same issue within 30 days."""
    d = df.sort_values(["customer_id", "product_sku", "created_at"]).copy()
    nxt = d.groupby(["customer_id", "product_sku"])["is_same_issue_repeat"].shift(-1)
    d["bounced"] = nxt.fillna(False).astype(bool)
    return d


def build_leaderboard(df: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    d = df.dropna(subset=["resolved_at", "agent_id"])
    d = d[d["is_attendance"]].copy()  # policy §10: attendance = resolved or closed
    d["week"] = week_start(d["resolved_at"])
    # A week is only reportable once it has fully happened. The export stops
    # at `cutoff`, so the last calendar week is a stub (v1 ranked agents on a
    # 2-day week: 35 tickets in total, "top" agent on 4).
    complete = d["week"] + pd.Timedelta(days=7) <= cutoff.normalize() + pd.Timedelta(days=1)
    weeks = sorted(d.loc[complete, "week"].unique())[-LEADERBOARD_WEEKS:]

    # Bounce-back rate needs 30 days of hindsight, so it is measured on the
    # 13 weeks of closures that ended 30 days before the cutoff.
    b_end = cutoff - pd.Timedelta(days=30)
    b = d[(d["resolved_at"] > b_end - pd.Timedelta(weeks=13)) & (d["resolved_at"] <= b_end)]
    bounce = b.groupby("agent_id").agg(bounce_rate=("bounced", "mean"), bounce_n=("bounced", "size"))

    tier1 = d[d["team"] != TIER2_TEAM]
    # How much does a weekly ranking actually tell you? Rank agents each week
    # and correlate with the next week. Low = the order is mostly noise.
    grid = (tier1[tier1["week"].isin(weeks)]
            .pivot_table(index="agent_id", columns="week", values="ticket_id", aggfunc="count", fill_value=0))
    rank_corr = [grid.iloc[:, i].rank().corr(grid.iloc[:, i + 1].rank()) for i in range(grid.shape[1] - 1)]

    by_week = {}
    periods = [(wk.strftime("%Y-%m-%d"), tier1["week"] == wk) for wk in weeks]
    periods.append((f"last_{len(weeks)}_weeks", tier1["week"].isin(weeks)))
    for key, mask in periods:
        w = tier1[mask]
        g = (w.groupby(["agent_id", "agent_name", "team", "site", "shift"])
              .agg(tickets_closed=("ticket_id", "count"), avg_csat=("csat_score", "mean"),
                   csat_n=("csat_score", "count"))
              .reset_index())
        g["team_median"] = g.groupby("team")["tickets_closed"].transform("median")
        g["vs_team_median"] = g["tickets_closed"] / g["team_median"]
        g = g.join(bounce, on="agent_id")
        by_week[key] = g.sort_values(["team", "tickets_closed"], ascending=[True, False]).to_dict("records")

    tier2 = d[d["team"] == TIER2_TEAM]
    tier2_summary = []
    for (aid, name), g in tier2.groupby(["agent_id", "agent_name"]):
        days = (g["resolved_at"] - g["created_at"]).dt.total_seconds() / 86400
        tier2_summary.append({
            "agent_id": aid, "agent_name": name, "cases_resolved": int(len(g)),
            "median_resolution_days": float(days.median()),
            "bounce_rate": float(bounce.loc[aid, "bounce_rate"]) if aid in bounce.index else None,
        })

    return {
        "weeks": [f"last_{len(weeks)}_weeks"] + [w.strftime("%Y-%m-%d") for w in reversed(weeks)],
        "week_to_week_rank_corr": float(np.nanmedian(rank_corr)) if rank_corr else None,
        "distinct_weekly_leaders": int(grid.idxmax().nunique()) if grid.size else 0,
        "avg_closed_per_agent_week": float(grid.values.mean()) if grid.size else 0.0,
        "by_week": by_week,
        "tier2_summary": tier2_summary,
        "desk_bounce_rate": float(b["bounced"].mean()),
    }


def build_goal(df: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    d = df.copy()
    d["quarter"] = d["created_at"].dt.to_period("Q").astype(str)
    q = (d.groupby("quarter")
          .agg(tickets=("ticket_id", "size"), chasers=("is_chaser", "sum"),
               chaser_cost=("service_cost_inr", lambda s: s[d.loc[s.index, "is_chaser"]].sum()))
          .reset_index())
    q["share"] = q["chasers"] / q["tickets"]
    base = q.iloc[-1]
    cost_per_chaser = base["chaser_cost"] / base["chasers"]
    avoided = base["chasers"] - GOAL_TARGET_SHARE * base["tickets"]
    d["week"] = week_start(d["created_at"])
    wk = d.groupby("week").agg(share=("is_chaser", "mean"), n=("ticket_id", "size")).reset_index()
    wk = wk[wk["week"] + pd.Timedelta(days=7) <= cutoff.normalize() + pd.Timedelta(days=1)].tail(26)
    by_issue = (d[d["quarter"] == base["quarter"]].loc[lambda x: x["is_chaser"]]
                .groupby("issue").size().sort_values(ascending=False))
    return {
        "metric": "Status-chaser share of tickets",
        "definition": "Tickets whose issue is: " + "; ".join(CHASER_ISSUES),
        "baseline_quarter": base["quarter"],
        "baseline_share": float(base["share"]),
        "baseline_chasers": int(base["chasers"]),
        "baseline_tickets": int(base["tickets"]),
        "target_share": GOAL_TARGET_SHARE,
        "tickets_avoided_per_quarter": int(round(avoided)),
        "service_cost_per_chaser_inr": float(cost_per_chaser),
        "value_per_quarter_inr": float(avoided * cost_per_chaser),
        "baseline_by_issue": by_issue.to_dict(),
        "quarterly": q.to_dict("records"),
        "weekly": [{"week": r.week.strftime("%Y-%m-%d"), "share": r.share, "tickets": int(r.n)}
                   for r in wk.itertuples()],
    }


def build_cost_summary(df: pd.DataFrame) -> dict:
    months = (df["created_at"].max() - df["created_at"].min()).days / 30.44
    per_year = 12 / months
    payouts = (df[df["refund_amount_inr"].notna()].groupby("refund_reason_code")["refund_amount_inr"]
               .agg(["count", "sum"]).sort_values("sum", ascending=False).reset_index())
    return {
        "months_covered": round(months, 1),
        "total_tickets": int(len(df)),
        "service_cost_per_year_inr": float(df["service_cost_inr"].sum() * per_year),
        "service_cost_components_per_year_inr": {
            k: float(df[c].sum() * per_year) for k, c in
            [("contacts", "contact_cost"), ("transfers", "transfer_cost"), ("sla_breach_credits", "breach_credit")]
        },
        "refunds_replacements_per_year_inr": float(df["refund_or_replacement_cost"].sum() * per_year),
        "refunds_by_reason": payouts.rename(columns={"count": "tickets", "sum": "inr"}).to_dict("records"),
        "replacements": {"tickets": int((df["replacement_issued"] == "Y").sum()),
                         "inr": float(df.loc[(df["replacement_issued"] == "Y") & df["refund_amount_inr"].isna(),
                                             "refund_or_replacement_cost"].sum())},
        "loose_repeat_rate": float(df["is_repeat_contact"].mean()),
        "same_issue_repeat_rate": float(df["is_same_issue_repeat"].mean()),
        "policy_violations_refund_and_replacement": df.loc[df["policy_violation_refund_and_replacement"],
                                                            "ticket_id"].tolist(),
        "service_cost_by_issue": (df.groupby("issue")["service_cost_inr"].sum() * per_year)
                                  .sort_values(ascending=False).round(0).to_dict(),
    }


def sanitize_for_json(obj):
    """NaN/NaT -> None: JSON.parse() in the browser rejects bare NaN."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.strftime("%Y-%m-%d %H:%M")
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["groq", "gemini"], default=None,
                        help="Classify recent tickets with an LLM (needs GEMINI_API_KEY / GROQ_API_KEY).")
    parser.add_argument("--llm-weeks", type=int, default=4, help="How many recent weeks go to the LLM.")
    args = parser.parse_args()

    print("Loading and cleaning data...")
    df = build_clean_dataset()
    cutoff = df["created_at"].max()
    print(f"  {len(df):,} tickets, export cut off at {cutoff:%Y-%m-%d %H:%M}")

    print("Classifying tickets" + (f" (LLM: {args.llm}, last {args.llm_weeks} weeks)" if args.llm else " (offline rules)") + "...")
    df = classify_all(df, args.llm, args.llm_weeks, cutoff)
    df = add_bounce_flag(df)
    stats = dict(llm_classify.STATS)
    if args.llm:
        print(f"  LLM: {stats['llm_ok']} live, {stats['llm_cached']} cached, "
              f"{stats['llm_failed_fell_back']} FAILED -> fell back to rules")
        for e in stats["errors"]:
            print("   ", e)

    payload = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data_cutoff": cutoff.strftime("%Y-%m-%d %H:%M"),
            "classifier": f"llm:{args.llm}:{llm_classify.MODELS[args.llm]} (last {args.llm_weeks} weeks) + rules"
                          if args.llm else "offline rules",
            "llm_stats": {k: v for k, v in stats.items() if k != "errors"} if args.llm else None,
        },
        "goal": build_goal(df, cutoff),
        "weekly_digest": build_weekly_digest(df, cutoff),
        "leaderboard": build_leaderboard(df, cutoff),
        "cost_summary": build_cost_summary(df),
    }
    payload = sanitize_for_json(payload)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False, allow_nan=False)

    g, c = payload["goal"], payload["cost_summary"]
    print(f"\nWrote {OUT_PATH}")
    print(f"Service cost: Rs {c['service_cost_per_year_inr']:,.0f}/yr  |  refunds+replacements: Rs {c['refunds_replacements_per_year_inr']:,.0f}/yr")
    print(f"Same-issue repeat rate: {c['same_issue_repeat_rate']:.1%} (loose definition: {c['loose_repeat_rate']:.1%})")
    print(f"GOAL: status chasers {g['baseline_share']:.1%} of tickets in {g['baseline_quarter']} -> {g['target_share']:.0%}: "
          f"~{g['tickets_avoided_per_quarter']} fewer tickets, ~Rs {g['value_per_quarter_inr']:,.0f} a quarter")


if __name__ == "__main__":
    main()
