"""
clean_and_join.py
------------------
Loads the five Vireo export files and produces one clean, enriched ticket-level
DataFrame. Every transformation here exists because of a specific rule in
support-policy.pdf or a specific note in the Sameer/README/email-thread files —
see the comment above each block. Nothing here is a generic "clean the data"
pass; skipping any one of these will quietly corrupt a downstream number.

Run standalone to sanity-check the cleaning steps:
    python clean_and_join.py
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CONTACT_COST = {"chat": 210, "email": 260, "voice": 520, "social": 240}
SLA_TARGET_HOURS = {"chat": 15 / 60, "email": 8, "voice": 2, "social": 4}
TRANSFER_COST = 305
REPLACEMENT_HANDLING_COST = 340  # reverse pickup + forward shipping, policy §5
BREACH_CREDIT = 350
REPEAT_CONTACT_WINDOW_DAYS = 30
# Phrases customers use when they are coming back about something "resolved".
REPEAT_MARKERS = (
    r"said it was fixed|supposedly sorted|again the same|same (?:issue|problem) again|"
    r"already (?:raised|contacted|complained)|third time|second time|still not (?:fixed|resolved)|reopen"
)


def load_raw():
    tickets = pd.read_csv(
        os.path.join(DATA_DIR, "tickets.csv"),
        parse_dates=["created_at", "first_response_at", "resolved_at"],
    )
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    agents = pd.read_csv(os.path.join(DATA_DIR, "agents.csv"))
    return tickets, orders, products, customers, agents


def dedupe_reimported_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    """
    Sameer's note: 'the new helpdesk re-imported a chunk of [legacy tickets],
    so some ticket IDs will look odd.' Confirmed in data: 653 ticket_ids each
    appear twice, once with source_system='helpdesk' and once 'legacy_fd',
    with identical created_at but resolved_at differing by exactly 5.5 hours.
    These are NOT two separate contacts — they are one ticket re-imported.
    Keep the 'helpdesk' copy (correctly timestamped) and drop the legacy
    duplicate.
    """
    dupe_ids = tickets.loc[tickets.duplicated("ticket_id", keep=False), "ticket_id"].unique()
    drop_mask = tickets["ticket_id"].isin(dupe_ids) & (tickets["source_system"] == "legacy_fd")
    return tickets.loc[~drop_mask].copy()


def fix_legacy_timezone(tickets: pd.DataFrame) -> pd.DataFrame:
    """
    support-policy.pdf §9: 'Resolution timestamps for migrated tickets were
    reconstructed from the legacy event log, which stores UTC' while the
    helpdesk otherwise displays/exports IST. created_at and first_response_at
    are unaffected (verified: identical across the duplicate pairs above) —
    only resolved_at for source_system == 'legacy_fd' needs +5:30 to line up
    with IST, or every handle-time / weekly-bucket calculation involving a
    legacy ticket's resolution will be off by 5.5 hours.
    """
    t = tickets.copy()
    legacy = t["source_system"] == "legacy_fd"
    t.loc[legacy, "resolved_at"] = t.loc[legacy, "resolved_at"] + pd.Timedelta(hours=5.5)
    return t


def fix_legacy_csat(tickets: pd.DataFrame) -> pd.DataFrame:
    """
    support-policy.pdf §8: 'A blank score means no response and must be
    excluded from averages, not treated as zero.' The current helpdesk stores
    non-response as a blank (NaN), but legacy_fd rows store non-response as
    literal 0 (confirmed: legacy_fd has zero 0s among *response* scores, only
    among the ~48% with no reply). Left uncorrected, every legacy-era CSAT
    average is dragged down by fake zeros.
    """
    t = tickets.copy()
    legacy = t["source_system"] == "legacy_fd"
    t.loc[legacy, "csat_score"] = t.loc[legacy, "csat_score"].replace(0, np.nan)
    return t


def flag_policy_violations(tickets: pd.DataFrame) -> pd.DataFrame:
    """
    support-policy.pdf §5: 'In no case is a customer to receive both a refund
    and a replacement for the same order; where this happens in error it must
    be escalated to the Team Lead and Finance the same day.' A handful of
    tickets in the export violate this (both refund_amount_inr populated AND
    replacement_issued == 'Y'). We flag them rather than silently pick one, so
    the tool doesn't quietly under- or over-state cost, and so Ops can action
    the escalation the policy requires.
    """
    t = tickets.copy()
    t["policy_violation_refund_and_replacement"] = t["refund_amount_inr"].notna() & (
        t["replacement_issued"] == "Y"
    )
    return t


def join_lot_code(tickets: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """
    README: 'order_id ... Blank when the customer did not quote it.
    customer_id + product_sku is the fallback join.' ~34% of tickets have no
    order_id. We join primarily on order_id, and for the rest fall back to the
    customer's most recent order of that SKU. This is an approximation: a
    customer who ordered the same SKU twice will be matched to one order
    somewhat arbitrarily. Documented as a validation edge case, not hidden.
    """
    orders_small = orders[["order_id", "sku", "lot_code", "order_date"]].drop_duplicates("order_id")
    merged = tickets.merge(orders_small, on="order_id", how="left", suffixes=("", "_ord"))

    need_fallback = merged["order_id"].isna() | merged["lot_code"].isna()
    fallback_orders = (
        orders[["customer_id", "sku", "lot_code", "order_date"]]
        .sort_values("order_date")
        .drop_duplicates(["customer_id", "sku"], keep="last")
    )
    fb = merged.loc[need_fallback].drop(columns=["sku", "lot_code", "order_date"], errors="ignore").merge(
        fallback_orders,
        left_on=["customer_id", "product_sku"],
        right_on=["customer_id", "sku"],
        how="left",
    )
    merged.loc[need_fallback, "lot_code"] = fb["lot_code"].values
    merged.loc[need_fallback, "sku"] = fb["sku"].values
    return merged


def join_products(tickets: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    return tickets.merge(
        products[["sku", "product_name", "family", "unit_cost_inr", "retail_price_inr", "warranty_months"]],
        left_on="product_sku",
        right_on="sku",
        how="left",
        suffixes=("", "_prod"),
    )


def latest_agent_roster(agents: pd.DataFrame) -> pd.DataFrame:
    """
    support-policy.pdf §7: an agent who changes site/shift gets a new roster
    row but keeps the same agent_id, so an agent_id can have multiple rows.
    In this export no agent_id actually repeats, but we still resolve to the
    'current' row (to_date blank, i.e. still active) defensively so the code
    doesn't silently misattribute site/team/tier if a future export does have
    multiple rows per agent.
    """
    a = agents.copy()
    a["to_date"] = pd.to_datetime(a["to_date"])
    a["_active"] = a["to_date"].isna()
    a = a.sort_values(["agent_id", "_active", "from_date"])
    return a.groupby("agent_id", as_index=False).tail(1).drop(columns="_active")


def compute_cost_fields(tickets: pd.DataFrame) -> pd.DataFrame:
    """
    Cost model straight from support-policy.pdf §4-5:
      - contact_cost: per-channel fully-loaded cost (§4)
      - transfer_cost: Rs 305 x number of hand-offs (§4)
      - breach_credit: Rs 350 store credit if first response missed the
        channel's SLA target (§3) — this is an SLA-desk cost, NOT a hardware
        cost, and must not be attributed to "defect" spend in reporting.
      - refund_or_replacement_cost: refund_amount_inr if a refund was raised,
        else (unit_cost + Rs 340 handling) if a replacement was issued,
        else 0. Never both (see flag_policy_violations) — we take refund
        first if both are erroneously present, since it's the direct cash
        cost actually paid out.
    """
    t = tickets.copy()
    t["contact_cost"] = t["channel"].map(CONTACT_COST).fillna(0)
    t["transfer_cost"] = t["transfers"].fillna(0) * TRANSFER_COST

    fr_hours = (t["first_response_at"] - t["created_at"]).dt.total_seconds() / 3600
    target = t["channel"].map(SLA_TARGET_HOURS)
    t["sla_breached"] = (fr_hours > target).fillna(False)
    t["breach_credit"] = t["sla_breached"] * BREACH_CREDIT

    replacement_cost = t["unit_cost_inr"].fillna(0) + REPLACEMENT_HANDLING_COST
    t["refund_or_replacement_cost"] = np.where(
        t["refund_amount_inr"].notna(),
        t["refund_amount_inr"],
        np.where(t["replacement_issued"] == "Y", replacement_cost, 0.0),
    )
    # service_cost = what it costs to *run the desk* for this ticket.
    # refund_or_replacement_cost is money out the door, but most of it
    # (returns, cancellations, duplicate-payment reversals) is the customer's
    # own money going back, not a support cost. Kept separate so nobody quotes
    # refunds as "support cost".
    t["service_cost_inr"] = t["contact_cost"] + t["transfer_cost"] + t["breach_credit"]
    t["total_cost_inr"] = t["service_cost_inr"] + t["refund_or_replacement_cost"]
    return t


def flag_repeat_contacts(tickets: pd.DataFrame) -> pd.DataFrame:
    """
    support-policy.pdf §10: 'First-contact resolution: a ticket is resolved
    at first contact if the same customer does not contact again about the
    same issue within 30 days of resolution. A repeat contact within that
    window is costed at the contact cost of the channel used.'
    'Same issue' is approximated here as 'same customer + same product_sku'
    (agent_notes/category can drift on re-tag; product is the stable key we
    have). Documented as an approximation, not a guarantee.
    """
    t = tickets.sort_values(["customer_id", "product_sku", "created_at"]).copy()
    grp = t.groupby(["customer_id", "product_sku"])
    prev_resolved = grp["resolved_at"].shift(1)
    prev_category = grp["category"].shift(1)
    gap_days = (t["created_at"] - prev_resolved).dt.total_seconds() / 86400
    # Loose version: any contact by the same customer about the same product
    # within 30 days. Only 39% of these share the previous ticket's category
    # (address change after a billing query, etc.), so this is an UPPER BOUND,
    # not the "same issue" the policy means. Kept for transparency.
    t["is_repeat_contact"] = gap_days.notna() & gap_days.between(0, REPEAT_CONTACT_WINDOW_DAYS)
    # Same-issue version (the headline metric): within the window AND either
    # the same category as the previous ticket, or the customer's own words
    # say it's a repeat ("your agent said it was fixed", "again the same").
    t["says_repeat"] = t["customer_message"].fillna("").str.lower().str.contains(REPEAT_MARKERS, regex=True)
    t["is_same_issue_repeat"] = t["is_repeat_contact"] & ((t["category"] == prev_category) | t["says_repeat"])
    t["repeat_contact_cost"] = np.where(t["is_same_issue_repeat"], t["contact_cost"], 0.0)
    t["repeat_service_cost"] = np.where(t["is_same_issue_repeat"], t["service_cost_inr"], 0.0)
    return t


def flag_attendance(tickets: pd.DataFrame) -> pd.DataFrame:
    """
    support-policy.pdf §10: 'Attendance: any ticket in status resolved or
    closed.' §8: auto-closed (72h, no reply) tickets 'count as a completed
    attendance, and are surveyed' — so 'closed' must NOT be excluded from
    resolution-based reporting just because it sounds like a non-outcome.
    """
    t = tickets.copy()
    t["is_attendance"] = t["status"].isin(["resolved", "closed"])
    return t


def build_clean_dataset() -> pd.DataFrame:
    tickets, orders, products, customers, agents = load_raw()

    t = dedupe_reimported_tickets(tickets)
    t = fix_legacy_timezone(t)
    t = fix_legacy_csat(t)
    t = flag_policy_violations(t)
    t = join_lot_code(t, orders)
    t = join_products(t, products)
    t = compute_cost_fields(t)
    t = flag_repeat_contacts(t)
    t = flag_attendance(t)

    agents_latest = latest_agent_roster(agents)
    t = t.merge(
        agents_latest[["agent_id", "name", "site", "team", "shift", "tier"]],
        on="agent_id",
        how="left",
        suffixes=("", "_agent"),
    )
    t = t.rename(columns={"name": "agent_name"})

    return t


if __name__ == "__main__":
    df = build_clean_dataset()
    print(f"Clean dataset: {len(df):,} tickets (raw export had {pd.read_csv(os.path.join(DATA_DIR, 'tickets.csv')).shape[0]:,} rows incl. re-imported dupes)")
    print(f"Policy violations (refund + replacement both issued): {df['policy_violation_refund_and_replacement'].sum()}")
    print(f"Repeat contacts, loose (same customer+product, 30d): {df['is_repeat_contact'].mean():.1%}")
    print(f"Repeat contacts, same issue: {df['is_same_issue_repeat'].mean():.1%}  |  contact cost: Rs {df['repeat_contact_cost'].sum():,.0f}")
    print(f"Service cost (contacts + transfers + SLA credits): Rs {df['service_cost_inr'].sum():,.0f}")
    print(f"Refunds + replacements: Rs {df['refund_or_replacement_cost'].sum():,.0f}")
    print()
    print("Total cost by category:")
    print(df.groupby("category")["total_cost_inr"].sum().sort_values(ascending=False).round(0))
