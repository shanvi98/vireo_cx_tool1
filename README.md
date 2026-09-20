# Vireo Support Digest

A weekly digest of what Vireo's customers complain about, an agent view that
won't mislead, and one business goal tracked every week. Built on 18 months
of Vireo support tickets (Jan 2025 – Jun 2026).

## The business goal

> **Cut "status-chaser" contacts from 27% of tickets to 18%, worth about ₹72,000 a quarter (₹2.9 lakh a year) in desk cost.**

A status chaser is a ticket whose only purpose is *where is my order / return
pickup / refund / repair?* In Q2 2026 that was 673 of 2,467 tickets (27.3%).
The share has sat at 23–27% every quarter for 18 months, so it isn't a blip.
Nothing is broken in these tickets. The customer just wasn't told, so
proactive status messages can take them out of the queue. That answers the
Finance question from the thread: *does this take contacts out of the queue,
and what does it save?*

- **Value:** 229 fewer tickets a quarter × ₹315 average desk cost per chaser
  ticket (contact + transfers + SLA credit, at the policy's per-channel rates)
  = **₹72,131 a quarter**.
- **Why 18%:** a one-third cut. This is a judgment call, not something the data
  proves. The data can show the size of the pool and the cost of each ticket,
  but not how many customers a status message would satisfy. I chose a target
  that stays worthwhile even if proactive messages only work half as well as
  hoped, and the dashboard tracks it weekly so the target can be reset after a
  quarter.
- **Honest scale:** the whole desk costs ₹26.6 lakh a year to run, so this is
  ~11% of desk cost. Refunds and replacements (₹54.7 lakh a year) are *not*
  desk cost. Most of that is customers' own money going back, and v1 wrongly
  added it in.

## Run it (clean machine)

Needs Python 3.10+.

```bash
pip install -r requirements.txt

python run_pipeline.py                 # offline: no key, no network, ~5 seconds
python -m http.server 8000 --directory web
# open http://localhost:8000   (must be served; opening index.html directly can't fetch the JSON)
```

With an LLM (recommended; this is how the shipped `web/dashboard_data.json` was built):

```bash
pip install google-genai
# macOS / Linux:       export GEMINI_API_KEY=...
# Windows PowerShell:  $env:GEMINI_API_KEY="..."
python run_pipeline.py --llm gemini                 # LLM reads the last 4 weeks; rules read the rest
python validation/validate_sample.py --llm gemini   # accuracy against the hand-labelled set
```

The figures above come from the LLM run. Offline mode gives 26.8% and about
₹68,500 a quarter; the gap is the classifier used on the last 4 weeks.

`cache/` holds the LLM answers already paid for, so re-running with
`--llm gemini` costs nothing until new tickets arrive. Groq also works
(`pip install groq`, `GROQ_API_KEY`, `--llm groq`), but only Gemini was
validated.

## What's here

```
data/                       the 5 source CSVs
pipeline/clean_and_join.py  cleaning, joins, cost model, repeat-contact flags
pipeline/llm_classify.py    issue / sentiment / hardware-defect extraction (LLM + offline rules)
run_pipeline.py             builds web/dashboard_data.json (goal, digest, agents, costs)
validation/gold_labels.csv  120 hand-labelled tickets (80 dev + 40 holdout)
validation/validate_sample.py  scores classifiers against them -> validation/results.md
web/index.html              the dashboard (static, no build step)
cache/                      LLM answers already paid for
memo.docx                   one-page memo to Priya Raman
submission-form.md          the submission form
```

## How do we know it's right, and how often is it wrong?

`validation/results.md` has the full table and every error. In summary:

| On the 40-ticket holdout | Issue right | Hardware faults found | False alarms |
|---|---|---|---|
| Intake bot's category tag | 80% (category level) | 7 of 7 | 0 |
| Offline keyword rules | 85% | 6 of 7 | 1 |
| **Gemini 2.5 Flash-Lite** | **100%** (40/40) | **7 of 7** | **1** |

- **How the labels were made:** for each ticket we read the full message and
  agent note and gave it one issue from the fixed list and a hardware-defect
  call (Y / N / U, where U = genuinely unclear, like "connection drops", and is
  excluded from defect scoring). The labels were drafted with an AI assistant
  and should be spot-checked by a person. This is stated here so nobody
  mistakes it for independent ground truth.
- **Dev vs holdout:** the keyword rules were written *after* reading the 80
  dev tickets, so their dev score (99%) is flattering. The 40 holdout tickets
  were labelled before the rules were run on them. One rule bug ("dis*charged
  twice*" matching "charged twice") was fixed after it appeared on a dev ticket;
  it also affected one holdout ticket. The LLM prompt never saw either split.
- **On all 120, the LLM makes 5 errors.** Four are issue mix-ups inside the
  same family (pairing vs dropping twice, pickup vs delivery, not charging vs
  one side dead). One is a false defect flag: a watch that went dark after a
  firmware update.
- **The tag is already good at "is this hardware?"** (97–100%), but wrong on
  about 1 ticket in 5 about *what* the problem is. That is why the digest uses
  the text. The digest's "hardware faults the tag missed" panel is where the
  text adds the most.
- **Not validated:** sentiment ("angry" share, the quotes panel) and the
  same-issue repeat definition. Both are heuristics, and the next thing to label.
- **Classifier v1 had a circular check:** its "0% core-issue inconsistency"
  was guaranteed, because the label was built from the category name. v1 also
  scored the defect flag against a proxy made from the same category tag. Both
  are gone.

## Decisions I made, and why

| Decision | Why |
|---|---|
| Deduplicate 653 re-imported tickets (keep the helpdesk copy) | Same ticket, two source systems; legacy copy's `resolved_at` is exactly 5.5 h behind. |
| Legacy `resolved_at` + 5:30 | Policy §9: legacy stores UTC. Before the fix, 2,263 tickets resolved before they were created; after, none. |
| Legacy CSAT 0 → blank | Policy §8: blank = no response. Legacy stored non-response as 0. |
| Separate desk cost from refunds/replacements | Returns, cancellations and duplicate-payment reversals are customers' own money, not the cost of running support. |
| "Same-issue repeat" = same customer + product within 30 days AND (same category OR the customer says it's a repeat) | Only 39% of the loose "same product" repeats shared a category. Loose: 27.3%. Same-issue: 11.1%. Both are shown. |
| Fixed issue list instead of free-text "core issue" | Free text can't be counted, so there was nothing to put in a weekly digest. |
| LLM only on the last 4 weeks, cached | A weekly digest only needs new tickets. Predictable bill (~₹5 a month), no "surprise per-ticket bill". |
| Monday-start weeks; partial weeks labelled; leaderboard uses complete weeks only | v1 used weeks that started on Tuesday and ranked agents on a 2-day stub week (35 tickets, "top" agent had 4). |
| Leaderboard: within team, 12-week default, plus "came back ≤30d" | An agent closes ~4.4 tickets a week, so weekly ranks are noise (week-to-week rank correlation 0.36; 5 different #1s in 12 weeks). The came-back rate checks for fast-but-bad closing. |
| Tier 2 never ranked on volume | Policy §6 and Neha's note in the thread. Shown as median days to resolve instead. |
| Checked who gets credit for each ticket | All 1,829 agent-signed notes match the credited agent, so attribution is sound. |

## What I left out, and why

- **A server, database or login.** Priya: "keep it simple, I don't need a
  platform." It's one weekly JSON file and a static page. The next step is a
  weekly scheduled job, not a rewrite.
- **LLM on all 18 months.** It would cost under ₹100 and would help the
  history view, but it isn't needed for a weekly digest. Rules cover history;
  their error rate is published above.
- **Lot-level defect claims.** A few Pulse 2 lots look high (PL2-2509-4: 42
  hardware tickets on 137 units vs a ~20% typical rate), but with 62 lots,
  some will look high by chance. Worth a watchlist, not a verdict.
- **SLA breach analysis.** Breach rates are flat, about 9%, across every
  hour, day and quarter. There's no lever in this data, and I couldn't check
  the SLA definitions against the policy PDF in this folder.
- **A sentiment gold set, and a check of the repeat-contact heuristic.** Next
  on the list (see above).

## Known limits

- Per-channel contact costs, the ₹305 transfer, ₹350 breach credit and ₹340
  replacement handling are taken as coded in `clean_and_join.py`, from
  `support-policy.pdf`. Re-check them against the PDF if it changes.
- Lot/order fallback join (for the 34% of tickets without an order id) picks
  the customer's latest order of that SKU, which can be the wrong lot for repeat buyers.
