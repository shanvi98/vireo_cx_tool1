# Submission form: Vireo Audio support digest

> The original `submission-form.md` template from the pack was not available
> when this was written. This version answers everything the brief asks for.
> If the official template has different headings, copy the answers across.
> Fields marked **[FILL IN]** can only be answered by the candidate.

## 1. Candidate

- Name: **[Shanvi Kumari]**
- Email: **[shanvi298@gmail.com]**
- Time spent: **[2 hours]**

## 2. Links

- Code (zip / repo): `vireo_cx_tool.zip`, or **[https://github.com/shanvi98/vireo_cx_tool1]**
- Screen recording (≤3 min): **[https://drive.google.com/file/d/1TeHzAivs097J3oMoOkaxNOEJEf9zzZat/view?usp=drive_link]**
- Memo: `https://docs.google.com/document/d/1729l2YePkyyafs8IMqOEsEB9E071zq7Q/edit?usp=drive_link&ouid=101871661173997960492&rtpof=true&sd=true` 

## 3. The business goal

**Cut "status-chaser" contacts from 27% of tickets to 18%, worth about ₹72,000 a quarter (₹2.9 lakh a year) in desk cost.**

Status chasers are tickets whose only purpose is "where is my order / return
pickup / refund / repair?". In Q2 2026 there were 673 of 2,467 tickets
(27.3%), stable at 23–27% every quarter. Hitting 18% removes about 229
tickets a quarter at ₹315 average desk cost each. The 18% target (a one-third
cut) is a judgment call; the README explains why and how it's tracked weekly.

## 4. What I built

- A pipeline that cleans the export (653 re-imported duplicates, legacy UTC
  timestamps, legacy CSAT zeros) and classifies every ticket into a fixed list
  of 26 issues, with sentiment and a hardware-defect flag. Gemini 2.5
  Flash-Lite reads recent weeks; offline keyword rules read history and are
  the no-key fallback.
- A static dashboard: goal tracker, weekly digest (top issues vs last week,
  hardware faults the intake tag missed, customer quotes), an agent view and a
  cost breakdown.
- A hand-labelled validation set (120 tickets) and a script that scores every
  classifier against it.

Runs from the README on a clean machine: `pip install -r requirements.txt`,
`python run_pipeline.py`, `python -m http.server 8000 --directory web`.

## 5. How I know it works, and how often it doesn't

On a 40-ticket holdout: the LLM got the issue right on 40/40, found 7/7
hardware faults and raised 1 false alarm. The offline rules got 85%, and the
intake tag's category was right 80% of the time. Across all 120 labelled
tickets the LLM makes 5 errors, all listed in `validation/results.md`.
Sentiment and the repeat-contact definition are **not** validated yet.
The labels were drafted with an AI assistant and should be spot-checked by a
person.

## 6. Decisions I made where the brief was unclear

- **Leaderboard:** built as asked, but agents are compared only within their
  team, and the default view is 12 weeks. At ~4.4 closures per agent per
  week, weekly ranks are mostly noise (rank correlation 0.36 week to week).
  It adds a "came back within 30 days" column so fast-but-bad closing isn't
  rewarded. Tier 2 is never ranked on volume (policy §6).
- **Cost:** desk service cost (₹26.6L a year) is kept separate from refunds
  and replacements (₹54.7L a year), most of which is customers' own money.
- **Repeat contacts:** "same issue" means same customer and product within 30
  days, plus the same category or the customer saying it's a repeat. That
  gives 11.1%. The loose version (27.3%) is shown alongside.
- **LLM scope:** only recent weeks go to the model, and answers are cached,
  so the monthly bill is about ₹5 and predictable.
- The rest are in the README table "Decisions I made, and why".

## 7. What I left out, and why

A server/database (Priya asked for simple), the LLM on all 18 months (not
needed for a weekly digest), lot-level defect verdicts (too little data per
lot), SLA breach analysis (flat everywhere, no lever), and validation of
sentiment and repeat detection (next on the list).

## 8. AI tools: what I used, what it cost, what I threw away

| Tool | Used for | Cost |
|---|---|---|
| **[FILL IN: tool used for the first version]** | First version of the pipeline, dashboard and memo | **[FILL IN]** |
| Claude Code (Claude Opus 5) | Review of v1; bug fixes; classifier v2/v3; gold-set labelling draft; validation script; dashboard v2; README and memo rewrite | **[FILL IN from your plan/usage]** |
| Gemini 2.5 Flash-Lite (API) | Ticket classification: ~1,000 calls (764 pipeline + 240 validation) | ≈ US$0.10 (~540 input + ~60 output tokens per call) |

**What I threw away:**

- v1 prompt's free-text "core issue". It couldn't be counted, so I replaced it
  with a fixed list of 26 issues.
- v1 validation. It was circular: a 0% "inconsistency" was guaranteed, and the
  defect proxy was built from the same category tag.
- v1 headline numbers: "₹81 lakh support cost" (included refunds), the "27%
  repeat contacts" (any contact about the same product), and "Pulse 2 costs
  ₹3.3 lakh" (actually ₹33 lakh; the category figure had been used as the
  product total).
- v1 goal "cut Charging & Battery repeats 32%→27%, ₹30K/year". On
  recomputation it's worth about ₹9K a year, too small to lead with.
- `gemini-2.0-flash`. Google has retired it (404), so the shipped v1 LLM path
  could never have worked.
- The weekly-only leaderboard. It ranked a 2-day stub week and was mostly noise.
