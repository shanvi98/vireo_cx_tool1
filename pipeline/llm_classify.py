"""
llm_classify.py
----------------
Reads a ticket's customer_message + agent_notes and extracts:
    issue                      one label from a FIXED list (ISSUES below)
    customer_sentiment         positive | neutral | negative | angry
    potential_hardware_defect  true/false + a one-line reason

Why a fixed list (v2): v1 asked the model for a free-text "5-8 word core
issue". That reads nicely per ticket but can't be counted - "battery dies
fast", "poor battery backup" and "drains in 2 hrs" are three different strings
- so the weekly digest had nothing to aggregate. A closed list is what makes
"top complaints this week" possible.

Two paths:
1. LLM (Gemini or Groq) - real calls, cached per ticket on disk so a re-run
   never re-bills a ticket it has already classified.
2. classify_rule_based() - a transparent keyword classifier using the same
   issue list. Runs on every ticket for free and is the default, so the tool
   works with no key. Its accuracy is measured in validation/, not assumed.
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

USE_LLM = False
LLM_PROVIDER = "gemini"  # "gemini" or "groq"
MODELS = {"gemini": "gemini-2.5-flash-lite", "groq": "llama-3.1-8b-instant"}
PROMPT_VERSION = "v3"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")

# (issue label, category it belongs to, counts as a potential hardware defect)
ISSUES = [
    ("battery drains fast", "Charging & Battery", True),
    ("device or case not charging", "Charging & Battery", True),
    ("no sound / one side dead", "Audio Quality", True),
    ("distortion, static or crackling", "Audio Quality", True),
    ("microphone not working", "Audio Quality", True),
    ("physical part broken", "Warranty & Repair", True),
    ("repair or warranty claim follow-up", "Warranty & Repair", True),
    ("connection keeps dropping", "Connectivity", False),
    ("cannot pair or connect", "Connectivity", False),
    ("wifi / speaker setup fails", "Connectivity", False),
    ("firmware update stuck or failed", "App & Firmware", False),
    ("app crashes or will not open", "App & Firmware", False),
    ("cannot log in / OTP", "Account & Login", False),
    ("order not delivered or delayed", "Delivery & Shipping", False),
    ("wrong item delivered", "Delivery & Shipping", False),
    ("damaged in transit", "Delivery & Shipping", False),
    ("return pickup not done", "Returns & Refunds", False),
    ("refund not received", "Returns & Refunds", False),
    ("charged twice", "Billing & Payments", False),
    ("payment taken, no order", "Billing & Payments", False),
    ("coupon or discount not applied", "Billing & Payments", False),
    ("invoice / GST bill", "Billing & Payments", False),
    ("cancel order", "Other", False),
    ("change delivery address", "Other", False),
    ("product or compatibility question", "Product Enquiry", False),
    ("other", "Other", False),
]
ISSUE_CATEGORY = {i: c for i, c, _ in ISSUES}
ISSUE_IS_HARDWARE = {i: h for i, _, h in ISSUES}
SENTIMENTS = ["positive", "neutral", "negative", "angry"]

EXTRACTION_PROMPT = """You are a customer-support analyst for Vireo Audio (earbuds, headphones, speakers, \
smartwatches, sold in India). Read one support ticket and return ONLY a JSON object with exactly these keys:

{{
  "issue": "<exactly one label from the ISSUE LIST>",
  "customer_sentiment": "<one of: positive, neutral, negative, angry>",
  "potential_hardware_defect": <true or false>,
  "defect_reason": "<one short clause if true, else empty string>"
}}

ISSUE LIST (pick what the customer actually needs, not what the intake tag says; the tag is often wrong):
{issue_list}

potential_hardware_defect = true ONLY when the customer describes the unit itself malfunctioning:
battery drain, not charging, no/one-sided sound, distortion/static, dead mic, a part that broke, or an
ongoing repair/warranty claim for such a fault. It is FALSE for damage in transit, wrong item, pairing
or setup problems, app/firmware problems, delivery, billing, and questions.

Sentiment: angry = threats, insults, "last time I buy", public-shaming, demands to escalate;
negative = disappointed or frustrated; neutral = plain request; positive = thanks/praise.
Messages mix English and Hindi and contain typos - read for meaning.

Intake tag (may be wrong): {category}
Customer message:
\"\"\"{customer_message}\"\"\"
Agent's closing note:
\"\"\"{agent_notes}\"\"\"
"""


def format_ticket_for_llm(ticket_row) -> str:
    return EXTRACTION_PROMPT.format(
        issue_list="\n".join(f"- {i}" for i, _, _ in ISSUES),
        category=ticket_row.get("category", ""),
        customer_message=str(ticket_row.get("customer_message", ""))[:800],
        agent_notes=str(ticket_row.get("agent_notes", ""))[:400],
    )


def call_llm_groq(prompt: str) -> dict:
    """Requires: pip install groq; GROQ_API_KEY set."""
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=MODELS["groq"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


_gemini_client = None


def call_llm_gemini(prompt: str) -> dict:
    """Requires: pip install google-genai; GEMINI_API_KEY set."""
    global _gemini_client
    from google import genai

    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = _gemini_client.models.generate_content(
        model=MODELS["gemini"],
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0},
    )
    return json.loads(resp.text)


def _normalise(raw: dict) -> dict:
    """LLMs drift: coerce the answer onto the schema instead of trusting it."""
    issue = str(raw.get("issue", "other")).strip()
    match = next((i for i in ISSUE_CATEGORY if i.lower() == issue.lower()), None)
    sentiment = str(raw.get("customer_sentiment", "neutral")).strip().lower()
    defect = raw.get("potential_hardware_defect", False)
    if isinstance(defect, str):
        defect = defect.strip().lower() == "true"
    return {
        "issue": match or "other",
        "issue_off_list": None if match else issue[:60],
        "customer_sentiment": sentiment if sentiment in SENTIMENTS else "neutral",
        "potential_hardware_defect": bool(defect),
        "defect_reason": str(raw.get("defect_reason", ""))[:120] if defect else "",
    }


# ---------------------------------------------------------------------------
# Offline classifier - same issue list, keyword rules. Order matters: first
# hit wins, and the customer's message is searched before the agent's note.
# ---------------------------------------------------------------------------
_ISSUE_PATTERNS = [
    ("repair or warranty claim follow-up", r"\brma\d*|claim number|update on my repair|repair status|service cent"),
    ("damaged in transit", r"damm?aged|arri\w*d damaged|broken (box|package)|transit damage"),
    ("charged twice", r"two entries|\bcharged (two|twice|2)|double (charge|debit)|statement disagrees|paid once"),
    ("cancel order", r"cancel|ordered by mistake|don'?t ship"),
    ("wrong item delivered", r"wrong (product|item|colou?r|variant)|different colou?r|not what i ordered"),
    ("physical part broken", r"snapp|strap|hinge|cracked|crack in|screen has a crac|broke\b|broken"),
    ("battery drains fast", r"batt\w* (drain|life|backup)|drains?\b|barely lasts|charge it twice|maybe \d+ ?h|24 hours is nowhere"),
    ("device or case not charging", r"won'?t charge|wont charge|not charg|no charge|never gets the green|0 ?percent|paperweight|does ?n.?t charge"),
    ("no sound / one side dead", r"no sound|no audio|one side|left side silent|right (earbud|side)|left (earbud|side|bud)|silent"),
    ("distortion, static or crackling", r"distort|static|crackl|fry\w* sound|buzz|hiss"),
    ("microphone not working", r"\bmic\b|microphone|can'?t hear me|cannot hear me"),
    ("wifi / speaker setup fails", r"wi-?fi|2\.4 ?ghz|router"),
    ("connection keeps dropping", r"disconnect|drop|stutter|losing my phone|keeps losing|cuts out"),
    ("cannot pair or connect", r"\bpair|re-?pair|device list|not discoverable|bluetooth|\bconnect"),
    ("firmware update stuck or failed", r"firmware|progress bar|update (is )?stuck|update fail|stuck at \d+"),
    ("app crashes or will not open", r"\bapp\b.*(crash|white screen|loading|not open|won'?t open)|white screen|loading screen"),
    ("cannot log in / OTP", r"log ?in|otp|password|account locked"),
    ("payment taken, no order", r"no order|deducted|failed after i paid|nothing shows in my account|payment went through"),
    ("coupon or discount not applied", r"coupon|promo|discount|offer vanish|festive offer"),
    ("invoice / GST bill", r"invoice|gst|tax bill"),
    ("return pickup not done", r"pick ?up|waiting for your courier"),
    ("refund not received", r"refund (not|hasn|still)|money hasn|waiting for my refund|still waiting for (my )?refund"),
    ("change delivery address", r"address|pincode|flat number"),
    ("order not delivered or delayed", r"not (been )?deliver|haven'?t received|not received|stuck on shipped|tracking|courier marked|still waiting|where is my"),
    ("product or compatibility question", r"compatib|work w\w*|will the .* run|can i connect|does .* work|spec sheet|specs?\b"),
]
_ANGRY = ["pathetic", "worst", "scam", "fraud", "twitter", "consumer court", "last time i buy",
          "lat time i buy", "refund. now", "someone senior", "anyone alive", "unacceptable", "furious"]
_NEGATIVE = ["disappoint", "regret", "losing patience", "loosing patience", "poor quality", "not acceptable",
             "loyal customer", "pareshan", "frustrat", "said it was fixed", "supposedly sorted", "again"]
_POSITIVE = ["thank you for", "great support", "appreciate", "resolved quickly"]


def classify_rule_based(ticket_row) -> dict:
    msg = str(ticket_row.get("customer_message", "") or "").lower()
    note = str(ticket_row.get("agent_notes", "") or "").lower()

    issue, hit = "other", ""
    for text in (msg, note):
        for label, pat in _ISSUE_PATTERNS:
            m = re.search(pat, text)
            if m:
                issue, hit = label, m.group(0)
                break
        if issue != "other":
            break

    if any(k in msg for k in _ANGRY):
        sentiment = "angry"
    elif any(k in msg for k in _NEGATIVE):
        sentiment = "negative"
    elif any(k in msg for k in _POSITIVE):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    defect = ISSUE_IS_HARDWARE[issue]
    return {
        "issue": issue,
        "issue_off_list": None,
        "customer_sentiment": sentiment,
        "potential_hardware_defect": defect,
        "defect_reason": f"matched '{hit}'" if defect else "",
    }


# ---------------------------------------------------------------------------
# Batch driver with cache, retry and honest error accounting.
# ---------------------------------------------------------------------------
STATS = {"llm_ok": 0, "llm_cached": 0, "llm_failed_fell_back": 0, "errors": []}
_lock = threading.Lock()


def _cache_path() -> str:
    return os.path.join(CACHE_DIR, f"{LLM_PROVIDER}_{MODELS[LLM_PROVIDER]}_{PROMPT_VERSION}.json")


def _load_cache() -> dict:
    try:
        with open(_cache_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(), "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _call_with_retry(prompt: str, attempts: int = 4) -> dict:
    fn = call_llm_groq if LLM_PROVIDER == "groq" else call_llm_gemini
    for i in range(attempts):
        try:
            return fn(prompt)
        except Exception as exc:  # rate limit / 5xx / bad JSON
            transient = re.search(r"429|503|500|rate|quota|overload|unavailable|timeout", str(exc), re.I)
            if i == attempts - 1 or not transient:
                raise
            time.sleep(2 ** i * 2)


def classify_ticket(ticket_row, cache: dict = None) -> dict:
    if not USE_LLM:
        return classify_rule_based(ticket_row)

    key = str(ticket_row.get("ticket_id"))
    if cache is not None and key in cache:
        with _lock:
            STATS["llm_cached"] += 1
        return dict(cache[key], source="llm")
    try:
        result = _normalise(_call_with_retry(format_ticket_for_llm(ticket_row)))
        with _lock:
            STATS["llm_ok"] += 1
            if cache is not None:
                cache[key] = result
        return dict(result, source="llm")
    except Exception as exc:
        # Never let one bad ticket kill the batch - but never hide it either:
        # the fallback is counted and reported in the output and the dashboard.
        with _lock:
            STATS["llm_failed_fell_back"] += 1
            if len(STATS["errors"]) < 5:
                STATS["errors"].append(f"{key}: {str(exc)[:160]}")
        return dict(classify_rule_based(ticket_row), source="rule_fallback")


def classify_many(rows: list, workers: int = 8) -> list:
    """rows: list of dicts. Returns results in the same order."""
    if not USE_LLM:
        return [dict(classify_rule_based(r), source="rule") for r in rows]
    cache = _load_cache()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda r: classify_ticket(r, cache), rows))
    finally:
        _save_cache(cache)
