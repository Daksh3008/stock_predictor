# NLP file for understanding user prompts on what they want to predict 
import re

# Priority patterns (first match wins)
PATTERNS = [
    (r"highest price|highest", ("High", "max")),
    (r"lowest price|lowest", ("Low", "min")),
    (r"closing price|close", ("Close", "last")),
    (r"open price|open", ("Open", "last")),
    (r"\bhigh\b", ("High", "last")),
    (r"\blow\b", ("Low", "last")),
]

# <<< NEW: horizon keywords detected here; concrete date resolving done in dates.py >>>
def parse_user_request(text: str):
    t = text.strip()
    low = t.lower()

    # target + aggregation
    target_col, aggregation = "Close", "last"
    for pat, (col, how) in PATTERNS:
        if re.search(pat, low):
            target_col, aggregation = col, how
            break

    # horizon flags
    horizon = {
        "today": any(k in low for k in ["today", "for today", "today's"]),
        "tomorrow": any(k in low for k in ["tomorrow", "next day", "next trading day"]),
        "week": "week" in low,
        "month": "month" in low,
        "quarter": "quarter" in low,
        "year": "year" in low,
        "explicit_days": _extract_days(low),
        "this_quarter": "this quarter" in low,
    }
    return {
        "target_col": target_col,
        "aggregation": aggregation,
        "horizon_flags": horizon,
        "raw": t,
    }

def _extract_days(txt: str):
    m = re.search(r"next\s+(\d+)\s*(day|days)", txt)
    if m:
        return int(m.group(1))
    return None
