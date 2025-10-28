import re

# Ordered list — priority matters!
PATTERNS = [
    (r"highest price|highest", ("High", "max")),
    (r"lowest price|lowest", ("Low", "min")),
    (r"closing price|close", ("Close", "last")),
    (r"open price|open", ("Open", "last")),
]

def parse_user_query(text: str):
    t = text.strip()
    # Ticker detection
    m = re.search(r"\b([A-Z\^][A-Z0-9\.\-]{1,9})\b", t)
    ticker = m.group(1) if m else None

    target_col = "Close"
    aggregation = "last"
    txt = t.lower()

    for pat, (col, agg) in PATTERNS:
        if re.search(pat, txt):
            target_col, aggregation = col, agg
            break

    return {
        "ticker": ticker,
        "target_col": target_col,
        "aggregation": aggregation,
        "raw": t
    }
