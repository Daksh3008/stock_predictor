# helper to search tickers by company name >>>
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class TickerHit:
    symbol: str
    name: str
    exchange: str

def search_top5(company_query: str) -> List[TickerHit]:
    """
    Try yahooquery first (best results). If it fails (network, etc.), return [] and the caller
    can ask user to type a ticker directly.
    """
    hits: List[TickerHit] = []
    try:
        from yahooquery import search
        res = search(company_query)  # returns dict
        quotes = []
        if isinstance(res, dict):
            quotes = res.get("quotes", []) or []
        # collect unique by symbol
        seen = set()
        for q in quotes:
            sym = q.get("symbol")
            name = q.get("shortname") or q.get("longname") or q.get("name") or ""
            exch = q.get("exchDisp") or q.get("exchange") or ""
            if not sym or sym in seen:
                continue
            seen.add(sym)
            hits.append(TickerHit(symbol=sym, name=name, exchange=exch))
            if len(hits) >= 5:
                break
    except Exception:
        # fallback: no search available
        hits = []
    return hits
