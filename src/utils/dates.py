#dates file to understand date manipulations  of quarter, weeks, months, years

# calendar utilities including Indian fiscal quarters and short horizons >>>
import pandas as pd

def end_of_fiscal_quarter(dt: pd.Timestamp) -> pd.Timestamp:
    y, m = dt.year, dt.month
    if 4 <= m <= 6:   # Q1 (Apr–Jun)
        return pd.Timestamp(y, 6, 30)
    elif 7 <= m <= 9: # Q2 (Jul–Sep)
        return pd.Timestamp(y, 9, 30)
    elif 10 <= m <= 12: # Q3 (Oct–Dec)
        return pd.Timestamp(y, 12, 31)
    else:             # Q4 (Jan–Mar)
        return pd.Timestamp(y, 3, 31)

def end_of_next_fiscal_quarter(dt: pd.Timestamp) -> pd.Timestamp:
    curr_end = end_of_fiscal_quarter(dt)
    next_ref = curr_end + pd.Timedelta(days=1)
    return end_of_fiscal_quarter(next_ref)

def end_of_next_month(dt: pd.Timestamp) -> pd.Timestamp:
    d = dt + pd.DateOffset(months=1)
    return (pd.Timestamp(year=d.year, month=d.month, day=1) + pd.offsets.MonthEnd(0))

def end_of_next_year(dt: pd.Timestamp) -> pd.Timestamp:
    return dt + pd.DateOffset(years=1)

def end_of_next_week(dt: pd.Timestamp) -> pd.Timestamp:
    return dt + pd.Timedelta(days=7)

# <<< NEW: resolve steps from flags; always daily forecasting >>>
def resolve_steps_from_flags(flags: dict, last_date: pd.Timestamp) -> tuple[int, str]:
    # explicit days override
    if flags.get("explicit_days"):
        n = int(flags["explicit_days"])
        return max(1, n), f"next_{n}_days"

    if flags.get("today"):
        return 1, "today"  # we treat "today" as next step (CP1 current price still shown)
    if flags.get("tomorrow"):
        return 2, "tomorrow"  # we’ll also print today & tomorrow separately anyway

    if flags.get("this_quarter"):
        td = end_of_fiscal_quarter(last_date)
        return max(1, (td - last_date).days), "this_quarter"
    if flags.get("quarter"):
        td = end_of_next_fiscal_quarter(last_date)
        return max(1, (td - last_date).days), "next_quarter"
    if flags.get("year"):
        td = end_of_next_year(last_date)
        return max(1, (td - last_date).days), "next_year"
    if flags.get("month"):
        td = end_of_next_month(last_date)
        return max(1, (td - last_date).days), "next_month"
    if flags.get("week"):
        td = end_of_next_week(last_date)
        return max(1, (td - last_date).days), "next_week"

    # default fallback (rare)
    return 30, "next_30_days"

# <<< NEW: simple market-open awareness (for message only) >>>
def market_closed_today_message(df_index: pd.DatetimeIndex, today: pd.Timestamp) -> str | None:
    if today.normalize() not in df_index.normalize():
        return "ℹ️ Market is closed today. Showing forecast instead."
    return None
