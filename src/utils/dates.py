import pandas as pd
import re

def end_of_fiscal_quarter(dt: pd.Timestamp) -> pd.Timestamp:
    y, m = dt.year, dt.month
    if 4 <= m <= 6:   # Q1 (Apr-Jun)
        return pd.Timestamp(y, 6, 30)
    elif 7 <= m <= 9:  # Q2 (Jul-Sep)
        return pd.Timestamp(y, 9, 30)
    elif 10 <= m <= 12:  # Q3 (Oct-Dec)
        return pd.Timestamp(y, 12, 31)
    else:  # Jan-Mar Q4
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

def parse_explicit_days(text: str):
    m = re.search(r"next\s+(\d+)\s*(day|days)", text.lower())
    if m:
        return int(m.group(1))
    return None
