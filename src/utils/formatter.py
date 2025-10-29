#formatter file for fomatting the output presented to the user

# support immediate forecasts + current price >>>
def pretty_card(
    ticker, curr_price, pred_price, pct_change, ci_low, ci_high,
    models_used, best_model_name, best_score, confidence, datapoints, horizon_label,
    today_pred=None, next_day_pred=None,  # <<< NEW
    market_msg: str | None = None        # <<< NEW
):
    arrow = "▲" if pct_change >= 0 else "▼"
    direction = "Upward" if pct_change >= 0 else "Downward"

    lines = []
    # Debug/market line
    if market_msg:
        lines.append(market_msg + "\n")

    lines.append(f"📌 Current Price ({ticker}): ${curr_price:,.2f}\n")  # <<< NEW

    # Immediate forecasts
    if today_pred is not None:
        lines.append(f"📌 Predicted Close (Today): ${today_pred:,.2f}")
    if next_day_pred is not None:
        lines.append(f"📌 Predicted Close (Next Trading Day): ${next_day_pred:,.2f}")

    # Requested horizon
    lines.append(f"\n🎯 Predicted Price ({horizon_label}): ${pred_price:,.2f}")
    lines.append(f"Expected Change: {arrow} {abs(pct_change)*100:.2f}% ({direction})")
    if ci_low is not None and ci_high is not None:
        lines.append(f"95% Confidence Interval: ${ci_low:,.2f} - ${ci_high:,.2f}\n")
    else:
        lines.append("Confidence Interval: n/a\n")

    lines.append("📊 Analysis Details:")
    lines.append(f"- Models Used: {models_used} (Prophet, XGBoost, Random Forest, Linear Regression, LSTM)")
    lines.append(f"- Best Performing Model: {best_model_name} (Score: {best_score:.2f})")
    lines.append(f"- Confidence: {confidence}")
    lines.append(f"- Data Points: {datapoints}")
    return "\n".join(lines)
