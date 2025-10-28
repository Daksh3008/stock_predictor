def pretty_card(ticker, curr_price, pred_price, pct_change, ci_low, ci_high,
                models_used, best_model_name, best_score, confidence, datapoints, horizon_label):
    arrow = "▲" if pct_change >= 0 else "▼"
    direction = "Upward" if pct_change >= 0 else "Downward"
    lines = []
    lines.append(f"📊 Price Prediction for {ticker}\n")
    if curr_price is not None:
        lines.append(f"Current Price: ${curr_price:,.2f}")
    lines.append(f"Predicted Price ({horizon_label}): ${pred_price:,.2f}")
    lines.append(f"Expected Change: {arrow} {abs(pct_change)*100:.2f}% ({direction})")
    if ci_low is not None and ci_high is not None:
        lines.append(f"95% Confidence Interval: ${ci_low:,.2f} - ${ci_high:,.2f}\n")
    else:
        lines.append("Confidence Interval: n/a\n")
    lines.append("Analysis Details:")
    lines.append(f"- Models Used: {models_used} (Prophet, XGBoost, Random Forest, Linear Regression, LSTM)")
    lines.append(f"- Best Performing Model: {best_model_name} (Score: {best_score:.2f})")
    lines.append(f"- Confidence: {confidence}")
    lines.append(f"- Data Points: {datapoints}")
    return "\n".join(lines)
