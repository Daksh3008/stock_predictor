from yahooquery import search

def find_ticker(company_name: str):
    results = search(company_name)
    quotes = results.get("quotes")
    if not quotes:
        raise ValueError(f"No tickers found for '{company_name}'")

    print(f"\nTop results for '{company_name}':")
    for i, q in enumerate(quotes[:5], 1):
        print(f"{i}. {q.get('shortname', 'N/A')} ({q.get('symbol', 'N/A')}) - {q.get('exchange', 'N/A')}")

    choice = int(input("\nEnter the number of correct stock (1-5): "))
    return quotes[choice - 1]["symbol"]
