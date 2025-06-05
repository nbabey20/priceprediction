import pandas as pd

def calculate_sentiment_metrics(
    input_csv="Tweet_with_tickers.csv",
    output_csv="sentiment_metrics2.csv"
):

    df = pd.read_csv(input_csv)


    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")


    df["DateOnly"] = df["Date"].dt.date

    sentiment_map = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1,
    }
    df["numeric_sentiment"] = df["Label"].map(sentiment_map)


    grouped = df.groupby(["DateOnly", "Ticker"])

    #compute sentiment metrics
    results = grouped.agg(
        sum_positive=("Label", lambda x: (x == "Positive").sum()),
        sum_negative=("Label", lambda x: (x == "Negative").sum()),
        sum_neutral=("Label", lambda x: (x == "Neutral").sum()),
        total_tweets=("Label", "count"),
        avg_sent=("numeric_sentiment", "mean"),
        std_sent=("numeric_sentiment", "std"),
    )

    #calculate
    results["% Positive"] = results["sum_positive"] / results["total_tweets"] * 100
    results["% Negative"] = results["sum_negative"] / results["total_tweets"] * 100
    results["% Neutral"]  = results["sum_neutral"]  / results["total_tweets"] * 100

    results.rename(
        columns={
            "avg_sent": "Avg. Sentiment",
            "std_sent": "Sentiment Volatility",
        },
        inplace=True
    )

    results = results[
        [
            "Avg. Sentiment",
            "% Positive",
            "% Negative",
            "% Neutral",
            "Sentiment Volatility",
            "sum_positive",
            "sum_negative",
            "sum_neutral",
            "total_tweets",
        ]
    ]


    results.reset_index(inplace=True)

    results.to_csv(output_csv, index=False)

    print(f"Sentiment metrics saved to {output_csv}")


if __name__ == "__main__":
    calculate_sentiment_metrics()
