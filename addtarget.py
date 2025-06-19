
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime


INPUT_CSV = "sentiment_metrics2.csv"


OUTPUT_CSV = "finalmetrics.csv"

TICKERS = ["AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA"]


START_DATE = "2015-01-01"
END_DATE   = "2017-06-12"


def fetch_eod_data(symbol, start_date, end_date):
    all_records = []
    limit = 1000
    offset = 0

    while True:
        params = {
            "access_key": MARKETSTACK_API_KEY,
            "symbols": symbol,
            "date_from": start_date,
            "date_to": end_date,
            "limit": limit,
            "offset": offset,
            "sort": "ASC"
        }
        resp = requests.get(MARKETSTACK_BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data or not data["data"]:
            break

        eod_list = data["data"]
        all_records.extend(eod_list)

        pagination = data.get("pagination", {})
        total_count = pagination.get("total", 0)
        count = pagination.get("count", 0)
        offset += count

        if offset >= total_count:
            break

    return all_records

def build_daily_df(eod_records):

    if not eod_records:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","symbol","date_dt"])

    df = pd.DataFrame(eod_records)
    #convert date
    df["date_dt"] = pd.to_datetime(df["date"]).dt.normalize()
    df["date"] = df["date_dt"].dt.strftime("%Y-%m-%d")

    keep_cols = ["symbol","date","open","high","low","close","volume","date_dt"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[keep_cols]
    df.sort_values("date_dt", inplace=True, ignore_index=True)
    return df

def compute_daily_price_change(daily_df):
    if daily_df.empty:
        return pd.DataFrame(columns=["date","daily_close","daily_change_pct"])

    daily_df = daily_df.copy().sort_values("date_dt")
    daily_df["prev_close"] = daily_df["close"].shift(1)
    daily_df["daily_change_pct"] = ((daily_df["close"] - daily_df["prev_close"]) / daily_df["prev_close"]) * 100

    # rename close -> daily_close
    daily_df.rename(columns={"close": "daily_close"}, inplace=True)

    return daily_df[["symbol", "date", "daily_close", "daily_change_pct"]]



def main():

    df_sent = pd.read_csv(INPUT_CSV)


    df_sent["DateOnly"] = pd.to_datetime(df_sent["DateOnly"], errors="coerce").dt.strftime("%Y-%m-%d")

    df_sent = df_sent[df_sent["Ticker"].isin(TICKERS)].copy()

    #fetch marketstack data to calculate target (pct change in stock price)
    frames = []
    for ticker in TICKERS:
        print(f"Fetching EOD data for {ticker} from {START_DATE} to {END_DATE}")
        eod_records = fetch_eod_data(ticker, START_DATE, END_DATE)

        if not eod_records:
            print(f"  No Marketstack data found for {ticker} in that range.")
            continue

        df_daily = build_daily_df(eod_records)
        df_change = compute_daily_price_change(df_daily)
        df_change["Ticker"] = ticker  # keep the same column name as in df_sent
        frames.append(df_change)


    if frames:
        df_price = pd.concat(frames, ignore_index=True)
    else:
        df_price = pd.DataFrame(columns=["Ticker","date","daily_close","daily_change_pct"])


    df_sent.rename(columns={"DateOnly": "date"}, inplace=True)


    df_final = pd.merge(
        df_sent,
        df_price,
        on=["Ticker", "date"],
        how="left"
    )

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote final data with daily price changes to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
