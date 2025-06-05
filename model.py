import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

INPUT_CSV = "FINAL_DATA.csv"  # The file with columns:

TARGET_TICKER = None  

DATE_COLUMN = "date"
TICKER_COLUMN = "Ticker"

# Features (sentiment + tweet metrics)
FEATURE_COLUMNS = [
    "Avg. Sentiment",
    "% Positive",
    "% Negative",
    "% Neutral",
    "Sentiment Volatility",
    "sum_positive",
    "sum_negative",
    "sum_neutral",
    "total_tweets"
]


TARGET_COLUMN = "daily_change_pct"

#phase space reduction embedding dimension and delay interval
M = 3   
TAU = 1

#training params
EPOCHS = 50
BATCH_SIZE = 32
TEST_SPLIT = 0.1  


def filter_ticker(df, ticker):
    if ticker is None:
        return df
    return df[df[TICKER_COLUMN] == ticker].copy()

#vectorizes data for time series prediction
def create_psr_vectors(df, feature_cols, target_col, m=3, tau=1):
    #X(i) = [features(i), features(i+tau), ... features(i+(m-1)*tau)]
    data_points = []
    max_index = len(df) - m * tau
    for i in range(max_index):
        x_t = []
        for j in range(m):
            idx = i + j * tau
            row_features = df.iloc[idx][feature_cols].values
            x_t.extend(row_features)
        y_t = df.iloc[i + (m * tau)][target_col]
        data_points.append((x_t, y_t))
    X = np.array([x for x, _ in data_points])
    y = np.array([y for _, y in data_points])
    return X, y

def main():
    #load
    df = pd.read_csv(INPUT_CSV)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    
    if TARGET_TICKER is not None:
        tickers = [TARGET_TICKER]
    else:
        tickers = df[TICKER_COLUMN].unique()
    
    results = {}
    
    #iterate for each ticker
    for ticker in tickers:
        print(f"\n=== Running predictions for ticker: {ticker} ===")
        df_ticker = filter_ticker(df, ticker)
        
        #sort and reset
        df_ticker.sort_values(by=[DATE_COLUMN], inplace=True)
        df_ticker.reset_index(drop=True, inplace=True)
        
        #data cleaning
        df_ticker.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN], inplace=True)
        

        if len(df_ticker) < (M * TAU + 1):
            print(f"Not enough data for ticker: {ticker}. Skipping.")
            continue
        
        #scale features and target
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        df_ticker[FEATURE_COLUMNS] = feature_scaler.fit_transform(df_ticker[FEATURE_COLUMNS])
        df_ticker[TARGET_COLUMN] = target_scaler.fit_transform(df_ticker[[TARGET_COLUMN]])
        
        #create psr vectors
        X_psr, y_psr = create_psr_vectors(df_ticker, FEATURE_COLUMNS, TARGET_COLUMN, m=M, tau=TAU)
        if len(X_psr) == 0:
            print(f"Not enough PSR vectors for ticker: {ticker}. Skipping.")
            continue
        
        #train-testsplit
        split_idx = int(len(X_psr) * (1 - TEST_SPLIT))
        X_train, y_train = X_psr[:split_idx], y_psr[:split_idx]
        X_test, y_test = X_psr[split_idx:], y_psr[split_idx:]
        
        #reshape for LTSM
        num_features = len(FEATURE_COLUMNS)
        X_train_lstm = X_train.reshape(X_train.shape[0], M, num_features)
        X_test_lstm = X_test.reshape(X_test.shape[0], M, num_features)
        
        #LSTM tool
        model = Sequential()
        model.add(LSTM(16, return_sequences=True, activation='tanh', input_shape=(M, num_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(16, activation='tanh'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        
        model.summary()
        
        #10% for training validation
        history = model.fit(
            X_train_lstm, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=1
        )
        
        #predictions on test
        y_pred_scaled = model.predict(X_test_lstm)
        
        #inverse transform predictions and test targets
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        y_pred = y_pred.ravel()
        y_test_inv = y_test_inv.ravel()
        
        #metrics
        mse = mean_squared_error(y_test_inv, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred)
        r2 = r2_score(y_test_inv, y_pred)
        
        print(f"Test metrics for {ticker}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")
        
        #results
        results[ticker] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        
        #plot actual vs predicted graph for analysis
        plt.figure(figsize=(10,6))
        plt.plot(y_test_inv, label='Actual % Change', marker='o')
        plt.plot(y_pred, label='Predicted % Change', marker='x')
        plt.title(f"LSTM Prediction for {ticker}")
        plt.xlabel('Time Step (Test Set)')
        plt.ylabel('Daily % Change')
        plt.ylim(-10, 10)
        plt.legend()
        plt.show()
    
    #summary results for all tickers
    print("\n=== Summary of Test Metrics for All Tickers ===")
    for ticker, metrics in results.items():
        print(f"{ticker}: RMSE = {metrics['RMSE']:.4f}, MAE = {metrics['MAE']:.4f}, R2 = {metrics['R2']:.4f}")

if __name__ == "__main__":
    main()
