# Ticker to download (S&P 500 index)
TICKER = "^GSPC"

# Raw download date range
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# Train / test split boundaries
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

# LSTM lookback window (number of past days)
LOOKBACK = 64

# LSTM hyperparameters
LSTM_CONFIG = {
    "units1": 64,
    "units2": 32,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
}
