# Project Proposal: Stock Return Prediction
**Author:** Kevin Couppoussamy

## Research Question
Can an LSTM model, trained on a curated set of highly informative technical features, accurately predict both the magnitude (Regression) and directional sign (Classification) of the next-day stock return, achieving superior performance compared to a standard baseline model?

## Goal
To build and evaluate a Multi-Variate Long Short-Term Memory (LSTM) network to forecast the daily financial movements of a single, major stock index: S&P 500.

---

## General Process

### 1. Data Acquisition
* **Objective:** Download 7 years (2018 to 2024) of historical OHLCV (Open, High, Low, Close, Volume) data for the S&P 500.
* **Tool:** `yfinance`

### 2. Feature Engineering
* **Objective:** Create 12 diverse technical indicators (TIs).
* **List of Indicators:** RSI, MACD, MACD_H, BBL, BBP, SMA_50, EMA_20, OBV, Close Price (normalized), Daily Volume (normalized), Lagged Log Return (t-1), ATR.
* **Targets:**
    1.  Next-Day Percentage Return (Regression).
    2.  Next-Day Direction (Up/Down Binary, Classification).
* **Tool:** `pandas-ta` OR `ta`

### 3. Preprocessing
* **Data Cleaning:** Delete days with NaN.
* **Outlier Handling:**
    * Calculate average ($\mu$) and standard deviation ($\sigma$) for returns.
    * *Superior limit* = Average + 3 * (Standard deviation).
    * *Inferior limit* = Average - 3 * (Standard deviation).
    * **Logic:** If Return > Superior limit, replace with Superior limit. If Return < Inferior limit, replace with Inferior limit.
* **Binary Classification:**
    * Return > 0 then Up (1).
    * Return < 0 then Down (0).
* **Scaling:** Use `MinMaxScaler` to scale all input features for uniform and proportional data influence.
* **Reshaping:** Convert final DataFrame into a 3D array (`[samples, time steps, features]`).
    * *Target Array Dimensions:* `[1700, 64, 12]`
    * *Breakdown:* 1764 trading days (approx. 7 years) minus 64 days look-back window = 1700 days.
* **Tools:** `sklearn.preprocessing.MinMaxScaler` and `numpy` reshaping.

### 4. Model Implementation
* **Architecture:** Build a Sequential model with 2-3 Stacked LSTM layers to enhance predictive power.
* **Output Layer:** Dense output layer with linear activation for return magnitude prediction.
* **Directional Sign Logic:** If predicted percentage return > 0, stock is predicted Up; else predicted Down.
* **Tool:** `tensorflow.keras`
* **Hyperparameter Tuning & Prevention of Overfitting:**
    * **LSTM:** Learning rate, Batch size, Epochs, Hidden layers, Neurons per layer, Dropout rate.
    * **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`.
    * **XGBoost:** `n_estimators`, `learning_rate`, `max_depth`, `gamma`.
    * **Linear Regression:** Classification threshold (test values around 0 to maximize F1-Score/Alpha).
* **Tuning Strategy:**
    * *Training Set:* 01/01/2018–31/12/2021 (4 years) to teach the model.
    * *Validation Set:* 01/01/2022–31/12/2022 (1 year) to classify hyperparameters based on F1-score.
    * *Final Step:* Take the best combination and train the final model on the complete set (Training + Validation).

### 5. Training
* **Training Period:** 01/01/2018 to 31/12/2022 (5 years).
* **Testing Period:** 01/01/2023 to 31/12/2024 (2 years).

### 6. Evaluation
* **Baseline Models:** Linear baseline, Random Forest, and XGBoost.
* **Metrics:**
    * **Magnitude:** RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
    * **Direction:** Accuracy and F1-Score.
* **Analysis:** Construct Confusion Matrix for LSTM and all baselines.
* **Tools:** `tensorflow.keras`, `scikit-learn`, `xgboost` (`mean_squared_error`, `mean_absolute_error`, `accuracy_score`, `f1_score`).

---

## Detailed Feature List (LSTM Inputs)

**1. RSI (Relative Strength Index)**
* *Code:* `data.ta.rsi(append=True)`
* *Logic:* Measures speed and change of price movements (standard period: 14). Crucial for overbought/oversold signals.

**2. MACD (Line)**
* *Code:* `data.ta.macd(append=True)`
* *Logic:* Difference between 26-period EMA and 12-period EMA. Indicates trend direction and momentum.

**3. MACD_H (Histogram)**
* *Code:* Included in `data.ta.macd` output.
* *Logic:* Difference between MACD Line and Signal Line. Measures trend acceleration.

**4. BBL (Bollinger Band Lower)**
* *Code:* Included in `data.ta.bbands` output.
* *Logic:* Lower band value. Shows potential support/oversold levels.

**5. BBP (Bollinger Band Percent B)**
* *Code:* Included in `data.ta.bbands` output.
* *Logic:* Shows current price location relative to top/bottom bands (0 to 1). Highly effective normalized volatility input.

**6. SMA_50 (Simple Moving Average)**
* *Code:* `data.ta.sma(length=50, append=True)`
* *Logic:* Average price over last 50 days. Captures medium-term trend.

**7. EMA_20 (Exponential Moving Average)**
* *Code:* `data.ta.ema(length=20, append=True)`
* *Logic:* Fast-reacting average (20 days). Used to track short-term trend shifts.

**8. OBV (On-Balance Volume)**
* *Code:* `data.ta.obv(append=True)`
* *Logic:* Cumulative measure of buying/selling pressure. Confirms price trends.

**9. Close Price (Normalized)**
* *Code:* Directly from `yfinance` output.
* *Logic:* Normalized closing price. Provides direct price level context.

**10. Daily Volume (Normalized)**
* *Code:* Directly from `yfinance` output.
* *Logic:* Raw trading volume. High-value input showing market conviction.

**11. Lagged Log Return (t-1)**
* *Code:* "Manual calculation" with numpy.
* *Logic:* Log return from the previous day. Statistically the simplest, most powerful predictor.

**12. ATR (Average True Range)**
* *Code:* `data.ta.atr(append=True)`
* *Logic:* Measures market volatility based on daily price range. Useful for risk assessment.