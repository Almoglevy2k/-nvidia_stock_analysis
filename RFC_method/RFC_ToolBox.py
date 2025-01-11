# RFCToolBox.py
import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def download_stock_data(ticker: str) -> yf.Ticker:
    """
    Creates a yfinance Ticker object for the given stock symbol.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g., "AAPL", "NVDA").

    Returns
    -------
    yf.Ticker
        A yfinance Ticker object, which can be used to fetch historical data.
    """
    return yf.Ticker(ticker)

def plot_data_by_date(ax: plt.Axes, data: pd.DataFrame, label: str) -> None:
    """
    Plots the 'Close' price from a pandas DataFrame on a given Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The Axes object on which to draw the line plot.
    data : pd.DataFrame
        The DataFrame containing stock data with a 'Close' column.
    label : str
        A label to use for the plot legend and title.
    """
    data.plot.line(y="Close", use_index=True, ax=ax, label=label)
    ax.set_title(f"{label} Closing Prices")
    ax.set_ylabel("Price")
    ax.legend()

def remove_unused_columns(data: pd.DataFrame, col_list: list) -> None:
    """
    Drops the specified columns from the DataFrame and removes any rows with NaN values.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to modify.
    col_list : list
        A list of column names to remove from the DataFrame.

    Returns
    -------
    None
        This function mutates the DataFrame in place.
    """
    data.drop(columns=col_list, inplace=True)
    data.dropna(inplace=True)

def add_tomorrow_column(data: pd.DataFrame) -> None:
    """
    Adds 'Tommorow' and 'Target' columns to the DataFrame.
    'Tommorow' is the shifted 'Close' by 1 day.
    'Target' is 1 if tomorrow's Close is higher than today's Close, else 0.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to which columns will be added.

    Returns
    -------
    None
        This function mutates the DataFrame in place.
    """
    data["Tommorow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tommorow"] > data["Close"]).astype(int)

def remove_old_data(data: pd.DataFrame, date: str) -> None:
    """
    Removes all rows whose index is older than the specified date.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to filter.
    date : str
        A date string (YYYY-MM-DD). Rows with indices before this date are dropped.

    Returns
    -------
    None
        This function mutates the DataFrame in place.
    """
    mask = data.index < date
    data.drop(index=data[mask].index, inplace=True)

def add_statistics(data: pd.DataFrame) -> None:
    """
    Adds rolling statistics for multiple horizons (e.g. 1, 5, 20, 60, 250, 1000).
    For each horizon:
      - Adds a 'Close_Ratio_{horizon}' column: data["Close"] / rolling_average["Close"]
      - Adds a 'Trend_{horizon}' column: sum of 'Target' over the last 'horizon' days

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to which new columns will be added.

    Returns
    -------
    None
        This function mutates the DataFrame in place.
    """
    horizons = [1, 5, 20, 60, 250, 1000]
    for horizon in horizons:
        rolling_average = data.rolling(window=horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_average["Close"]

        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]

    data.dropna(inplace=True)

def data_preprocessing(data: pd.DataFrame) -> None:
    """
    Performs all necessary preprocessing steps in sequence:
    1. Removes unused columns (Dividends, Stock Splits)
    2. Adds 'Tommorow' and 'Target' columns
    3. Removes rows older than 1990-01-01
    4. Adds rolling statistics

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing raw stock data.

    Returns
    -------
    None
        This function mutates the DataFrame in place.
    """
    remove_unused_columns(data, ["Dividends", "Stock Splits"])
    add_tomorrow_column(data)
    remove_old_data(data, "1990-01-01")
    add_statistics(data)

def define_model() -> RandomForestClassifier:
    """
    Creates and returns a Random Forest Classifier with fixed hyperparameters.

    Returns
    -------
    RandomForestClassifier
        A configured RandomForestClassifier instance.
    """
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_split=50,
        random_state=1
    )

def get_predictions(model: RandomForestClassifier, 
                    test: pd.DataFrame, 
                    predictors: list, 
                    threshold: float) -> pd.Series:
    """
    Generates predictions for the test set using a custom probability threshold.

    Parameters
    ----------
    model : RandomForestClassifier
        The trained Random Forest model.
    test : pd.DataFrame
        The test subset of data on which we predict.
    predictors : list
        A list of column names used as features.
    threshold : float
        Probability cutoff. If predicted probability >= threshold => 1, else 0.

    Returns
    -------
    pd.Series
        A Series of predictions (0 or 1) indexed like the test data.
    """
    preds_proba = model.predict_proba(test[predictors])[:, 1]
    preds = np.where(preds_proba >= threshold, 1, 0)
    return pd.Series(preds, index=test.index, name="Predictions")

def predictor(train: pd.DataFrame, 
              test: pd.DataFrame, 
              predictors: list, 
              model: RandomForestClassifier, 
              threshold: float) -> pd.DataFrame:
    """
    Trains the given model on 'train' data and predicts on 'test' data using a specified threshold.

    Parameters
    ----------
    train : pd.DataFrame
        The training subset of data (features + target).
    test : pd.DataFrame
        The test subset of data (features + target).
    predictors : list
        Column names used as features for training and prediction.
    model : RandomForestClassifier
        The model to be fit and used for prediction.
    threshold : float
        Probability cutoff for generating binary predictions.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: ['Target', 'Predictions'],
        indexed the same as 'test'.
    """
    model.fit(train[predictors], train["Target"])
    preds = get_predictions(model, test, predictors, threshold)
    combined = pd.concat([test["Target"], preds], axis=1)
    combined.columns = ["Target", "Predictions"]
    return combined

def backtest(data: pd.DataFrame, 
             model: RandomForestClassifier, 
             predictors: list, 
             threshold: float, 
             start: int = 500, 
             step: int = 250) -> pd.DataFrame:
    """
    Backtests the model by incrementally training on [0 : i] rows and testing on [i : i+step] rows.
    For each segment, predictions are appended, allowing analysis across the entire dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The entire dataset (features + target) sorted by date.
    model : RandomForestClassifier
        The (untrained) model to use for training/prediction.
    predictors : list
        Column names used as features for training and prediction.
    threshold : float
        Probability cutoff for generating binary predictions.
    start : int, optional
        Row index at which to start the incremental tests. Default is 500.
    step : int, optional
        Number of rows to use in each test segment. Default is 250.

    Returns
    -------
    pd.DataFrame
        A single DataFrame with columns ['Target', 'Predictions'],
        containing predictions for all test segments. The index is time-based (DatetimeIndex).
    """
    all_predictions = []
    total_steps = (data.shape[0] - start) // step + 1
    current_step = 0

    for i in range(start, data.shape[0], step):
        current_step += 1
        sys.stdout.write(f"\rProcessing batch {current_step}/{total_steps}...")
        sys.stdout.flush()

        train = data.iloc[:i].copy()
        test = data.iloc[i : i + step].copy()

        predictions = predictor(train, test, predictors, model, threshold)
        all_predictions.append(predictions)

    sys.stdout.write("\nBacktesting complete.\n")

    return pd.concat(all_predictions, axis=0)

def daily_trading_strategy(data: pd.DataFrame,
                           ax: plt.Axes,
                           start_wallet: float = 10000.0,
                           start_date: str = "2023-01-01",
                           end_date: str = "2024-01-01") -> tuple:
    """
    Simulates a daily trading strategy on a date-filtered portion of data.
    For each day:
      - If 'Predictions' == 1 => buy at 'Open' price, then sell at 'Close' same day.
      - Otherwise, do nothing (remain in cash).
    
    Parameters
    ----------
    data : pd.DataFrame
        The stock data including 'Open', 'Close', 'Target', and 'Predictions' columns.
    ax : matplotlib.pyplot.Axes
        The Axes object to plot the wallet value over time.
    start_wallet : float, optional
        Initial wallet amount in USD. Default is 10000.0.
    start_date : str, optional
        Start date for the simulation in YYYY-MM-DD format. Default is "2023-01-01".
    end_date : str, optional
        End date for the simulation in YYYY-MM-DD format. Default is "2024-01-01".

    Returns
    -------
    tuple
        A tuple of (results, accuracy, final_wallet):
          - results (pd.DataFrame): wallet values over time, indexed by Date.
          - accuracy (float): proportion of 'up' predictions that were actually correct.
          - final_wallet (float): wallet amount at the end of the simulation period.
    """
    data = data.sort_index()
    filtered_data = data.loc[start_date:end_date].copy()

    required_cols = ["Open", "Close", "Target", "Predictions"]
    for col in required_cols:
        if col not in filtered_data.columns:
            raise ValueError(f"Missing required column '{col}' in data!")

    wallet = start_wallet
    wallet_values = []

    for current_date, row in filtered_data.iterrows():
        # Record wallet at the start of the day
        wallet_values.append((current_date, wallet))

        if row["Predictions"] == 1:
            open_price = row["Open"]
            close_price = row["Close"]
            if open_price <= 0:
                continue
            shares = wallet / open_price
            wallet = shares * close_price

    if not filtered_data.empty:
        last_date = filtered_data.index[-1]
        wallet_values.append((last_date, wallet))

    predicted_up_mask = (filtered_data["Predictions"] == 1)
    correct_mask = (predicted_up_mask & (filtered_data["Target"] == 1))
    total_pred_up = predicted_up_mask.sum()
    correct_pred_up = correct_mask.sum()
    accuracy = correct_pred_up / total_pred_up if total_pred_up else 0.0

    results = pd.DataFrame(wallet_values, columns=["Date", "Wallet"])
    results.set_index("Date", inplace=True)

    # Plot on the provided Axes
    ax.plot(results.index, results["Wallet"], label="Wallet Value", color="tab:orange")
    ax.set_title("Daily Trading Strategy Wallet Value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wallet Value (USD)")
    ax.legend()
    ax.grid(True)

    return results, accuracy, wallet

def run_RFC_sim(ticker: str = "NVDA", threshold: float = 0.52) -> None:
    """
    Main driver function to run the Random Forest Classifier simulation workflow:
      1. Download historical data and preprocess
      2. Train/backtest with rolling windows
      3. Evaluate model performance
      4. Simulate daily trading
      5. Print results and show plots

    Parameters
    ----------
    ticker : str, optional
        Stock ticker symbol to download (default: "NVDA").
    threshold : float, optional
        Probability threshold for classification (default: 0.52).

    Returns
    -------
    None
        This function handles plotting and prints results directly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Download and plot raw data
    curr_stock = download_stock_data(ticker).history(period="max")
    plot_data_by_date(axes[0], curr_stock, label=f"{ticker} (Raw)")

    # Preprocess data
    data_preprocessing(curr_stock)

    # Define predictors
    ratio_predictors = [col for col in curr_stock.columns if col.startswith("Close_Ratio_")]
    trend_predictors = [col for col in curr_stock.columns if col.startswith("Trend_")]
    predictors = ratio_predictors + trend_predictors

    # Train and backtest
    model = define_model()
    predictions = backtest(
        data=curr_stock,
        model=model,
        predictors=predictors,
        threshold=threshold,
        start=500,
        step=250
    )

    # Add predictions
    curr_stock["Predictions"] = predictions["Predictions"]

    # Model performance
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"Model Precision (Threshold={threshold}): {precision:.2f}")

    # Daily trading strategy
    results_df, final_accuracy, final_wallet = daily_trading_strategy(
        data=curr_stock,
        ax=axes[1],
        start_wallet=10000.0,
        start_date="2023-01-01",
        end_date="2024-01-01"
    )

    # Print summary
    profit_percent = (final_wallet - 10000) / 10000 * 100
    print(f"Starting Wallet: 10,000.00")
    print(f"Ending Wallet:   {final_wallet:,.2f}")
    print(f"Model Accuracy (when predicted up): {final_accuracy:.2%}")
    print(f"Profit %: {profit_percent:.2f}%")

    plt.tight_layout()
    plt.show()
