import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import sys # Required to flush output  

def download_stock_data(ticker):
    return yf.Ticker(ticker)

def plot_data_by_date(ax, data, label):
    data.plot.line(y="Close", use_index=True, ax=ax, label=label)
    ax.set_title(f"{label} Closing Prices")
    ax.set_ylabel("Price")
    ax.legend()

def remove_unused_columns(data, col_list):
    """Remove unused columns and handle missing data in place."""
    data.drop(columns=col_list, inplace=True)
    data.dropna(inplace=True)

def add_tomorrow_column(data):
    """Add 'Tommorow' and 'Target' columns directly to the data."""
    data["Tommorow"] = data['Close'].shift(-1)
    data["Target"] = (data["Tommorow"] > data["Close"]).astype(int)

def remove_old_data(data, date):
    """Filter rows in place based on the date."""
    mask = data.index < date
    data.drop(index=data[mask].index, inplace=True)

def data_preprocessing(data):
    """Perform all preprocessing steps."""
    remove_unused_columns(data, ['Dividends', 'Stock Splits'])
    add_tomorrow_column(data)
    remove_old_data(data, "1990-01-01")
    add_statistics(data)

def define_model():
    """Train Random Forest Classifier model."""
    return RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def get_predictions(model, test, predictors):
    """Predict the target values for the test data."""
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1  # Threshold for classifying as 1
    preds[preds < 0.6] = 0   # Threshold for classifying as 0
    return pd.Series(preds, index=test.index, name="Predictions")

def predictor(train, test, predictors, model):
    """Predict the target values for the test data."""
    model.fit(train[predictors], train["Target"])
    preds = get_predictions(model, test, predictors)
    combined = pd.concat([test["Target"], preds], axis=1)
    combined.columns = ["Target", "Predictions"]
    return combined

def backtest(data, model, predictors, start=500, step=250):
    """Backtest the model using the test data."""
    all_predictions = []
    total_steps = (data.shape[0] - start) // step  # Calculate the total number of steps
    current_step = 0  # Initialize the current step

    for i in range(start, data.shape[0], step):
        current_step += 1
        # Print progress on the same line
        sys.stdout.write(f"\rProcessing batch {current_step}/{total_steps+1}...")
        sys.stdout.flush()
        train = data.iloc[:i].copy()
        test = data.iloc[i:i+step].copy()
        predictions = predictor(train, test, predictors, model)
        all_predictions.append(predictions)

    # Print completion message
    sys.stdout.write("\nBacktesting complete.\n")

    return pd.concat(all_predictions, ignore_index=True)

def add_statistics(data):
    horizons=[1, 5, 60, 20, 250,1000]
    for horizon in horizons:

        #calculate the rolling average
        rolling_average =data.rolling(window=horizon).mean()

        #add columns for the close ratio
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_average["Close"]
        
        #add columns for the volume ratio
        trend_column = f"Trend_{horizon}"
        data[trend_column] =data.shift(1).rolling(horizon).sum()["Target"]
    data.dropna(inplace=True)                       
    
def main():
    # Create a figure with two side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Download and preprocess S&P 500 data
    sp500 = download_stock_data("^GSPC").history(period="max")
    plot_data_by_date(axes[0], sp500, label="S&P 500 (Raw)")
    data_preprocessing(sp500)

    # Dynamically define predictors
    ratio_predictors = [col for col in sp500.columns if col.startswith("Close_Ratio_")]
    trend_predictors = [col for col in sp500.columns if col.startswith("Trend_")]
    predictors = ratio_predictors + trend_predictors

    # Define and backtest the model
    model = define_model()
    predictions = backtest(sp500, model, predictors, start=500, step=250)

    # Print prediction stats
    print(predictions["Predictions"].value_counts())  # Count prediction values
    total_precision = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"Precision: {total_precision:.2f}")
    target_distribution = predictions["Target"].value_counts() / predictions.shape[0]
    print("Target Distribution:\n", target_distribution)

    # Plot actual vs. predicted prices
    score = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"Precision: {score:.2f}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
