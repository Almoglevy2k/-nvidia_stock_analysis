import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Download historical stock data for a given ticker and date range
def download_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# Extract only the 'Close' prices from stock data
def get_close_prices(data):
    return data[['Close']]

# Build training matrix using a sliding window approach
def build_training_matrix(close_prices_train, training_window):
    close_prices_array_train = close_prices_train['Close'].values
    n_samples_train = len(close_prices_array_train) - training_window
    learn_matrix = np.zeros((n_samples_train, training_window))

    for i in range(n_samples_train):
        learn_matrix[i] = close_prices_array_train[i:i + training_window].flatten()

    return learn_matrix

# Get the vector of next day's prices (target variable)
def get_next_day_vector(close_prices_array, training_window):
    return close_prices_array[training_window:]

# Solve least squares regression to find optimal coefficients
def solve_least_squares(learn_matrix, next_day_vector):
    A = learn_matrix
    b = next_day_vector
    # Solve A.T @ A @ x = A.T @ b
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x

# Build test matrix using a sliding window approach
def build_test_matrix(close_prices_test, training_window):
    close_prices_array_test = close_prices_test['Close'].values
    n_samples_test = len(close_prices_array_test) - training_window
    test_matrix = np.zeros((n_samples_test, training_window))

    for i in range(n_samples_test):
        test_matrix[i] = close_prices_array_test[i:i + training_window].flatten()

    return test_matrix

# Predict next day's prices using the test matrix and learned coefficients
def get_prediction_vector(test_matrix, x):
    return test_matrix @ x

# Plot actual vs. predicted prices with time as x-axis
def plot_predictions_by_date(close_prices_test, prediction_vector, training_window):
    actual_prices = close_prices_test['Close']
    predicted_index = actual_prices.index[training_window:]
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices.index, actual_prices, label='Actual Prices')
    plt.plot(predicted_index, prediction_vector, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Predicted vs. Actual Closing Prices')
    plt.legend()
    plt.show()

# Simulate a simple trading strategy based on prediction thresholds
def simple_threshold_strategy(actual_closes, predicted_closes, threshold=1.005, initial_wallet=10_000.0):
    actual_closes = actual_closes.flatten()
    predicted_closes = predicted_closes.flatten()

    wallet = initial_wallet
    wallet_values = [wallet]  # Track wallet value over time
    trades = [False]  # Track whether a trade occurred
    n = len(actual_closes)

    for t in range(1, n):
        trade_made = False
        # Decide whether to buy based on yesterday's actual and today's predicted price
        if predicted_closes[t] > actual_closes[t - 1] * threshold:
            # Simulate buying today and selling tomorrow
            wallet *= actual_closes[t] / actual_closes[t - 1]
            trade_made = True

        # Track wallet value and whether a trade occurred
        wallet_values.append(wallet)
        trades.append(trade_made)

    return wallet, wallet_values, trades

# Plot wallet value over time and mark trade days
def plot_wallet_values(wallet_values, trades):
    x = range(len(wallet_values))
    y = wallet_values

    # Identify trade days and their outcomes
    trade_days = [i for i, traded in enumerate(trades) if traded]
    trade_values = [wallet_values[i] for i in trade_days]
    trade_colors = [
        'green' if trade_values[i] > (wallet_values[trade_days[i] - 1] if trade_days[i] > 0 else wallet_values[0])
        else 'red'
        for i in range(len(trade_days))
    ]

    plt.figure(figsize=(12, 6))
    plt.scatter(trade_days, trade_values, c=trade_colors, s=80, edgecolors='black', marker='o', label='Trade Days')
    plt.plot(x, y, linestyle='--', alpha=0.7, label='Wallet Trend')
    plt.title('Wallet Value Over Time During Trading')
    plt.xlabel('Trading Day')
    plt.ylabel('Wallet Value ($)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Train model, predict prices, and simulate strategy
def train_and_show(ticker, threshold=1.005):
    train_start_date = "2022-01-01"
    train_end_date = "2023-12-31"
    test_start_date = "2024-01-01"
    test_end_date = "2024-12-31"
    training_window = 30

    # Download train and test data
    stock_data_train = download_stock_data(ticker, train_start_date, train_end_date)
    stock_data_test = download_stock_data(ticker, test_start_date, test_end_date)

    # Prepare training matrix and next day vector
    learn_matrix = build_training_matrix(stock_data_train, training_window)
    train_prices_array = stock_data_train['Close'].values
    b_train = get_next_day_vector(train_prices_array, training_window)

    # Solve least squares to find coefficients
    x = solve_least_squares(learn_matrix, b_train)

    # Prepare test matrix
    test_matrix = build_test_matrix(stock_data_test, training_window)

    # Predict prices using test data
    prediction_vector = get_prediction_vector(test_matrix, x)

    # Calculate training MSE
    train_predictions = learn_matrix @ x
    train_mse = np.mean((b_train - train_predictions) ** 2)

    # Calculate test MSE
    test_closes_array = stock_data_test['Close'].values
    actual_closes_for_trading = test_closes_array[training_window:]
    test_mse = np.mean((actual_closes_for_trading - prediction_vector) ** 2)

    # Variance of test set
    test_variance = np.var(actual_closes_for_trading)

    # Print metrics
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test Variance: {test_variance:.4f}")
    print(f"Test MSE/Variance Ratio: {test_mse / test_variance:.4f}")

    # Plot predictions
    plot_predictions_by_date(stock_data_test, prediction_vector, training_window)

    # Simulate trading strategy
    predicted_closes_for_trading = prediction_vector
    final_wallet, wallet_values, trades = simple_threshold_strategy(
        actual_closes=actual_closes_for_trading,
        predicted_closes=predicted_closes_for_trading,
        threshold=threshold,
        initial_wallet=10_000.0
    )
    print(f"Final wallet: ${final_wallet:.2f} (Started with $10,000.00, Change: {((final_wallet - 10_000) / 10_000) * 100:.2f}%)")
    plot_wallet_values(wallet_values, trades)

