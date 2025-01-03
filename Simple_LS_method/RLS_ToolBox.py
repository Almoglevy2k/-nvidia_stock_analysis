import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from LS_ToolBox import *

def solve_regularized_least_squares(learn_matrix, next_day_vector, alpha=1.0):
    """
    Solves the regularized least squares problem manually (Ridge regression).
    
    Parameters:
        learn_matrix (ndarray): The feature matrix.
        next_day_vector (ndarray): The target vector.
        alpha (float): Regularization strength.

    Returns:
        ndarray: Coefficients for the model.
    """
    A = learn_matrix
    b = next_day_vector
    I = np.identity(A.shape[1])  # Identity matrix for regularization
    x = np.linalg.inv(A.T @ A + alpha * I) @ (A.T @ b)  # Regularized LS solution
    return x

def tune_alpha(learn_matrix, next_day_vector, alphas):
    """
    Tunes alpha for Ridge regression by manually testing different values.

    Parameters:
        learn_matrix (ndarray): The feature matrix.
        next_day_vector (ndarray): The target vector.
        alphas (list): A list of alpha values to test.

    Returns:
        float: Best alpha value.
    """
    best_alpha = None
    best_mse = float('inf')

    for alpha in alphas:
        x = solve_regularized_least_squares(learn_matrix, next_day_vector, alpha=alpha)
        predictions = learn_matrix @ x  # Predicted values
        mse = np.mean((next_day_vector - predictions) ** 2)  # Mean Squared Error

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    print(f"Tuned alpha: {best_alpha}, MSE: {best_mse:.4f}")
    return best_alpha

def trainREG_and_show(ticker):
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

    # Tune alpha for regularization
    alphas = [0.1, 1.0, 10.0, 100.0]  # Define a range of alphas
    best_alpha = tune_alpha(learn_matrix, b_train, alphas)

    # Solve regularized least squares with the tuned alpha
    x = solve_regularized_least_squares(learn_matrix, b_train, alpha=best_alpha)

    # Prepare test matrix
    test_matrix = build_test_matrix(stock_data_test, training_window)

    # Predict prices using test data
    prediction_vector = get_prediction_vector(test_matrix, x)

    # Plot predictions
    plot_predictions_by_date(stock_data_test, prediction_vector, training_window)

    # Simulate trading strategy
    test_closes_array = stock_data_test['Close'].values
    actual_closes_for_trading = test_closes_array[training_window:]
    predicted_closes_for_trading = prediction_vector

    final_wallet, wallet_values, trades = simple_threshold_strategy(
        actual_closes=actual_closes_for_trading,
        predicted_closes=predicted_closes_for_trading,
        threshold=1.005,
        initial_wallet=10_000.0
    )
    print(f"Final wallet: ${final_wallet:.2f} (Started with $10,000.00, Change: {((final_wallet - 10_000) / 10_000) * 100:.2f}%)")
    plot_wallet_values(wallet_values, trades)

