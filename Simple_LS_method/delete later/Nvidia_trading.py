from ToolBox import *

def main():
    # 1. Define parameters
    ticker = "NVDA"
    train_start_date = "2022-01-01"
    train_end_date   = "2023-12-31"
    test_start_date  = "2024-01-01"
    test_end_date    = "2024-12-31"
    training_window  = 30
    
    # 2. Download train and test data
    nvidia_data_train = download_stock_data(ticker, train_start_date, train_end_date)
    nvidia_data_test  = download_stock_data(ticker, test_start_date, test_end_date)
    
    # 3. Build training matrix and target vector
    learn_matrix = build_training_matrix(nvidia_data_train, training_window)
    train_prices_array = nvidia_data_train['Close'].values
    b_train = get_next_day_vector(train_prices_array, training_window)
    
    # 4. Solve for x using least squares 
    x = solve_least_squares(learn_matrix, b_train)
    #print("Learned coefficients x:\n", x)
    
    # 5. Build test matrix
    test_matrix = build_test_matrix(nvidia_data_test, training_window)
    
    # 6. Predict on the test set
    prediction_vector = get_prediction_vector(test_matrix, x)

    
    # 7. Plot neatly
    plot_predictions_by_date(nvidia_data_test, prediction_vector, training_window)


    # 8. Run the simple threshold strategy
    test_closes_array = nvidia_data_test['Close'].values
    actual_closes_for_trading    = test_closes_array[training_window:]
    predicted_closes_for_trading = prediction_vector 
    
    final_wallet, wallet_values, trades  = simple_threshold_strategy(
        actual_closes=actual_closes_for_trading,
        predicted_closes=predicted_closes_for_trading,
        threshold=1.005,
        initial_wallet=10_000.0
    )
    print(f"Final wallet: ${final_wallet:.2f} (Started with $10,000.00, Change: {((final_wallet - 10_000) / 10_000) * 100:.2f}%)")
    plot_wallet_values(wallet_values, trades)

if __name__ == "__main__":
    main()