import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

# Fetch NVIDIA stock data and initialize learning variables
nvidia_data_train=yf.download("NVDA", start="2022-01-01", end="2023-12-31")
nvidia_data_test=yf.download("NVDA", start="2024-01-01", end="2024-12-31")
close_prices_train= nvidia_data_train[['Close']]
close_prices_test= nvidia_data_test[['Close']]
training_window = 30

# build training matrix
close_prices_array_train = close_prices_train['Close'].values # convert to numpy array
n_samples_train = len(close_prices_array_train) - training_window  #since we cant predict the first 30 days
learn_matrix = np.zeros((n_samples_train, training_window)) # initialize matrix size
#print(learn_matrix.shape)
#print(len(close_prices_array))
#print(n_samples)
for i in range(n_samples_train): # fill matrix with past k values in each row
    learn_matrix[i] = close_prices_array_train[i:i + training_window].flatten()
#print(learn_matrix)
next_day_vector=close_prices_array_train[training_window:] # vector b is the next day values
# Print the first row of learn_matrix
# solve for the least squares solution
# ||Ax-b||^2 when A is the learn_matrix and b is the next_day_vector
# to solve for x, we use the formula x = (A^T A)^-1 A^T b 
A = learn_matrix
b = next_day_vector
x = np.linalg.inv(A.T @ A) @ A.T @ b
print("\nLeast squares solution:")
print(x)


# now we can predict the next day value by multiplying the last k values with the weights
# lets run a test ! 
close_prices_array_test = close_prices_test['Close'].values # convert to numpy array
n_samples_test = len(close_prices_array_test) - training_window  #since we cant predict the first 30 days
test_matrix = np.zeros((n_samples_test, training_window)) # initialize matrix size
for i in range(n_samples_test): # fill matrix with past k values in each row
    test_matrix[i] = close_prices_array_test[i:i + training_window].flatten()

prediction_vector = test_matrix @ x 

# Safety check: ensure n_samples_test >= 5
max_rows_to_print = min(n_samples_test, 5)

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

plot_predictions_by_date(close_prices_test, prediction_vector, training_window)
