# -nvidia_stock_analysis

### Files Overview
| File                  | Description                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------|
| `Sim_trade_LS.py`     | Main script for running non-regularized least squares (LS) stock price prediction and trading simulation. |
| `Sim_trade_RLS.py`    | Main script for running regularized Ridge regression (RLS) stock price prediction and trading simulation. |
| `LS_ToolBox.py`       | Toolbox containing helper functions for building matrices, downloading stock data, and solving LS regression models. |
| `RLS_ToolBox.py`      | Toolbox containing helper functions for regularized Ridge regression, including alpha tuning and prediction generation. |


## What Each Script Returns

### `Sim_trade_LS.py`
- Predicts stock prices using non-regularized least squares.
- Evaluates performance using metrics:
  - **Training MSE**.
  - **Test MSE**.
  - **Variance of test data**.
  - **MSE-to-variance ratio**.
- Simulates a simple trading strategy:
  - Outputs the final wallet value after trading based on the model predictions.

### `Sim_trade_RLS.py`
- Predicts stock prices using regularized Ridge regression.
- Tunes the regularization strength (`alpha`) and reports the best value.
- Evaluates performance with the same metrics as `Sim_trade_LS.py`.
- Simulates trading and reports the final wallet value.

### `LS_ToolBox.py`
- Utility functions for:
  - Downloading stock data from Yahoo Finance (`yfinance`).
  - Preparing training and test matrices.
  - Solving least squares models.
  - Simulating a simple trading strategy.

### `RLS_ToolBox.py`
- Utility functions for:
  - Solving Ridge regression models with regularization.
  - Tuning the `alpha` parameter for Ridge regression.
  - Generating predictions using the regularized model.

---

## How to Use

### 1. Non-Regularized LS Model
Run `Sim_trade_LS.py` to predict stock prices and simulate trading with non-regularized least squares:
```bash
python Sim_trade_LS.py <ticker> [<threshold>]

### 2. Regularized LS Model
Run `Sim_trade_RLS.py` to predict stock prices and simulate trading with non-regularized least squares:
```bash
python Sim_trade_LS.py <ticker> [<threshold>]
