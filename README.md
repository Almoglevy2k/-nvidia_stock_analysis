# Stock Analysis & Trading Simulations

This repository contains **two complementary approaches** to analyzing and simulating trades on NVIDIA (or any other) stock data:

1. **Linear and Regularized Least Squares (LS / RLS)**  
2. **Random Forest Classifier (RFC)**  

Both approaches download stock data, create predictive models, backtest on historical periods, **print performance metrics**, and run a **simple daily trading simulation** (e.g., buy if tomorrow is predicted up, sell at the close).

---

## Repository Structure

### 1. LS-Based Directory
- **`Sim_trade_LS.py`**  
  Main script for **non-regularized** least squares. Predicts stock prices, prints metrics (MSE, MSE-to-variance, etc.), and simulates a simple trading strategy.
- **`Sim_trade_RLS.py`**  
  Main script for **regularized** (Ridge) least squares. Tunes the regularization parameter (`alpha`), prints best alpha, prints metrics, and simulates trading.
- **`LS_ToolBox.py`**  
  Helper functions for building matrices, downloading data, and solving non-regularized least squares.
- **`RLS_ToolBox.py`**  
  Helper functions for **Ridge regression** (regularized LS): alpha tuning, solution, predictions, etc.

### 2. RFC-Based Directory
- **`RFC_ToolBox.py`**  
  Core logic for data preprocessing, model training, backtesting, and a simple daily trading strategy, all using a **Random Forest Classifier**.
- **`Sim_trade_RFC.py`**  
  Command-line script that accepts optional arguments (`--ticker`, `--threshold`) to run the **RFC** workflow. Prints model precision, final wallet, and simulates daily trades.

### 3. (Optional) `requirements.txt`
A file listing the packages and versions needed (e.g., `yfinance`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`).

---

## Installation & Setup

1. **Clone or Download** this repository.
2. **Install Dependencies** (if you have a `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```

## How to run 
Each script supports a --help option that explains usage, arguments, and defaults. For example:

# LS Simulation 
 ```bash
python Sim_trade_LS.py --help
 ```
# RLS Simulation 
 ```bash
python Sim_trade_RLS.py --help
 ```
# RFC Simulation
  ```bash
python Sim_trade_RFC.py --help
 ```
