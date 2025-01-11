### NVIDIA Stock Analysis

A project demonstrating how to:
- **Download and preprocess** stock data (NVIDIA by default, but adaptable to any ticker).
- **Train and backtest** a Random Forest Classifier to predict daily market movements.
- **Print performance metrics** (precision, daily wallet value, etc.).
- **Simulate a simple daily trading strategy** to see how the model might perform in practice.

## File Overview

| **File**              | **Description**                                                                                                                        |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `RFC_ToolBox.py`      | Core logic and helper functions for data preprocessing, model training, backtesting, and a simple trading strategy simulation.         |
| `Sim_trade_RFC.py`    | Command-line script that reads optional arguments (`--ticker`, `--threshold`) and runs the simulation workflow, printing results.     |
| `requirements.txt`    | List of the packages and versions needed to run this project.                                                                          |

## Installation

1. **Clone or Download** this repository.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

## How to Run
Navigate to the project directory (where both Sim_trade_RFC.py and RFC_ToolBox.py are located).
Run the Script:
Default usage (ticker = "NVDA", threshold = 0.52):
```bash
python Sim_trade_RFC.py
```
Custom usage (e.g., Apple and threshold 0.60):
```bash
python Sim_trade_RFC.py --ticker AAPL --threshold 0.60
```
Get help:Displays usage info, default values, and example commands.
```bash
python Sim_trade_RFC.py --help
```

