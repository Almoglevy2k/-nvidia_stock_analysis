#!/usr/bin/env python3
import argparse
from RFC_ToolBox import run_RFC_sim

def main():
    # Create argument parser, including a short epilog usage example
    parser = argparse.ArgumentParser(
        description="Run a Random Forest Classifier simulation with optional ticker and threshold.",
        epilog="Example usage: python Sim_trade_RFC.py --ticker AAPL --threshold 0.60"
    )
    
    # Optional: ticker
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Stock ticker symbol to download (default: NVDA)"
    )
    
    # Optional: threshold
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.52,
        help="Probability threshold for 'buy' predictions (default: 0.52)"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call our simulation function with these args
    run_RFC_sim(ticker=args.ticker, threshold=args.threshold)

if __name__ == "__main__":
    main()
