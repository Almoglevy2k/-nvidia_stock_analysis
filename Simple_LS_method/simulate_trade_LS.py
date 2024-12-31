import sys
from ToolBox import *

def main():
    if len(sys.argv) != 2:
        print("Usage: python simulate_trade_LS.py <ticker>")
        sys.exit(1)
    
    ticker = sys.argv[1]
    train_and_show(ticker)
    
if __name__ == "__main__":
    main()