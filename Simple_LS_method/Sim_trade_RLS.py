import sys
from RLS_ToolBox import *

def main():
    # Check command-line arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python Sim_trade_RLS.py <ticker> [<threshold>]")
        sys.exit(1)
    
    # Parse the ticker
    ticker = sys.argv[1]
    
    # Parse the optional threshold or use a default value
    threshold = float(sys.argv[2]) if len(sys.argv) == 3 else 1.005
    
    # Call trainREG_and_show with the custom threshold
    trainREG_and_show(ticker, threshold)

if __name__ == "__main__":
    main()
