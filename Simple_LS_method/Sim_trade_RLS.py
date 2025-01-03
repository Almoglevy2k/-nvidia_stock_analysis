import sys
from RLS_ToolBox import *



def main():
    if len(sys.argv) != 2:
        print("Usage: python Sim_trade_RLS.py <ticker>")
        sys.exit(1)
    
    ticker = sys.argv[1]
    trainREG_and_show(ticker)
    
if __name__ == "__main__":
    main()