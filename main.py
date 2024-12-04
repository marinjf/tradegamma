from datetime import timedelta
from src.trader.voltrader import BacktestVolatilityTraderInput, BacktestVolatilityTrader

inputdata = BacktestVolatilityTraderInput(
    deposit_usd  = 100000,
    reload_market  = True ,
    first_data_point  = 2000000,
    last_data_point  = 3000000, 
    time_serie_length  = timedelta(days=5),
    dt  = 1/(365*24),
    max_spread_to_close  = 0.05,
    max_local_exposure  = 0.2, 
    exposure=0.1
)
bt = BacktestVolatilityTrader(inputdata).launch_backtest()

bt.write_full_report()
