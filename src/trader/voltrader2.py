from typing import List 
from dataclasses import dataclass
import numpy as np 
from datetime import timedelta, datetime
from src.portfolio import (
    Trade, 
    CashFlow, 
    Book, 
    Portfolio,
    Position, 
    trades_to_positions, 
    settle_book_expired_positions)
from src.instruments import Option, Currency
from src.market import Market
from src.quant.timeserie import (
    TimeSerie, 
    get_square_log_difference_time_serie, 
    get_log_difference_time_serie,
    get_rolling_realised_vol_time_series, 
    get_rolling_mean_time_series,
    get_spread_time_series, 
    get_z_score_time_series, 
    get_z_score_time_serie, 
    get_annualized_realised_volatility_time_serie, 
    get_dt_from_time_delta, 
    filter_time_serie_from_dates, 
    filter_many_time_serie_from_dates)
from src.quant.ssvi import SSVI
from src.trader.base import (
    TraderParameters, 
    Trader, 
    BacktestInput, 
    BacktestTrader, 
    OptionPnlEngine)

@dataclass
class VolatilityBlockTrade: 
    trades : List[Trade]
    market : Market

    def __post_init__(self): 
        self._id = self.get_id()
        self.book =  Book(self.trades, CashFlow(0,Currency('USD')))
        self.portfolio = self.book.to_portfolio(self.market)
        self.positions = self.book.to_positions()
        self.pnlengine = OptionPnlEngine(self.book,self.market)
        self.spotq = self.market.get_quote(self.market.spot.name)

    def get_id(self) -> str: 
        _id = ''
        for t in self.trades: 
            if t.number_contracts<0: 
                _id = _id + '_SHORT_'+t.instrument.name
            elif t.number_contracts>0: 
                _id = _id + '_LONG_'+t.instrument.name
            else: 
                _id = _id + '_NEUTRAL_'+t.instrument.name
        return _id
    
    def get_trades_usd_fee(self) -> float: 
        fee = 0
        for t in self.trades: 
            tfee = t.trade_fee()
            if tfee.currency!=Currency('USD'):
                fee = fee + tfee.amount*self.spotq.order_book.mid
            else: fee = fee + tfee.amount
        deltahedgetrade = self.portfolio.perpetual_delta_hedging_trade()
        dhtfee = deltahedgetrade.trade_fee()
        if dhtfee.currency!=Currency('USD'):
                fee = fee + dhtfee.amount*self.spotq.order_book.mid
        else: fee = fee + dhtfee.amount
        return fee
    
    def get_initial_margin_usd(self) -> float: 
        bookmargin = self.portfolio.get_margin() 
        if bookmargin.currency == Currency('USD'): margin=bookmargin.initial
        else: margin=bookmargin.initial*self.spotq.order_book.mid
        deltahedgetrade = self.portfolio.perpetual_delta_hedging_trade()
        deltahedgepft = Portfolio(
            trades_to_positions([deltahedgetrade]),
            self.market, 
            self.book.initial_deposit)
        deltamargin = deltahedgepft.get_margin()
        if deltamargin.currency == Currency('USD'): margin=margin+deltamargin.initial
        else: margin=margin+deltamargin.initial*self.spotq.order_book.mid
        return margin

    def get_usd_premium(self) -> float: 
        premium = 0 
        for p in self.positions: 
            if p.price_currency == Currency('USD'): 
                premium = premium + p.get_premium()
            else: premium = premium + p.get_premium()*self.spotq.order_book.mid
        return premium
    
    def get_usd_spread(self) -> float: 
        spread = 0 
        for t in self.trades: 
            quote = self.market.get_quote(t.instrument.name)
            n = t.number_contracts*t.instrument.contract_size
            bid_usd = quote.order_book.best_bid*self.spotq.order_book.mid
            ask_usd = quote.order_book.best_ask*self.spotq.order_book.mid
            spread = spread + abs(n)*(ask_usd-bid_usd)
        return spread

    def usd_capital_requirement(self) -> float: 
        return self.get_trades_usd_fee() + self.get_initial_margin_usd() \
            - self.get_usd_premium() + self.get_usd_spread()

    def get_theta_gamma_pnl(self,dt:float,realised_volatility:float)->float:
        return self.pnlengine.proxy_theta_gamma_pnl(
            realised_volatility,dt)  

    def get_vega_pnl(self,dt:float,realised_volatility:float)->float:
        pass 

    def get_volatility_pnl(
            self, 
            dt:float, 
            ssvi_forecast:SSVI, 
            realised_volatility:float) -> float: 
        theta_gamma_appprox = self.pnlengine.proxy_theta_gamma_pnl(
            realised_volatility,dt) 
        #vega = self.pnlengine.vega_pnl(ssvi_forecast)
        return theta_gamma_appprox#+vega

    def get_roi(
            self, 
            dt:float, 
            ssvi_forecast:SSVI, 
            realised_volatility:float) -> float: 
        vol_pnl = self.get_volatility_pnl(dt,ssvi_forecast,realised_volatility)
        capital_requirement = max(self.usd_capital_requirement(), 0.00000001)
        return vol_pnl/capital_requirement

    def get_number_contracts_to_trade(self, q:float) -> float: 
        output = list()
        for t in self.trades: 
            quote = self.market.get_quote(t.instrument.name)
            if q < 0: q = min(q, quote.order_book.best_bid_size)
            else: q = min(q, quote.order_book.best_ask_size)
            output.append(q)
        return min(output)
        
def generate_block_trade(leg1:Option, 
                         leg2:Option, 
                         same_direction: bool, 
                         market:Market) -> List[VolatilityBlockTrade]: 
    atm_chain = market.atm_chain
    quote1 = atm_chain.mapped_quotes[leg1.name]
    quote2 = atm_chain.mapped_quotes[leg2.name]
    firstdata = {'instrument':leg1, 
                    'reference_time': market.reference_time, 
                    'currency':quote1.order_book.quote_currency}
    seconddata = {'instrument':leg2, 
                    'reference_time':  market.reference_time, 
                    'currency':quote2.order_book.quote_currency}
    fdatalong, fdatashort = firstdata.copy(), firstdata.copy()
    sdatalong, sdatashort = seconddata.copy(), seconddata.copy()
    fdatalong['number_contracts']=1
    sdatalong['number_contracts']=1
    fdatashort['number_contracts']=-1
    sdatashort['number_contracts']=-1
    fdatalong['traded_price'] = quote1.order_book.best_ask
    sdatalong['traded_price'] = quote2.order_book.best_ask
    fdatashort['traded_price'] = quote1.order_book.best_bid
    sdatashort['traded_price'] = quote2.order_book.best_bid
    flongtrade = Trade(**fdatalong)
    fshorttrade = Trade(**fdatashort)
    slongtrade = Trade(**sdatalong)
    sshorttrade = Trade(**sdatashort) 
    if same_direction: 
        bt1 = VolatilityBlockTrade([flongtrade, slongtrade], market)
        bt2 = VolatilityBlockTrade([fshorttrade, sshorttrade], market)
    else: 
        bt1 = VolatilityBlockTrade([flongtrade, sshorttrade], market)
        bt2 = VolatilityBlockTrade([fshorttrade, slongtrade], market)
    return [bt1, bt2]

def get_atm_calendar_spread(market: Market, call=True) -> List[VolatilityBlockTrade]: 
    atm_chain = market.atm_chain
    output = list()
    if call: options = atm_chain.calls
    else: options = atm_chain.puts
    mapped_cs = dict()
    for i in range(0, len(options)-1): 
        for u in range(0, len(options)-1): 
            n1, n2 = options[i].name, options[u].name
            keys = list(mapped_cs.keys())
            combi1, combi2 = n1+n2, n2+n1
            if n1 == n2: continue
            elif combi1 in keys or combi2 in keys : continue
            else: 
                mapped_cs[n1+n2] = [options[i], options[u]]
    for k in list(mapped_cs.keys()):
        opt1, opt2 = mapped_cs[k][0], mapped_cs[k][1]
        output = output + generate_block_trade(
            opt1, opt2, False, market)
    return output

def get_atm_straddle(market: Market) -> List[VolatilityBlockTrade]:
    output = list()
    for c in market.atm_chain.calls: 
        for p in market.atm_chain.puts: 
            output = output + generate_block_trade(c,p,True,market)
    return output 

def get_atm_block_trades(market: Market) -> List[VolatilityBlockTrade]: 
    return get_atm_straddle(market)\
        +get_atm_calendar_spread(market,True)\
        +get_atm_calendar_spread(market,False)

def get_ordered_mapped_gamma_roi(
        bt: List[VolatilityBlockTrade], 
        rvol_forecast: float, 
        dt: float) -> dict[str, float]: 
    output = dict()
    for b in bt: 
        pnl = b.pnlengine.proxy_theta_gamma_pnl(rvol_forecast, dt)
        if pnl > 0: output[b._id] = pnl
        else: continue
    return dict(sorted(output.items(),key=lambda item: item[1],reverse=True))

def get_ordered_mapped_vega_roi(
        bt: List[VolatilityBlockTrade], 
        ssvi_forecast: SSVI) -> dict[str, float]: 
    output = dict()
    for b in bt: 
        pnl = b.pnlengine.vega_pnl(ssvi_forecast)
        if pnl > 0:output[b._id] = pnl
        else: continue
    return dict(sorted(output.items(),key=lambda item: item[1],reverse=True))

@dataclass
class VolatilityTraderModelInput: 
    perp_mark_price : TimeSerie
    ssvi_rho : TimeSerie
    ssvi_gamma : TimeSerie
    ssvi_nu : TimeSerie
    ssvi_atm_volatility : dict[float, TimeSerie]

@dataclass 
class VolatilityTraderModel: 
    def __init__(self, inputdata: VolatilityTraderModelInput): 
        self.perp_mark_price = inputdata.perp_mark_price
        self.ssvi_rho = inputdata.ssvi_rho
        self.ssvi_gamma = inputdata.ssvi_gamma
        self.ssvi_nu = inputdata.ssvi_nu
        self.ssvi_atm_volatility = inputdata.ssvi_atm_volatility
    
    @staticmethod
    def apply_date_shift_time_serie(target:TimeSerie, factors:List[TimeSerie]) \
        -> tuple[TimeSerie, List[TimeSerie]]: 
        min_date = max([min(d.dates) for d in factors])
        max_date = min([max(d.dates) for d in factors])
        target = filter_time_serie_from_dates(
            target,min_date,max_date,False,True)
        factors = filter_many_time_serie_from_dates(
            factors,min_date,max_date,True,False)
        return (target, factors)

    @staticmethod
    def get_forecast(
        target: TimeSerie, 
        factors:List[TimeSerie], 
        last_factors_values: List[float]) -> float: 
        mean_target = np.mean(np.array(target.values))
        std_target = np.std(np.array(target.values))
        target = get_z_score_time_serie(target)
        factors = get_z_score_time_series(factors)
        y = np.array(target.values)
        x = np.array([f.values for f in factors])
        lsq = np.linalg.lstsq(np.transpose(x), y)
        zs = np.sum(lsq[0]*np.array(last_factors_values))
        f = (std_target*zs)+mean_target
        return f.item()

    def rvol_compute_target_data(self) -> TimeSerie: 
        dt = get_dt_from_time_delta(timedelta(hours=1))
        data = get_annualized_realised_volatility_time_serie(
            self.perp_mark_price, dt)
        #data = get_log_difference_time_serie(self.perp_mark_price)
        #return get_z_score_time_serie(data)
        return data
    
    def rvol_compute_factor_data(self) -> List[TimeSerie]: 
        mpsqld = get_square_log_difference_time_serie(self.perp_mark_price)
        rv = get_rolling_realised_vol_time_series(
            mpsqld, 
            [timedelta(hours=24), 
            timedelta(hours=12), 
            timedelta(hours=6), 
            timedelta(hours=3), 
            timedelta(hours=1)])
        rvs = get_spread_time_series(rv)
        #return get_z_score_time_series(rvs)
        return rvs
    
    @staticmethod
    def ssvi_param_compute_factor_data(data:TimeSerie) -> List[TimeSerie]: 
        ldm = get_rolling_mean_time_series(
            data, 
            [timedelta(hours=24), 
            timedelta(hours=12), 
            timedelta(hours=6), 
            timedelta(hours=3), 
            timedelta(hours=1)])
        ldms = get_spread_time_series(ldm)
        #return get_z_score_time_series(rvs)
        return ldms

    def ssvi_rho_forecast(self) -> float: 
        factors = self.ssvi_param_compute_factor_data(self.ssvi_rho)
        max_date = min([max(d.dates) for d in factors])
        ssvirho_lfactors = [d.datamap[max_date] for d in factors]
        target, factors = self.apply_date_shift_time_serie(
            self.ssvi_rho,factors)
        return self.get_forecast(target,factors, ssvirho_lfactors)
    
    def ssvi_gamma_forecast(self) -> float: 
        factors = self.ssvi_param_compute_factor_data(self.ssvi_gamma)
        max_date = min([max(d.dates) for d in factors])
        ssvigamma_lfactors = [d.datamap[max_date] for d in factors]
        target, factors = self.apply_date_shift_time_serie(
            self.ssvi_gamma,factors)
        return self.get_forecast(target,factors, ssvigamma_lfactors)
    
    def ssvi_nu_forecast(self) -> float: 
        factors = self.ssvi_param_compute_factor_data(self.ssvi_nu)
        max_date = min([max(d.dates) for d in factors])
        ssvinu_lfactors = [d.datamap[max_date] for d in factors]
        target, factors = self.apply_date_shift_time_serie(
            self.ssvi_nu,factors)
        return self.get_forecast(target,factors, ssvinu_lfactors)
    
    def ssvi_atm_vol_forecast(self, t:float) -> float: 
        data = get_log_difference_time_serie(self.ssvi_atm_volatility[t])
        factors = self.ssvi_param_compute_factor_data(data)
        max_date = min([max(d.dates) for d in factors])
        ssviatmvol_lfactors = [d.datamap[max_date] for d in factors]
        target, factors = self.apply_date_shift_time_serie(
            data,factors)
        atmvol = self.ssvi_atm_volatility[t].datamap[max_date]
        r = self.get_forecast(target,factors,ssviatmvol_lfactors)
        value = atmvol*(1+r)
        return value.item()
    
    def rvol_forecast(self) -> float: 
        rvol_target = self.rvol_compute_target_data()
        rvol_factors = self.rvol_compute_factor_data()
        max_date = min([max(d.dates) for d in rvol_factors])
        rvol_lfactors = [d.datamap[max_date] for d in rvol_factors]
        target, factors = self.apply_date_shift_time_serie(
            rvol_target,rvol_factors)
        return self.get_forecast(target,factors, rvol_lfactors)

    def ssvi_forecast(self) -> SSVI: 
        atmtvarmap = {0:0}
        for t in list(self.ssvi_atm_volatility.keys()):
            atmtvarmap[t] = t*(self.ssvi_atm_vol_forecast(t)**2)
        nu = self.ssvi_nu_forecast()
        _gamma = self.ssvi_gamma_forecast()
        rho = self.ssvi_rho_forecast()
        return SSVI(rho,nu,_gamma, atmtvarmap)

@dataclass
class VolatilityTraderParameters(TraderParameters): 
    model_input : VolatilityTraderModelInput
    gamma_pnl_target_factor : float = 0.001
    vega_pnl_target_factor : float = 0.0005
    dt : float = 1/(365*24)
    
    def __post_init__(self):
        self.book = settle_book_expired_positions(self.book, self.market)
        self.trader_model = VolatilityTraderModel(self.model_input)
        self.portfolio = self.book.to_portfolio(self.market)
        self.poftvalue = self.portfolio.get_usd_total_value()
        

class VolatilityTrader(Trader): 
    def __init__(self, parameters : VolatilityTraderParameters): 
        self.parameters = parameters
        self.market = self.parameters.market
        self.book = self.parameters.book
        self.ssvi_forecast = self.parameters.trader_model.ssvi_forecast()
        self.rvol_forecast = self.parameters.trader_model.rvol_forecast()
        self.atm_bt = get_atm_block_trades(self.market)
        self.mapped_atm_bt = {b._id:b for b in self.atm_bt}
        self.gamma_trades = self.get_gamma_trades()
        self.vega_trades = self.get_vega_trades()

    def get_gamma_pnl_target(self) -> float: 
        pftvalue = self.book.initial_deposit.amount
        gamma_pnl_target = self.parameters.gamma_pnl_target_factor*pftvalue
        pnlengine = OptionPnlEngine(self.book, self.market)
        pft_gamma_pnl = pnlengine.proxy_theta_gamma_pnl(
            self.rvol_forecast, self.parameters.dt)
        return max(gamma_pnl_target-pft_gamma_pnl,0)
    
    def get_vega_pnl_target(self) -> float: 
        pftvalue = self.book.initial_deposit.amount
        vega_pnl_target = self.parameters.vega_pnl_target_factor*pftvalue
        pnlengine = OptionPnlEngine(self.book, self.market)
        pft_vega_pnl = pnlengine.vega_pnl(self.ssvi_forecast)
        return max(vega_pnl_target-pft_vega_pnl,0)
    
    def get_gamma_trades(self) -> List[Trade]: 
        ref,dt = self.rvol_forecast, self.parameters.dt
        mapped_roi = get_ordered_mapped_gamma_roi(self.atm_bt,ref,dt)
        _ids = list(mapped_roi.keys())
        target = self.get_gamma_pnl_target()
        winner_bt = self.mapped_atm_bt[_ids[0]]
        gamma_pnl = winner_bt.pnlengine.proxy_theta_gamma_pnl(ref,dt)
        q = winner_bt.get_number_contracts_to_trade(target/gamma_pnl)
        output = list()
        for t in winner_bt.trades: 
            trade_dict = t.__dict__
            trade_dict['number_contracts'] = q*t.number_contracts
            output.append(Trade(**trade_dict))
        return output 
    
    def get_vega_trades(self) -> List[Trade]: 
        mapped_roi = get_ordered_mapped_vega_roi(self.atm_bt,self.ssvi_forecast)
        _ids = list(mapped_roi.keys())
        gammabt = VolatilityBlockTrade(self.gamma_trades,self.market)
        gammabt_vegapnl = gammabt.pnlengine.vega_pnl(self.ssvi_forecast)
        target = self.get_vega_pnl_target()-gammabt_vegapnl
        winner_bt = self.mapped_atm_bt[_ids[0]]
        vega_pnl = winner_bt.pnlengine.vega_pnl(self.ssvi_forecast)
        q = winner_bt.get_number_contracts_to_trade(target/vega_pnl)
        output = list()
        for t in winner_bt.trades: 
            trade_dict = t.__dict__
            trade_dict['number_contracts'] = q*t.number_contracts
            output.append(Trade(**trade_dict))
        return output 
        
    def update_book(self) -> Book:
        trades = self.gamma_trades + self.vega_trades + self.book.trades
        updated_book = Book(trades, self.book.initial_deposit)
        pft = updated_book.to_portfolio(self.market)
        delta_hedging_trade = pft.perpetual_delta_hedging_trade()
        trades.append(delta_hedging_trade)
        updated_book = Book(trades, self.book.initial_deposit)
        pft = updated_book.to_portfolio(self.market)
        if pft.is_cash_sufficient(): return updated_book
        else: 
            trades = self.book.trades
            pft = self.book.to_portfolio(self.market)
            delta_hedging_trade = pft.perpetual_delta_hedging_trade()
            trades.append(delta_hedging_trade)
            return Book(trades, self.book.initial_deposit)


@dataclass
class BacktestVolatilityTraderInput(BacktestInput): 
    time_serie_length : timedelta = timedelta(days=20)
    dt : float = 1/(365*24)
    gamma_pnl_target_factor : float = 0.001
    vega_pnl_target_factor : float = 0.0005
    atm_t_vector = [0.05, 0.1, 0.33, 0.5, 1]


class BacktestVolatilityTrader(BacktestTrader): 
    def __init__(self, inputdata: BacktestVolatilityTraderInput):
        self.parameters = inputdata
        super().__init__(inputdata)
     
    def get_date_vector_for_backtest(self, dates:List[datetime]) -> List[datetime]: 
        min_date = min(dates)
        min_date_for_bt = min_date + self.parameters.time_serie_length
        return [d for d in dates if d>min_date_for_bt]
    
    def get_model_input(self, reference_time: datetime) -> VolatilityTraderModelInput:
        ml, rft  = self.loader, reference_time
        rf = self.risk_factor
        perp_name = rf.base_currency.code + '-PERPETUAL'
        dt = self.parameters.time_serie_length
        mp = ml.get_instrument_mark_price_time_serie(perp_name,rft,dt)
        rho = ml.get_risk_factor_ssvi_rho_time_serie(ml.btcusd,rft,dt)
        _gamma = ml.get_risk_factor_ssvi_gamma_time_serie(ml.btcusd,rft,dt)
        nu = ml.get_risk_factor_ssvi_nu_time_serie(ml.btcusd,rft,dt)
        ssvi_atm_vol = dict()
        for t in self.parameters.atm_t_vector: 
            ssvi_atm_vol[t] = ml.get_risk_factor_atm_volatility(ml.btcusd,rft,dt,t)
        return VolatilityTraderModelInput(mp,rho,_gamma,nu,ssvi_atm_vol)
    
    def update_book(self, reference_time: datetime, initial_book: Book) -> Book:
        market = [m for m in self.loader.markets 
          if m.risk_factor==self.risk_factor
          and m.reference_time==reference_time][0]
        traderparam = VolatilityTraderParameters(
            book = initial_book,
            market=market,
            dt = self.parameters.dt, 
            model_input=self.get_model_input(reference_time), 
            gamma_pnl_target_factor=self.parameters.gamma_pnl_target_factor, 
            vega_pnl_target_factor=self.parameters.vega_pnl_target_factor)
        trader = VolatilityTrader(traderparam)
        return trader.update_book()