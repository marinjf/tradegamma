from typing import List 
from dataclasses import dataclass
import numpy as np 
from datetime import datetime
from abc import abstractmethod, ABC
import pandas as pd 
from src.portfolio import (
    Trade, 
    CashFlow, 
    Portfolio, 
    Position, 
    trades_to_positions, 
    settle_expired_position, 
    settle_book_expired_positions, 
    Book
)
from src.instruments import Option, Currency, Future, PerpetualFuture, Spot
from src.market import Market 
from src.loader import MarketLoader, update_market_loader, get_market_loader
from src.quant.blackscholes import BlackScholes
from src.quant.ssvi import SSVI


class OptionPnlEngine: 

    def __init__(self, book:Book, market:Market): 
        self.book, self.market = book, market
        self.portfolio = self.book.to_portfolio(market)
        self.positions = self.book.to_positions()
        self.spot_quote = self.market.get_quote(self.market.spot.name)
        self.futts = self.market.get_future_term_structure()
        self._set_data()
        self.delta_hedge_trade = self.portfolio.perpetual_delta_hedging_trade()

    def _set_data(self) -> None: 
        sigma = list()
        gamma = list()
        vega = list()
        theta = list()
        delta = list()
        vega = list()
        strike = list()
        t_vector = list()
        F = list()
        for p in self.positions: 
            if p.number_contracts == 0: continue
            pftbis = Portfolio([p], self.market, CashFlow(0, Currency('USD')))
            i = p.instrument
            if not isinstance(i,Option): continue
            name = p.instrument.name
            #if p.number_contracts>=0: 
                #s=self.market.get_implied_volatility_quote(name,'ask')
            #else: 
                #s=self.market.get_implied_volatility_quote(name,'bid')
            s=self.market.get_implied_volatility_quote(name,'mid')
            t = i.time_to_expiry(self.market.reference_time) 
            t_vector.append(t)
            F.append(self.futts.future_price(t))
            sigma.append(s)
            delta.append(pftbis.sensitivities.delta)
            theta.append(pftbis.sensitivities.theta)
            gamma.append(pftbis.sensitivities.gamma)
            vega.append(pftbis.sensitivities.vega)
            strike.append(p.instrument.strike)
        self.t = np.array(t_vector)
        self.sigma = np.array(sigma)
        self.gamma = np.array(gamma)
        self.vega = np.array(vega)
        self.theta = np.array(theta)
        self.delta = np.array(delta)
        self.F = np.array(F)
        self.K = np.array(strike)

    def proxy_theta_gamma_pnl(self, realised_volatility: float, dt:float) -> float:
        s, g, F = self.sigma, self.gamma, self.F
        dreiv = realised_volatility- s
        dreiv_factor = dreiv*(.5*dreiv + s)
        return np.sum(g*(F**2)*dt*dreiv_factor)
    
    def theta_pnl(self, dt:float) -> float: 
        return np.sum(self.theta*dt)
    
    def gamma_pnl(self, spot_relative_change:float) -> float: 
        spot_change = spot_relative_change*self.F
        return np.sum(.5*(spot_change**2)*self.gamma)
    
    def vega_pnl(self, ssvi_forecast:SSVI) -> float:
        k = np.log(self.K/self.F)
        forecasted_sigma = ssvi_forecast.implied_volatility(k, self.t)
        return np.sum(self.vega*(forecasted_sigma-self.sigma))
    
    def delta_pnl(self, spot_relative_change:float) -> float: 
        return np.sum(self.delta*spot_relative_change*self.F)
    
@dataclass
class TraderParameters(ABC): 
    book : Book
    market : Market 

class Trader(ABC): 
    def __init__(self, parameters : TraderParameters): 
        self.book = parameters.book
        self.market = parameters.market

    @abstractmethod
    def update_book(self) -> Book: 
        pass 

@dataclass
class BacktestInput(ABC): 
    deposit_usd : float = 100000
    reload_market : bool = False 
    first_data_point : int = 1000000
    last_data_point : int = 2000000

    def get_market_loader(self) -> MarketLoader: 
        if self.reload_market: 
            update_market_loader(self.first_data_point,self.last_data_point)
        return get_market_loader()  

@dataclass
class BacktestOutput: 
    portfolio_report : pd.DataFrame
    option_position_report : pd.DataFrame
    perp_position_report : pd.DataFrame
    spot_position_report : pd.DataFrame
    last_book : Book
    last_portfolio : Portfolio

    def option_pnl_report(self) -> pd.DataFrame: 
        data = self.option_position_report
        optnames = list(set(list(data.name)))
        output = list()
        for o in optnames: 
            dataset = data[data.name.isin([o])]
            dates = list(dataset.time)
            theta_pnl, vega_pnl, delta_pnl, gamma_pnl, mgamma_pnl = 0,0,0,0,0
            explained_pnl, modified_explained_pnl = 0, 0
            for i in range(1, len(dates)-1): 
                dt = 1/(365*24)
                d = dataset[dataset.time.isin([dates[i-1]])].to_dict('records')[0]
                ddt = dataset[dataset.time.isin([dates[i]])].to_dict('records')[0]
                iv, ivdt = d['iv'], ddt['iv']
                iv_change = ivdt-iv
                f, fdt = d['future_price'], ddt['future_price']
                f_change, iv_change = fdt-f, ivdt-iv
                f_rchange = f_change/f 
                re = np.sqrt((f_rchange**2)/dt)
                theta_pnl = theta_pnl+d['theta']*dt
                vega_pnl  = vega_pnl+d['vega']*iv_change
                delta_pnl = delta_pnl+d['delta']*f_change
                gamma_pnl = gamma_pnl+.5*(f_change**2)*d['gamma']
                mgamma_pnl = mgamma_pnl+d['gamma']*(f**2)*dt*(.5*(re-iv)**2 + iv*(re-iv))
                epnl = d['theta']*dt+d['vega']*iv_change+d['delta']*f_change+.5*(f_change**2)*d['gamma']
                mepnl = d['vega']*iv_change+d['delta']*f_change+d['gamma']*(f**2)*dt*(.5*(re-iv)**2 + iv*(re-iv))
                explained_pnl = explained_pnl+epnl
                modified_explained_pnl = modified_explained_pnl+mepnl
                dict_out = {
                    'time' : dates[i], 
                    'name' : o, 
                    'delta_pnl' : delta_pnl, 
                    'gamma_pnl' : gamma_pnl, 
                    'vega_pnl' : vega_pnl, 
                    'theta_pnl' : theta_pnl, 
                    'approx_gamma_theta_pnl' : mgamma_pnl,
                    'explained_pnl' : explained_pnl, 
                    'approx_explained_pnl' : modified_explained_pnl, 
                    'total_pnl' : ddt['usd_realised']+ ddt['usd_unrealised'],
                    'realised_pnl' : ddt['usd_realised'],
                    'unrealised_pnl' : ddt['usd_unrealised'],
                    'net_fee_total_pnl' : ddt['usd_realised']+ ddt['usd_unrealised']+ddt['usd_fees'],
                }
                output.append(dict_out)
        return pd.DataFrame(output)

    def trades_report(self) -> pd.DataFrame: 
        output = list()
        for p in self.last_portfolio.positions: 
            for t in p.trades: 
                dict_out = {
                    'instrument_name' : t.instrument.name,
                    'time' : t.reference_time, 
                    'traded_price' : t.traded_price,
                    'number_contracts': t.number_contracts, 
                    'trade_fee' : t.trade_fee()
                }
                output.append(dict_out)
        return pd.DataFrame(output)

    def write_full_report(self) -> None: 
        pft = self.portfolio_report
        spt = self.spot_position_report
        perp = self.perp_position_report
        opt = self.option_position_report
        optpnl = self.option_pnl_report()
        trades = self.trades_report()
        tmsp = round(datetime.now().timestamp())
        path = 'data/export/'+str(tmsp)+'_backtest_report.xlsx'
        with pd.ExcelWriter(path) as writer: 
            pft.to_excel(writer, sheet_name='Portfolio')
            trades.to_excel(writer, sheet_name='Trades')
            opt.to_excel(writer, sheet_name='Options Position')
            perp.to_excel(writer, sheet_name='Perpetual Position')
            spt.to_excel(writer, sheet_name='Spot Position')
            optpnl.to_excel(writer, sheet_name='Option P&L breakdown')

class BacktestTrader(ABC): 
    def __init__(self, inputdata : BacktestInput): 
        self.inputdata = inputdata
        self.loader = inputdata.get_market_loader()
        self.deposit = CashFlow(inputdata.deposit_usd,self.loader.usd)
        self.risk_factor = self.loader.btcusd
        self.dates = self.get_date_vector_for_backtest(self.loader.dates_dt)
        
    def initialize_book(self) -> Book: 
        date0 = self.dates[0]
        market = [m for m in self.loader.markets 
          if m.risk_factor==self.risk_factor
          and m.reference_time==date0][0]
        spot_instrument = market.spot 
        spot_quote = market.get_quote(spot_instrument.name)
        quantity_to_trade = self.deposit.amount/spot_quote.order_book.best_ask
        initial_trade = Trade(spot_instrument, quantity_to_trade,
                            spot_quote.order_book.best_ask,
                            self.loader.usd,date0)
        return Book([initial_trade], self.deposit)

    @abstractmethod
    def get_date_vector_for_backtest(self, dates:List[datetime]) -> List[datetime]: 
        pass

    @abstractmethod
    def update_book(self, reference_time:datetime, initial_book:Book) -> Book: 
        pass
    
    def portfolio_report(self, portfolio:Portfolio) -> dict: 
        ref_date = portfolio.market.reference_time
        margin = portfolio.get_margin()
        spot = portfolio.spot_quote.order_book.mid
        n_open_position = len([p for p in portfolio.positions if p.number_contracts!=0])
        return {
            'time' : ref_date, 
            'total_usd_pnl' : portfolio.get_usd_total_pnl(), 
            'realised_usd_pnl': portfolio.get_usd_realised_pnl(), 
            'unrealised_usd_pnl' : portfolio.get_usd_unrealised_pnl(), 
            'initial_margin_usd': margin.initial*spot, 
            'maintenance_margin_usd' : margin.maintenance*spot, 
            'delta' : portfolio.sensitivities.delta, 
            'gamma' : portfolio.sensitivities.gamma, 
            'vega' : portfolio.sensitivities.vega, 
            'theta' : portfolio.sensitivities.theta, 
            'is_cash_sufficient' : portfolio.is_cash_sufficient(), 
            'total_fee_usd' : portfolio.get_usd_fee(), 
            'number_open_position': n_open_position
        }

    def option_position_report(self, portfolio:Portfolio) -> List[dict]: 
        output = list()
        market = portfolio.market
        spot = portfolio.spot_quote.order_book.mid
        futts = market.get_future_term_structure()
        for p in portfolio.positions: 
            if isinstance(p.instrument, Option): #and p.number_contracts!=0: 
                i = p.instrument
                if p.number_contracts!=0:
                    quote = market.get_quote(i.name)
                    sigma = market.get_implied_volatility_quote(i.name,'mid')
                    t = i.time_to_expiry(market.reference_time)
                    F = futts.future_price(t)
                    mark_price = quote.order_book.mark_price
                else: 
                    mark_price = np.nan 
                    sigma, t, F = np.nan, np.nan, np.nan
                if i.call_or_put == 'C': put_or_call=1
                else: put_or_call=-1
                bs = BlackScholes(
                    S = F, K = i.strike, t = t, sigma = sigma, 
                    q = 0, r=0, future=0, call_or_put=put_or_call)
                q = p.number_contracts*i.contract_size
                dict_out = {
                    'time': market.reference_time,
                    'name': i.name, 
                    'position' : p.number_contracts, 
                    'traded_price' : p.fifo_price,
                    'mark_price' : mark_price,
                    'delta' : q*bs.delta(),
                    'gamma' : q*bs.gamma(),
                    'vega' : q*bs.vega(),
                    'theta' : q*bs.theta(),
                    't' : t, 
                    'strike' : i.strike, 
                    'iv' : sigma,
                    'future_price' : F,
                    'usd_unrealised' : portfolio.get_position_usd_unrealised_pnl(p), 
                    'usd_realised': p.get_realised_pnl()*spot, 
                    'usd_fees': p.get_fee_cash_flow().amount*spot     
                }
                output.append(dict_out)
            else: continue
        return output
    
    def perp_position_report(self, portfolio:Portfolio) -> List[dict]: 
        output = list()
        market = portfolio.market
        spot = portfolio.spot_quote.order_book.mid
        for p in portfolio.positions: 
            if p.number_contracts != 0 and isinstance(p.instrument, PerpetualFuture): 
                i = p.instrument
                quote = market.get_quote(i.name)
                dict_out = {
                    'time': market.reference_time,
                    'name': i.name, 
                    'position' : p.number_contracts, 
                    'traded_price' : p.fifo_price,
                    'mark_price' : quote.order_book.mark_price, 
                    'usd_realised': p.get_realised_pnl()*spot,
                    'usd_unrealised' : portfolio.get_position_usd_unrealised_pnl(p),    
                }
                output.append(dict_out)
            else: continue
        return output
    
    def spot_position_report(self, portfolio:Portfolio) -> List[dict]: 
        output = list()
        market = portfolio.market
        for p in portfolio.positions: 
            if p.number_contracts != 0 and isinstance(p.instrument, Spot): 
                i = p.instrument
                quote = market.get_quote(i.name)
                dict_out = {
                    'time': market.reference_time,
                    'name': i.name, 
                    'position' : p.number_contracts, 
                    'traded_price' : p.fifo_price,
                    'mark_price' : quote.order_book.mark_price, 
                    'usd_realised': p.get_realised_pnl(),
                    'usd_unrealised' : portfolio.get_position_usd_unrealised_pnl(p),    
                }
                output.append(dict_out)
            else: continue
        return output
    
    def launch_backtest(self) -> BacktestOutput: 
        i = 0
        book = self.initialize_book()
        portfolio_report = list()
        option_position_report = list()
        perp_position_report = list()
        spot_position_report = list()
        for d in self.dates: 
            print(d)
            book = self.update_book(d, book)
            market = [m for m in self.loader.markets 
                    if m.risk_factor==self.risk_factor
                    and m.reference_time==d][0]
            pft = book.to_portfolio(market)
            portfolio_report.append(self.portfolio_report(pft))
            option_position_report = option_position_report + self.option_position_report(pft)
            perp_position_report = perp_position_report + self.perp_position_report(pft)
            spot_position_report = spot_position_report + self.spot_position_report(pft)
            i = i +1
            perc = round(100*i/len(self.dates),2)
            print(str(perc) + '% of the backtest')
        print('Backtest is done.')
        output = BacktestOutput(
            portfolio_report=pd.DataFrame(portfolio_report), 
            option_position_report=pd.DataFrame(option_position_report), 
            perp_position_report=pd.DataFrame(perp_position_report), 
            spot_position_report=pd.DataFrame(spot_position_report),
            last_book=book,
            last_portfolio=pft)
        return output 
        
