from dataclasses import dataclass
import operator
from datetime import datetime
from typing import List
import numpy as np
from src.instruments import (
    Currency, 
    Instrument, 
    Spot, 
    Option, 
    PerpetualFuture, 
    Future, 
    Sensitivities
)
from src.market import Market
from src.quant.blackscholes import BlackScholes

@dataclass
class CashFlow: 
    amount: float 
    currency : Currency

@dataclass
class Margin: 
    initial : float 
    maintenance : float 
    currency : Currency

@dataclass
class Trade:
    instrument : Instrument 
    number_contracts : float
    traded_price : float
    currency : Currency
    reference_time : datetime

    def _set_premium(self) -> float: 
        if isinstance(self.instrument, Future): 
            return 0 
        elif isinstance(self.instrument, PerpetualFuture): 
            return 0 
        else: 
            return -self.number_contracts*self.traded_price

    def get_size(self) -> float: 
        n = self.number_contracts
        cz = self.instrument.contract_size
        if isinstance(self.instrument,Spot): return abs(n)
        if isinstance(self.instrument,Option): return abs(n)
        if isinstance(self.instrument,Future): 
            return abs(n*cz/self.traded_price)
        if isinstance(self.instrument,PerpetualFuture): 
            return abs(n*cz/self.traded_price)

    def trade_fee(self) -> CashFlow: 
        ccy = self.instrument.risk_factor.base_currency
        if isinstance(self.instrument,Spot): 
            return CashFlow(0,ccy)
        if isinstance(self.instrument,Option): 
            cap = 0.125*abs(self.number_contracts)*self.traded_price
            amount = max(-0.03*self.get_size()/100, -cap)
            return CashFlow(amount,ccy)
        if isinstance(self.instrument,Future): 
            return CashFlow(-0.05*self.get_size()/100,ccy)
        if isinstance(self.instrument,PerpetualFuture):  
            return CashFlow(-0.05*self.get_size()/100,ccy)

@dataclass
class Position: 
    trades : List[Trade]

    def __post_init__(self): 
        self.number_contracts = self.get_net_position()
        self.instrument = self.trades[0].instrument
        self.price_currency = self.trades[0].currency
        self.fifo_price = self.get_fifo_price()
        self.premium = self.get_premium()

    def get_premium(self) -> float: 
        if isinstance(self.instrument, Future): 
            return 0 
        elif isinstance(self.instrument, PerpetualFuture): 
            return 0 
        else: 
            return -self.number_contracts*self.fifo_price

    def get_net_position(self) -> float: 
        return round(sum([t.number_contracts for t in self.trades]),5)
    
    def get_fifo_price(self) -> float: 
        trades = sorted(self.trades, 
                        key=operator.attrgetter('reference_time'),
                        reverse=True)
        xmax, xmin = self.number_contracts,self.number_contracts
        cfmax, cfmin = list(), list()
        for t in trades: 
            qmax = max(xmax-max(t.number_contracts,0), 0) 
            qmin = min(xmin-min(t.number_contracts,0), 0)
            deltamax, deltamin = xmax-qmax, xmin-qmin
            cfmax.append(deltamax*t.traded_price), cfmin.append(deltamin*t.traded_price)
            xmax, xmin = qmax, qmin
        if self.number_contracts == 0: return 0 
        elif self.number_contracts>0: return sum(cfmax)/self.number_contracts
        elif self.number_contracts<0: return sum(cfmin)/self.number_contracts
    
    def get_premium_cash_flow(self) -> CashFlow: 
        return CashFlow(
            self.get_premium(), 
            self.price_currency)
    
    def get_fee_cash_flow(self) -> CashFlow: 
        amount = sum([t.trade_fee().amount for t in self.trades]) 
        ccy = [t.trade_fee().currency.code for t in self.trades][0]
        return CashFlow(amount, Currency(ccy))
    
    def get_realised_pnl(self) -> float: 
        if isinstance(self.instrument, PerpetualFuture):
            c = self.instrument.contract_size
            total_cash_flow = sum([t.number_contracts*c/t.traded_price for t in self.trades])
            if self.number_contracts == 0: premium = 0  
            else: premium = -self.number_contracts*c/self.fifo_price 
        else: 
            total_cash_flow = sum([-t.number_contracts*t.traded_price for t in self.trades])
            premium = self.number_contracts*self.fifo_price
        return (total_cash_flow+premium)

    def get_realised_pnl_cash_flow(self) -> CashFlow: 
        if isinstance(self.instrument, PerpetualFuture):
            return CashFlow(self.get_realised_pnl(),self.instrument.risk_factor.base_currency)
        else: 
            return CashFlow(self.get_realised_pnl(),self.price_currency)  
    
    def to_trade(self, reference_time:datetime) -> Trade: 
        return Trade(
            instrument = self.instrument, 
            number_contracts=self.number_contracts, 
            traded_price=self.fifo_price,
            currency=self.price_currency, 
            reference_time=reference_time)

    def get_settlement_trade(
            self, 
            settlement_price: float, 
            reference_time:datetime) -> Trade:
        return Trade(
            instrument = self.instrument, 
            number_contracts=-self.number_contracts, 
            traded_price=settlement_price,
            currency=self.price_currency, 
            reference_time=reference_time) 
    
    def get_cash_flows(self) -> List[CashFlow]: 
        return [
            self.get_fee_cash_flow(), 
            self.get_realised_pnl_cash_flow(), 
            self.get_premium_cash_flow()
        ]

def trades_to_positions(trades : List[Trade]) -> List[Position]: 
    instrument_names = list(set([t.instrument.name for t in trades]))
    output = list()
    for n in instrument_names: 
        trade = [t for t in trades if t.instrument.name==n]
        output.append(Position(trade))
    return output

def settle_expired_position(position:Position, market:Market) -> Position:
    spot_quote = market.get_quote(market.spot.name)
    i = position.instrument
    S = spot_quote.order_book.mark_price 
    if isinstance(i,Future): 
        exp, ref = i.expiry_dt, market.reference_time
        if position.number_contracts!=0 and exp==ref: 
            settle_trade = position.get_settlement_trade(S, ref)
            new_trades = position.trades.copy()
            new_trades.append(settle_trade)
            return Position(new_trades)
        else: return position
    elif isinstance(i,Option): 
        exp, ref = i.expiry_dt, market.reference_time
        if position.number_contracts!=0 and exp==ref: 
            K = i.strike
            if i.call_or_put=='C': x = 1 
            else: x=-1
            price = max(x*(S-K), 0)/S
            settle_trade = position.get_settlement_trade(price, ref)
            new_trades = position.trades.copy()
            new_trades.append(settle_trade)
            return Position(new_trades)
        else: return position
    else: return position

@dataclass
class Portfolio: 
    positions : List[Position] 
    market : Market 
    initial_deposit : CashFlow 

    def __post_init__(self): 
        self.spot_quote = self.market.get_quote(self.market.spot.name)
        self.sensitivities = self.get_usd_sensitivities('mid')

    def get_usd_sensitivities(self, quote_type:str) -> Sensitivities: 
        delta_exposure,gamma_exposure = list(), list()
        vega_exposure, theta_exposure = list(), list()
        futts = self.market.get_future_term_structure()
        for p in self.positions: 
            if p.number_contracts==0: continue
            q = p.number_contracts*p.instrument.contract_size
            if Option.__instancecheck__(p.instrument): 
                strike = p.instrument.strike
                t = p.instrument.time_to_expiry(self.market.reference_time)
                futprice = futts.future_price(t)
                sigma = self.market.get_implied_volatility_quote(
                    p.instrument.name,quote_type)
                if p.instrument.call_or_put == 'C': put_or_call = 1
                else:  put_or_call = -1
                bs = BlackScholes(
                    S = futprice, K = strike, t = t, sigma = sigma, 
                    q = 0, r=0, future=0, call_or_put=put_or_call)
                delta_exposure.append(q*bs.delta())
                gamma_exposure.append(q*bs.gamma())
                vega_exposure.append(q*bs.vega())
                theta_exposure.append(q*bs.theta())
            elif Spot.__instancecheck__(p.instrument): 
                delta_exposure.append(q)
                gamma_exposure.append(0)
                vega_exposure.append(0) 
                theta_exposure.append(0)
            else: 
                d =  q/self.spot_quote.order_book.mid
                delta_exposure.append(d)
                gamma_exposure.append(0)
                vega_exposure.append(0) 
                theta_exposure.append(0)
        return Sensitivities(
            delta = sum(delta_exposure), 
            gamma = sum(gamma_exposure), 
            theta = sum(theta_exposure), 
            vega = sum(vega_exposure))
    
    def get_cash_account(self) -> List[CashFlow]: 
        deposit = self.initial_deposit
        output_dict = {deposit.currency.code : deposit.amount}
        for p in self.positions: 
            cash_flows = p.get_cash_flows()
            for cf in cash_flows: 
                ccy, amount = cf.currency.code, cf.amount
                if ccy in output_dict:
                    output_dict[ccy] = output_dict[ccy]+amount
                else: output_dict[ccy] = amount
            if isinstance(p.instrument, Spot): 
                ccyspot = p.instrument.risk_factor.base_currency.code
                if ccyspot in output_dict:
                    output_dict[ccyspot] = output_dict[ccyspot]+p.number_contracts
                else: output_dict[ccyspot] = p.number_contracts
        return [CashFlow(round(output_dict[o],5),Currency(o)) for o in output_dict]

    def perpetual_delta_hedging_trade(self) -> Trade: 
        perp = self.market.perpetual
        quote = self.market.get_quote(perp.name)
        exposure = self.sensitivities.delta
        if exposure<=0: price = quote.order_book.best_ask
        else: price = quote.order_book.best_bid
        quantity = round(exposure*price/perp.contract_size)
        return Trade(
            instrument = perp,
            number_contracts=-quantity,
            traded_price=price,
            currency=quote.order_book.quote_currency, 
            reference_time=self.market.reference_time
        ) 

    def get_margin(self) -> Margin: 
        im, mm = list(), list()
        base_ccy = self.market.risk_factor.base_currency
        futts = self.market.get_future_term_structure()
        for p in self.positions:
            if p.number_contracts==0: continue
            instrument = p.instrument        
            if isinstance(instrument,Spot): mm_rate,im_rate,size = 0,0,0
            if isinstance(instrument,Option): 
                size = p.number_contracts*instrument.contract_size
                if size>=0: im_rate, mm_rate = 0,0
                else: 
                    quote = self.market.get_quote(instrument.name)
                    t = instrument.time_to_expiry(self.market.reference_time)
                    ump = futts.future_price(t)
                    mp = quote.order_book.mark_price
                    K = instrument.strike
                    if instrument.call_or_put == 'C': 
                        otm_amount = max(ump-K, 0) 
                        mm_rate = 0.075 + mp
                        im_rate = max(0.15-otm_amount/ump,0.1)+mp
                    else: 
                        otm_amount = max(K-ump, 0) 
                        mm_rate = max(0.075,0.075*mp)+mp
                        im_rate = max(max(0.15-otm_amount/ump,0.1)+mp,mm_rate)
            if isinstance(instrument,Future): 
                spot_price = self.spot_quote.order_book.mid
                size = p.number_contracts*instrument.contract_size/spot_price
                if base_ccy == Currency('ETH'):
                    im_rate = 0.04 + abs(size)*0.004/100
                    mm_rate = 0.02 + abs(size)*0.004/100
                if base_ccy == Currency('BTC'):
                    im_rate = 0.04 + abs(size)*0.005/100
                    mm_rate = 0.02 + abs(size)*0.005/100
            if isinstance(instrument,PerpetualFuture): 
                spot_price = self.spot_quote.order_book.mid
                size = p.number_contracts*instrument.contract_size/spot_price
                if base_ccy == Currency('ETH'):
                    im_rate = 0.02 + abs(size)*0.004/100
                    mm_rate = 0.01 + abs(size)*0.004/100
                if base_ccy == Currency('BTC'):
                    im_rate = 0.02 + abs(size)*0.005/100
                    mm_rate = 0.01 + abs(size)*0.005/100
            im.append(im_rate*abs(size))
            mm.append(mm_rate*abs(size))
        return Margin(sum(im),sum(mm),self.market.risk_factor.base_currency)

    def is_cash_sufficient(self) -> bool: 
        margin = self.get_margin()
        try:
            cash_account = [ca for ca in self.get_cash_account() 
                            if ca.currency==margin.currency][0]
        except IndexError: 
            cash_account = CashFlow(0,margin.currency)
        if cash_account.amount - margin.initial > margin.maintenance: 
            return True
        else: return False
    
    def get_usd_cash_account(self) -> float: 
        output = 0 
        for c in self.get_cash_account(): 
            if c.currency != Currency('USD'): 
                output = output + c.amount*self.spot_quote.order_book.mid
            else: output = output + c.amount
        margin = self.get_margin()
        if margin.currency != Currency('USD'): 
                output = output - margin.initial*self.spot_quote.order_book.mid
        else: output = output - margin.initial
        return output
    
    def get_position_usd_unrealised_pnl(self, position:Position) -> float: 
        if position.number_contracts==0: return 0
        spot_price = self.spot_quote.order_book.mid
        i = position.instrument
        if isinstance(i, PerpetualFuture):
            n = position.number_contracts*i.contract_size/spot_price
        else:
            n = position.number_contracts*i.contract_size
        fifo_price = position.fifo_price
        quote = self.market.get_quote(i.name)
        if n<0: quoted_price = quote.order_book.mark_price
        else: quoted_price = quote.order_book.mark_price
        if quote.order_book.quote_currency != Currency('USD'):
            return n*(quoted_price-fifo_price)*spot_price
        else: return n*(quoted_price-fifo_price)

    def get_usd_unrealised_pnl(self) -> float: 
        unrealised_pnl = list()
        for p in self.positions: 
            if p.number_contracts == 0: unrealised_pnl.append(0)
            else: unrealised_pnl.append(self.get_position_usd_unrealised_pnl(p))
        return sum(unrealised_pnl)
    
    def get_usd_realised_pnl(self) -> float: 
        realised_pnl = 0
        spot_price = self.spot_quote.order_book.mid
        for p in self.positions: 
            rpnl = p.get_realised_pnl_cash_flow()
            if rpnl.currency!=Currency('USD'): 
                realised_pnl = realised_pnl + spot_price*rpnl.amount
            else: realised_pnl = realised_pnl + rpnl.amount
        return realised_pnl

    def get_usd_fee(self) -> float: 
        output = 0 
        spot_price = self.spot_quote.order_book.mid
        for p in self.positions: 
            feecf = p.get_fee_cash_flow()
            if feecf.currency!=Currency('USD'): 
                output = output + spot_price*feecf.amount
            else: output = output + feecf.amount
        return output 
    
    def get_usd_total_pnl(self) -> float: 
        return self.get_usd_fee() + self.get_usd_realised_pnl() \
            + self.get_usd_unrealised_pnl()

    def get_usd_total_value(self) -> float:
        return self.initial_deposit.amount + self.get_usd_total_pnl()

@dataclass
class Book: 
    trades : List[Trade]
    initial_deposit : CashFlow

    def to_positions(self) -> List[Position]: 
        return trades_to_positions(self.trades)

    def to_portfolio(self, market:Market) -> Portfolio: 
        return Portfolio(self.to_positions(),market, self.initial_deposit)

def settle_book_expired_positions(book:Book, market:Market) -> Book: 
    new_position = [settle_expired_position(p, market) for p in book.to_positions()]
    trades = list()
    for p in new_position: 
        trades = trades + p.trades
    return Book(trades, book.initial_deposit)
