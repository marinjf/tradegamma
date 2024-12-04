from dataclasses import dataclass
from datetime import datetime 
from typing import List
from abc import ABC, abstractmethod
import math 

def deribit_date_to_datetime(deritbit_date: str) -> datetime: 
    hour = 8 
    mapping_deribit_month_number = {
    'JAN' : 1, 'FEB' : 2, 'MAR' : 3, 'APR' : 4, 
    'MAY' : 5, 'JUN' : 6, 'JUL' : 7, 'AUG' : 8, 
    'SEP' : 9, 'OCT' : 10, 'NOV' : 11, 'DEC' : 12}
    if len(deritbit_date)==6: 
        day = int(deritbit_date[0:1])
        year = int('20'+deritbit_date[4:6])
        month = mapping_deribit_month_number[deritbit_date[1:4]]
    else: 
        day = int(deritbit_date[0:2])
        year = int('20'+deritbit_date[5:7])
        month = mapping_deribit_month_number[deritbit_date[2:5]]
    return datetime(year,month,day,hour)

@dataclass
class Currency:
    code : str 

@dataclass
class RiskFactor:
    base_currency : Currency
    quote_currency : Currency
    
    def __post_init__(self): 
        self.code = self.base_currency.code+self.quote_currency.code

@dataclass
class Instrument(ABC): 
    name : str 
    risk_factor : RiskFactor
    underlying_name : str
    contract_size : float 
    contract_size_currency : Currency
    minimum_contract_amount : float


@dataclass 
class PerpetualFuture(Instrument): 
    pass 

@dataclass
class Future(Instrument): 

    def __post_init__(self):
        self.split_name = self.name.split('-')
        self.expiry_dt = deribit_date_to_datetime(self.split_name[1]) 

    def time_to_expiry(self, reftime:datetime) -> float: 
        delta = self.expiry_dt - reftime
        year = 365*24*60*60
        return delta.total_seconds()/year

@dataclass
class Spot(Instrument): 
    pass 
    
@dataclass
class Option(Instrument): 

    def __post_init__(self):
        self.split_name = self.name.split('-')
        self.expiry_dt = deribit_date_to_datetime(self.split_name[1]) 
        self.strike = int(self.split_name[2])
        self.call_or_put = self.split_name[3]
    
    def time_to_expiry(self, reftime:datetime) -> float: 
        delta = self.expiry_dt - reftime
        year = 365*24*60*60
        return delta.total_seconds()/year

def compute_spread(bid:float, ask:float) -> float: 
    try : return (ask-bid)/bid
    except ZeroDivisionError as e: return math.nan

@dataclass 
class OrderBook:
    quote_currency : Currency
    mark_price : float
    best_bid : float 
    best_ask : float 
    best_bid_size : float 
    best_ask_size : float 
    asks : List[float]
    bids : List[float]

    def __post_init__(self): 
        self.mid = .5*(self.best_bid+self.best_ask)
        self.spread = compute_spread(self.best_bid,self.best_ask)

@dataclass
class Sensitivities: 
    delta : float 
    gamma : float 
    theta : float 
    vega : float 

@dataclass
class InstrumentQuote: 
    reference_time : datetime
    instrument_name : str 
    order_book : OrderBook
    volume_usd : float
    open_interest : float 
    sensitivities : Sensitivities
    bid_iv : float 
    ask_iv : float 

    def __post_init__(self): 
        self.mid_iv = .5*(self.bid_iv + self.ask_iv)


