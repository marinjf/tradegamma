from dataclasses import dataclass
from typing import List
import pandas as pd 
import math 
from datetime import datetime, timedelta
import numpy as np 
import scipy.interpolate
import matplotlib.pyplot as plt 
from src.instruments import (
    InstrumentQuote, 
    Option, 
    Spot, 
    Future, 
    PerpetualFuture, 
    RiskFactor, 
)
from src.quant.ssvi import SSVI, CalibrateSSVI
from src.tools import update_dict_of_list

@dataclass
class VolatilitySurface: 
    reference_time : datetime 
    risk_factor : RiskFactor
    ssvi : SSVI

@dataclass
class FutureTermStructure: 
    reference_time : datetime 
    risk_factor : RiskFactor
    datamap : dict[float, float]

    def __post_init__(self): 
        self.datamap = dict(sorted(self.datamap.items()))
    
    def future_price(self, t:float): 
        t_vec = np.array(list(self.datamap.keys())) 
        max_t_vec = np.max(t_vec)
        t = min(t, max_t_vec)
        v = np.array(list(self.datamap.values())) 
        interp = scipy.interpolate.interp1d(t_vec,v)
        return interp(t)

@dataclass
class ATMChainQuote: 
    calls : List[Option] 
    puts : List[Option]
    mapped_quotes : dict[str, InstrumentQuote]

@dataclass
class Market: 
    reference_time : datetime 
    risk_factor : RiskFactor
    quotes : List[InstrumentQuote]
    spot : Spot 
    perpetual : PerpetualFuture
    futures : List[Future]
    options : List[Option]

    def __post_init__(self): 
        self._set_option_data()
        self.liquid_options = self.get_liquid_option()
        self._set_advanced_option_data()
        self.atmtvarmap = self.get_atmtvar_map()
        self.atm_factor = self.get_atm_factor()
        self.atm_chain = self.get_atm_chain()
        self.volatility_surface = self.get_volatility_surface()
    
    def _set_option_data(self) -> None: 
        self.option_expiries = list()
        self.option_names = list()
        self.put_options_name = list()
        self.call_options_name = list()
        self.mapped_expiries_option_names = dict()
        for o in self.options: 
            self.option_names.append(o.name)
            if o.expiry_dt not in self.option_expiries:
                self.option_expiries.append(o.expiry_dt)
            if o.call_or_put == 'C': 
                self.call_options_name.append(o.name)
            else: self.put_options_name.append(o.name)
            self.mapped_expiries_option_names = update_dict_of_list(
                o.expiry_dt,o.name,self.mapped_expiries_option_names)
    
    def _set_advanced_option_data(self) -> None: 
        self.mapped_liquid_strikes = dict()
        self.mapped_liquid_exptime = dict()
        self.mapped_liquid_futprice = dict()
        self.mapped_liquid_ivquote = dict()
        fut_terms = self.get_future_term_structure()
        for o in self.liquid_options: 
            quote = self.get_quote(o.name)
            expiry = o.expiry_dt
            t = o.time_to_expiry(self.reference_time)
            price = fut_terms.future_price(t)
            iv = quote.mid_iv/100
            self.mapped_liquid_strikes = update_dict_of_list(
                expiry,o.strike,self.mapped_liquid_strikes)
            self.mapped_liquid_exptime = update_dict_of_list(
                expiry,t,self.mapped_liquid_exptime)
            self.mapped_liquid_futprice = update_dict_of_list(
                expiry,price,self.mapped_liquid_futprice)
            self.mapped_liquid_ivquote = update_dict_of_list(
                expiry,iv,self.mapped_liquid_ivquote)
    
    def get_quote(self, instrument_name:str) -> InstrumentQuote: 
        return [q for q in self.quotes if q.instrument_name==instrument_name][0]
    
    def get_option(self, instrument_name:str) -> Option: 
        return [q for q in self.options if q.name==instrument_name][0]
       
    def get_future_term_structure(self) -> FutureTermStructure: 
        futname = [f.name for f in self.futures]
        futexp = [f.time_to_expiry(self.reference_time) for f in self.futures]
        filterquotes = [q for q in self.quotes if q.instrument_name in futname]
        futprices = [q.order_book.mark_price for q in filterquotes]
        futexp.insert(0,0)
        perp_quote = self.get_quote(self.perpetual.name)
        futprices.insert(0,perp_quote.order_book.mark_price)
        datamap = dict(zip(futexp,futprices))
        return FutureTermStructure(self.reference_time,self.risk_factor,datamap)
    
    def get_atm_chain(self) -> ATMChainQuote: 
        calls = list()
        puts = list()
        mapped_quotes = dict()
        for e in self.option_expiries:
            if e <= self.reference_time: continue
            opt_name = self.mapped_expiries_option_names[e]
            mapped_put_delta = {q.instrument_name: q.sensitivities.delta 
                                for q in self.quotes 
                                if q.instrument_name in opt_name 
                                and q.instrument_name in self.put_options_name}
            mapped_call_delta = {q.instrument_name: q.sensitivities.delta 
                                 for q in self.quotes
                                 if q.instrument_name in opt_name 
                                 and q.instrument_name in self.call_options_name}
            call_diff_to_atm = [abs(mapped_call_delta[k]-0.5)
                               for k in list(mapped_call_delta.keys())]
            put_diff_to_atm = [abs(mapped_put_delta[k]+0.5)
                               for k in list(mapped_put_delta.keys())]
            call_name = [k for k in list(mapped_call_delta.keys())
                         if abs(mapped_call_delta[k]-0.5) == min(call_diff_to_atm)][0]
            put_name = [k for k in list(mapped_put_delta.keys())
                         if abs(mapped_put_delta[k]+0.5) == min(put_diff_to_atm)][0]
            calls.append(self.get_option(call_name))
            puts.append(self.get_option(put_name))
            mapped_quotes[call_name] = self.get_quote(call_name)
            mapped_quotes[put_name] = self.get_quote(put_name)
        return ATMChainQuote(calls, puts, mapped_quotes)
               
    def get_liquid_option(self, n = 5) -> List[Option]: 
        filtered_names = list()
        quotes = [q for q in self.quotes 
                  if q.instrument_name in self.option_names
                  and not np.isnan(q.mid_iv)
                  and not np.isnan(q.order_book.spread)
                  and q.order_book.spread > 0
                  and q.ask_iv>0
                  and q.bid_iv>0]
        for e in self.option_expiries: 
            opt_name = self.mapped_expiries_option_names[e]
            put_spreads = {q.instrument_name: q.order_book.spread for q in quotes
                           if q.instrument_name in opt_name 
                           and q.instrument_name in self.put_options_name}
            call_spreads = {q.instrument_name: q.order_book.spread for q in quotes
                            if q.instrument_name in opt_name 
                            and q.instrument_name in self.call_options_name}
            put_spreads = dict(sorted(
                put_spreads.items(), 
                key=lambda item: item[1]))
            call_spreads = dict(sorted(
                call_spreads.items(), 
                key=lambda item: item[1]))
            if len(put_spreads)>=n: filt_put_names = list(put_spreads.keys())[0:n]
            else: filt_put_names = list(put_spreads.keys())
            if len(call_spreads)>=n: filt_call_names = list(call_spreads.keys())[0:n]
            else: filt_call_names = list(call_spreads.keys())
            filtered_names = filtered_names+filt_put_names
            filtered_names = filtered_names+filt_call_names
        return [o for o in self.options if o.name in filtered_names]

    def get_atm_factor(self) -> float: 
        t_vec = [k for k in list(self.atmtvarmap.keys())]
        v = [self.atmtvarmap[t] for t in t_vec]
        l = scipy.stats.linregress(np.array(t_vec), np.array(v))
        return np.sqrt(l.slope.item())

    def get_atmtvar_map(self) -> dict[float, float]: 
        output = dict()
        output[0] = 0
        time_to_expiry_limit = 5*60*60/(365*24*60*60)
        for e in list(self.mapped_liquid_strikes.keys()): 
            t_vec = self.mapped_liquid_exptime[e]
            t = t_vec[0]
            if t<time_to_expiry_limit: 
                continue
            K = self.mapped_liquid_strikes[e]
            F = self.mapped_liquid_futprice[e]
            k = np.log(np.array(K)/np.array(F))
            iv = self.mapped_liquid_ivquote[e]
            if len(iv)==1: 
                output[t] = t*(iv**2)
                continue
            k_iv_map = dict(zip(k,iv))
            k_iv_map = dict(sorted(k_iv_map.items()))
            k = list(k_iv_map.keys())
            iv = list(k_iv_map.values())
            if np.all(np.array(k)>0): 
                output[t] = t*k_iv_map[min(k)]**2
                continue 
            if np.all(np.array(k)<0): 
                output[t] = t*k_iv_map[max(k)]**2
                continue 
            #print(k), print(iv)
            interp = scipy.interpolate.interp1d(np.array(k), np.array(iv))
            output[t] = t*(interp(0).item()**2)
        return dict(sorted(output.items())) 
    
    def get_volatility_surface(self) -> VolatilitySurface: 
        k, sigma, t = list(), list(), list()
        for e in list(self.mapped_liquid_strikes.keys()): 
            K = self.mapped_liquid_strikes[e]
            F = self.mapped_liquid_futprice[e]
            k = k + list(np.log(np.array(K)/np.array(F)))
            t = t + self.mapped_liquid_exptime[e]
            sigma = sigma + self.mapped_liquid_ivquote[e]
        ssvicalibrator = CalibrateSSVI(
            np.array(sigma),
            k,
            np.array(t),
            self.get_atmtvar_map())
        return VolatilitySurface(
            self.reference_time,
            self.risk_factor,
            ssvicalibrator.calibrate())

    def check_ssvi_fit(self) -> None: 
        volsurface = self.get_volatility_surface()
        ssvi = volsurface.ssvi
        fut_terms = self.get_future_term_structure()
        expiries = list(set([o.expiry_dt for o in self.liquid_options]))
        for e in expiries: 
            opt = [o for o in self.liquid_options if o.expiry_dt==e]
            opt_names = [o.name for o in opt]
            quotes = [self.get_quote(n) for n in opt_names]
            t = opt[0].time_to_expiry(self.reference_time)
            K = [o.strike for o in opt]
            F = fut_terms.future_price(t).item()
            k = np.log(np.array(K)/F)
            iv = [q.mid_iv/100 for q in quotes]
            k_iv_map = dict(zip(k,iv))
            k_iv_map = dict(sorted(k_iv_map.items()))
            k = list(k_iv_map.keys())
            iv = list(k_iv_map.values())
            plt.plot(k, iv)
            plt.plot(k, ssvi.implied_volatility(np.array(k),t))
            plt.legend(['Market', 'SSVI'])
            plt.title(self.risk_factor.code + ' : ' + str(t))
            plt.show()
   
    def get_implied_volatility_quote(self, option_name:str, quote_type:str) -> float: 
        quote = self.get_quote(option_name) 
        match quote_type: 
            case 'bid': sigma = quote.bid_iv/100 
            case 'mid': sigma = quote.mid_iv/100 
            case 'ask': sigma = quote.ask_iv/100  
        if sigma == 0: 
            volsurf = self.volatility_surface
            option = self.get_option(option_name)
            t = option.time_to_expiry(self.reference_time)
            K = option.strike
            futts = self.get_future_term_structure()
            F = futts.future_price(t)
            k = np.log(K/F)
            sigma = volsurf.ssvi.implied_volatility(k,t)
        return sigma