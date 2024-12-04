from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np 
from typing import List
from arch.univariate.base import ARCHModelResult
import matplotlib.pyplot as plt
from arch.univariate import (
    ZeroMean, 
    GARCH, 
    SkewStudent, 
    StudentsT, 
    Normal, 
    EGARCH, 
    ARX)
from arch.univariate.volatility import VarianceForecast

@dataclass
class TimeSerie2:
    datamap : dict[datetime, float]

    def __post_init__(self): 
        self.datamap = dict(sorted(self.datamap.items()))
        self.log_difference = self.get_log_difference()
    
    def get_log_difference(self) -> np.array: 
        data = np.log(np.array(list(self.datamap.values())))
        return np.diff(data)
    
    def mean_log_difference(self) -> float: 
        return np.mean(self.log_difference())
    
    def std_log_difference(self) -> float: 
        return np.std(self.log_difference())
    
    def z_score_log_difference(self, ld: np.array) -> np.array: 
        return (ld - self.mean_log_difference())/self.std_log_difference()
    
    def _plot(self) -> None: 
        plt.plot(list(self.datamap.keys()), list(self.datamap.values()))
        plt.show()
    
    def normal_egarch_fit(self) -> ARCHModelResult: 
        archmodel = ZeroMean(self.log_difference, rescale=False)
        archmodel.volatility = EGARCH(1,0,1)
        archmodel.distribution = Normal()
        return archmodel.fit(disp=False, show_warning=False)
    
    def normal_garch_fit(self) -> ARCHModelResult: 
        archmodel = ZeroMean(self.log_difference, rescale=False)
        archmodel.volatility = GARCH(1,0,1)
        archmodel.distribution = Normal()
        return archmodel.fit(disp=False, show_warning=False)
    
    def ar_1lag_fit(self) -> ARCHModelResult: 
        archmodel = ARX(self.log_difference, lags=1, rescale=False)
        return archmodel.fit(disp=False, show_warning=False)

def get_dt_from_time_delta(delta:timedelta) -> float: 
    return delta.total_seconds()/(365*24*60*60)

@dataclass
class TimeSerie:
    datamap : dict[datetime, float]

    def __post_init__(self): 
        self.datamap = dict(sorted(self.datamap.items()))
        self.dates = list(self.datamap.keys())
        self.values = list(self.datamap.values())
    
    def _plot(self) -> None: 
        plt.plot(list(self.datamap.keys()), list(self.datamap.values()))
        plt.show()

def get_z_score_time_serie(data: TimeSerie) -> TimeSerie: 
    values = np.array(data.values)
    std, mean = np.std(values), np.mean(values)
    zscore = (values - mean)/std
    zscore = zscore.tolist()
    return TimeSerie(dict(zip(data.dates,zscore)))

def get_z_score_time_series(data: List[TimeSerie]) -> List[TimeSerie]: 
    return [get_z_score_time_serie(d) for d in data]

def get_difference_time_serie(data: TimeSerie) -> TimeSerie: 
    values = np.array(data.values)
    diffv = np.diff(values)
    dates = data.dates
    return TimeSerie(dict(zip(dates[1:len(dates)],diffv.tolist())))

def get_log_difference_time_serie(data: TimeSerie) -> TimeSerie: 
    values = np.array(data.values)
    diffv = np.diff(np.log(values))
    dates = data.dates
    return TimeSerie(dict(zip(dates[1:len(dates)],diffv.tolist())))

def get_square_log_difference_time_serie(data: TimeSerie) -> TimeSerie: 
    values = np.array(data.values)
    diffv = np.diff(np.log(values))**2
    dates = data.dates
    return TimeSerie(dict(zip(dates[1:len(dates)],diffv.tolist())))

def get_annualized_realised_volatility_time_serie(data: TimeSerie, dt: float) -> TimeSerie: 
    values = np.array(data.values)
    arv = np.sqrt((np.diff(np.log(values))**2)/dt)
    dates = data.dates
    return TimeSerie(dict(zip(dates[1:len(dates)],arv.tolist())))

def get_rolling_realised_vol_time_series(
        data: TimeSerie, 
        windows : List[timedelta]) -> List[TimeSerie]: 
    output = list()
    dates = list(data.datamap.keys())
    for w in windows:
        output_dict = dict()
        dt = get_dt_from_time_delta(w)
        for d in data.dates: 
            if d >= min(data.dates)+w:
                dstart = d - w
                data_filter = {k:data.datamap[k] 
                            for k in dates 
                            if k<=d and k >= dstart}
                values = np.sum(np.array(list(data_filter.values())))
                output_dict[d] = np.sqrt(values/dt).item()
            else: continue
        output.append(TimeSerie(output_dict))
    return output

def get_rolling_mean_time_series(
        data: TimeSerie, 
        windows : List[timedelta]) -> List[TimeSerie]: 
    output = list()
    dates = list(data.datamap.keys())
    for w in windows:
        output_dict = dict()
        dt = get_dt_from_time_delta(w)
        for d in data.dates: 
            if d >= min(data.dates)+w:
                dstart = d - w
                data_filter = {k:data.datamap[k] 
                            for k in dates 
                            if k<=d and k >= dstart}
                values = np.sum(np.array(list(data_filter.values())))
                output_dict[d] = np.mean(values).item()
            else: continue
        output.append(TimeSerie(output_dict))
    return output

def get_spread_time_series(data: List[TimeSerie]) -> List[TimeSerie]: 
    output = list()
    for i in range(0, len(data)-1):
        d1 = data[i]
        for u in range(i+1, len(data)): 
            d2 = data[u]
            output_dict = dict()
            for d in d1.dates: 
                if d in d2.dates: 
                    v =  d1.datamap[d] - d2.datamap[d]
                    output_dict[d] = v
                else: continue
            output.append(TimeSerie(output_dict))
    return output

def time_delta_shift_time_serie(data:TimeSerie, delta:timedelta) -> TimeSerie: 
    end_date, start_date = max(data.dates) + delta, min(data.dates) + delta 
    return TimeSerie({k:data.datamap[k] for k in data.dates 
                      if k >= start_date
                      and k <= end_date })

def filter_time_serie_from_dates(
        data: TimeSerie, 
        min_date:datetime, 
        max_date: datetime,
        inclusive_min : bool, 
        inclusive_max: bool) -> TimeSerie: 
    if inclusive_min and inclusive_max:
        d = {k: data.datamap[k] for k in data.dates
                    if k >= min_date
                    and k <= max_date} 
    elif not inclusive_min and inclusive_max: 
        d = {k: data.datamap[k] for k in data.dates
                    if k > min_date
                    and k <= max_date}  
    elif inclusive_min and not inclusive_max: 
        d = {k: data.datamap[k] for k in data.dates
                    if k >= min_date
                    and k < max_date}  
    else: 
        d = {k: data.datamap[k] for k in data.dates
                    if k > min_date
                    and k < max_date} 
    return TimeSerie(d) 

def filter_many_time_serie_from_dates(
        data: List[TimeSerie], 
        min_date:datetime, 
        max_date: datetime,
        inclusive_min : bool, 
        inclusive_max: bool) -> List[TimeSerie]: 
    return [filter_time_serie_from_dates(
        d, min_date, max_date, 
        inclusive_min, inclusive_max) for d in data]
