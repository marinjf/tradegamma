import numpy as np 
import scipy.optimize
import scipy.interpolate
import scipy.stats
from typing import List

class SSVI: 
    def __init__(self, 
                 rho:float, 
                 _nu: float, 
                 _gamma: float, 
                 atmtvarmap : dict[float, float]) -> None: 
        self.rho, self.nu, self._gamma = rho, _nu, _gamma
        self.atmtvarmap = dict(sorted(atmtvarmap.items()))
    
    def atm_total_variance(self, t:np.array) -> np.array:
        t_vec = np.array(list(self.atmtvarmap.keys()))
        v = np.array(list(self.atmtvarmap.values()))
        atminterp = scipy.interpolate.interp1d(t_vec, v)
        max_t = max(list(self.atmtvarmap.keys()))
        return atminterp(np.minimum(t, max_t))

    def parametrization(self, t:np.array) -> np.array: 
        atmtvar = self.atm_total_variance(t)
        return self.nu*(atmtvar**(-self._gamma)) 
    
    def total_variance(self, t: np.array, k: np.array) -> np.array:
        f = self.parametrization(t)
        atmtvar = self.atm_total_variance(t)
        term1 = self.rho*f*k
        term2 = np.sqrt((f*k+self.rho)**2 + (1-self.rho**2))
        return .5*atmtvar*(1 + term1 + term2)
    
    def implied_variance(self, k: np.array, t:np.array) -> np.array:
        return self.total_variance(t=t,k=k)/t
    
    def implied_volatility(self, k: np.array, t: np.array) -> np.array:
        return np.sqrt(self.implied_variance(k=k,t=t))
    
    def derivative(self,t: np.array) -> np.array:
        g = self._gamma
        return (1-g)*self.parametrization(t=t)
    
    def parameters_check(self) -> int: 
        cond1g = self._gamma>0
        cond2g = self._gamma<1
        cond_gamma = (cond1g and cond2g)
        cond_nu = self.nu > 0
        cond_model = (cond_gamma and cond_nu)
        cond_rho = abs(self.rho) < 1
        if cond_rho and cond_model: return 0
        else: return 1
    
    def butterfly_check(self) -> int:
        t = np.array(list(self.atmtvarmap.values()))
        atmtvar = self.atm_total_variance(t) 
        f = atmtvar*self.parametrization(t)
        f2 = atmtvar*self.parametrization(t)**2
        cond1 = np.all(f*(1+np.abs(self.rho))<=4)
        cond2 = np.all(f2*(1+np.abs(self.rho))<=4)
        if cond1 and cond2: return 0 
        else: return 1

    def calendar_spread_check(self) -> int: 
        t = np.array(list(self.atmtvarmap.keys()))
        atmtvar = np.array(list(self.atmtvarmap.values()))
        is_increasing = np.all(np.diff(atmtvar) >= 0)
        deriv = self.derivative(t)
        f = self.parametrization(t)
        value = f*(1+np.sqrt(1-self.rho**2))/(self.rho**2)
        cond1 = np.all(deriv>=0) 
        cond2 = np.all(deriv<=value)
        cond = (cond1 and cond2)
        if is_increasing and cond: return 0
        else: return 1

    

class CalibrateSSVI: 
    def __init__(self, 
                 volatilities:np.array, 
                 logmoneyness:np.array, 
                 t:np.array, 
                 atmtvarmap : dict[float, float]) -> None:  
        self.sigma = volatilities
        self.k = logmoneyness
        self.t = t 
        self.atmtvarmap = atmtvarmap
    
    def target(self) -> np.array: 
        return self.sigma
    
    def initial_guess(self) -> List[float]: 
        return [0, 0.5, 1] 
    
    @staticmethod
    def penality(ssvi:SSVI) -> int: 
        cs = ssvi.calendar_spread_check()
        bf = ssvi.butterfly_check()
        pm = ssvi.parameters_check()
        return (cs+bf+pm)*1000
    
    def loss(self, params: List[float]) -> float: 
        rho, _nu, _gamma = params[0], params[2], params[1]
        ssvi = SSVI(rho,_nu,_gamma,self.atmtvarmap)
        if ssvi.parameters_check()==1: return 1e9
        #pen = self.penality(ssvi)
        model = ssvi.implied_volatility(k=self.k, t=self.t)
        market = self.sigma
        return np.sum(((market-model)**2)/model)
    
    def calibrate(self) -> SSVI: 
        fit = scipy.optimize.minimize(
                fun = self.loss, 
                x0 = self.initial_guess(), 
                method = 'Nelder-Mead') 
        x = fit.x 
        rho, _nu, _gamma = x[0], x[2], x[1]
        return SSVI(rho,_nu,_gamma, self.atmtvarmap)
        

    


