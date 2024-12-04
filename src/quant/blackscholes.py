import numpy as np 
from scipy.stats import norm

class BlackScholes: 
    def __init__(
            self,
            S : np.array, 
            K : np.array, 
            sigma : np.array, 
            t : np.array, 
            q : np.array, 
            r : np.array, 
            call_or_put : np.array, 
            future : np.array): 
        self.S, self.K, self.sigma = S, K, sigma 
        self.t, self.q, self.r = t, q, r
        self.p, self.future = call_or_put, future
        self.mu = self.future*(self.r - self.q)
        self.F = self.S*np.exp(self.mu*self.t)
        self.d1 = self.get_d1()
        self.d2 = self.get_d2()
        self.Nd1 = norm.cdf(self.p*self.d1)
        self.Nd2 = norm.cdf(self.p*self.d2)
        self.nd1 = norm.pdf(self.d1)
        self.nd2 = norm.pdf(self.d2)

    def get_d1(self):
        F, K, sigma, t = self.F,self.K,self.sigma,self.t
        return(np.log(F/K)+t*.5*(sigma**2))/(sigma*np.sqrt(t))

    def get_d2(self):
        return self.d1-self.sigma*np.sqrt(self.t)
    
    def price(self) -> np.array: 
        F, K, r, t = self.F,self.K,self.r,self.t
        return np.exp(-r*t)*self.p*(F*self.Nd1-K*self.Nd2) 
    
    def delta(self) -> np.array: 
        t, mu, r = self.t, self.mu, self.r
        return self.p*np.exp((mu-r)*t)*self.Nd1
    
    def vega(self) -> np.array: 
        F, t, r = self.F, self.t, self.r
        return F*np.exp(-r*t)*self.nd1*np.sqrt(t)     
    
    def vanna(self) -> np.array: 
        t, mu, sigma, r = self.t, self.mu, self.sigma, self.r
        return -np.exp((mu-r)*t)*self.nd1*self.d2/sigma
    
    def gamma(self) -> np.array: 
        F, t, mu, sigma, r = self.F, self.t, self.mu, self.sigma, self.r
        return np.exp((mu-r)*t)*self.nd1/(np.exp(-mu*t)*F*sigma*np.sqrt(t))
    
    def theta(self) -> np.array: 
        F,t,mu,sigma,r,K = self.F, self.t, self.mu, self.sigma, self.r, self.K
        term1 = -F*np.exp(-r*t)*self.nd1*sigma/(2*np.sqrt(t))
        term2 = -self.p*r*K*np.exp(-r*t)*self.Nd2
        term3 = self.p*(r-mu)*F*np.exp(-r*t)*self.Nd1
        return term1 + term2 + term3 
    
    def rho(self) -> np.array: 
        t,r,K = self.t, self.r, self.K
        cond = self.inputdata.future
        inv_cond = np.logical_not(cond)
        rho = self.p*K*t*self.Nd2*np.exp(-r*t)
        return cond*rho+inv_cond*(-t*np.exp(-r*t)*self.price())
    
    def epsilon(self) -> np.array: 
        t,r,F = self.t, self.r, self.F
        return self.inputdata.future*(-self.p*F*t*self.Nd1*np.exp(-r*t))
    
    def charm(self) -> np.array: 
        t,mu,sigma,r = self.t, self.mu, self.sigma, self.r
        term1 = (mu-r)*np.exp((mu-r)*t)*self.Nd1
        term2 = (2*mu*t - self.d2*sigma*np.sqrt(t))/(2*t*sigma*np.sqrt(t))
        factor = np.exp((mu-r)*t)*self.nd1
        return self.p*term1 - factor*term2
    
    def veta(self) -> np.array: 
        F,t,mu,sigma,r = self.F, self.t, self.mu, self.sigma, self.r
        factor = -F*np.exp(-r*t)*self.nd1*np.sqrt(t)
        term1 = (r-mu)+mu*self.d1/(sigma*np.sqrt(t))
        term2 = (1+self.d1*self.d2)/(2*t)
        return factor*(term1-term2)
    
    def volga(self) -> np.array: 
        return self.vega()*self.d1*self.d2/self.sigma
    
    def speed(self) -> np.array: 
        F,t,mu,sigma = self.F, self.t, self.mu, self.sigma
        num = -self.gamma()*(self.d1/(sigma*np.sqrt(t))+1)
        den = np.exp(-mu*t)*F
        return num/den
    
    def zomma(self) -> np.array: 
        return self.gamma()*(self.d1*self.d2-1)/self.sigma
    
    def ultima(self) -> np.array:
        d1, d2, sigma = self.d1, self.d2, self.sigma
        d12 = d1*d2
        return -self.vega()*(d12*(1-d12) + d1**2 + d2**2)/sigma**2
    
    def color(self) -> np.array:
        t,mu,sigma,r = self.t, self.mu, self.sigma, self.r
        term1 = 2*(r-mu)+1
        term2 = self.d1*(2*mu*t - self.d2*sigma*np.sqrt(t))/sigma*np.sqrt(t)
        factor = self.gamma()/(2*t)
        return factor*(term1 + term2)
    
    def dual_delta(self) -> np.array: 
        return -self.p*np.exp(-self.r*self.t)*self.Nd2
    
    def dual_gamma(self) -> np.array: 
        den = self.K*self.sigma*np.sqrt(self.t)
        return np.exp(-self.r*self.t)*self.nd2/den