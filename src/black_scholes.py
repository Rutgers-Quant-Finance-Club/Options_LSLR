from math import exp, log, sqrt
from scipy.stats import norm


def black_scholes_put(S: float, K: float, T: float, r: float, b: float, sig: float) -> float:
	"""
	Get the price of a european put option using black-scholes model

	S: Stock price
	K: Strike price
	T: Time to maturity
	r: Risk-free rate
	b: Cost of carry (b = r for non-dividend paying stocks)
	sig: Volatility
	"""
	# Conditional stock receipt of a european option
	d1 = (log(S/K) + (b + sig**2 / 2) * T) / (sig * sqrt(T))

	# Probability of exercise of a european option 
	d2 = d1  - (sig * sqrt(T))

	# Put price	
	return K * exp(-r * T) * norm.cdf(-d2) - S * exp((b - r) * T) * norm.cdf(-d1)

