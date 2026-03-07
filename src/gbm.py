import numpy as np


def geometric_brownian_motion(S0: float, mu: float, sigma: float, T: float, M: int, n: int):
	"""
	S0: initial stock price
	mu: drift coefficient
	sigma: volatility 
	T: time in years
	M: number of paths/simulations
	n: number of time steps

	dS = rSdt + σSdZ

	St = S0 * e^((mu - sig^2 / 2)t+sigW)
	"""
	# Time step
	dt = T / n

	# Wiener process M x n 
	Wt = np.random.normal(0, np.sqrt(dt), size=(M, n)).T

	# Create M x n matrix of incremental price multipliers 
	# [ 1.03 0.95 ... ]
	# [ 0.97 1.02 ... ]
	
	# Row 1 -> time = 0
	# Row 2 -> time = 1dt
	St = np.exp(
		(mu - 0.5 * sigma**2) * dt 
		+ sigma * Wt
	)

	# Add initial row of t(0) = 1 to matrix
	St = np.vstack([np.ones(M), St])

	# Convert incremental multipliers into total price from starting price S0
	St = S0 * St.cumprod(axis=0)
	
	return St