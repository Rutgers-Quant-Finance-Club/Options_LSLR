from dataclasses import dataclass
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class geometric_brownian_motion:
	"""	
	Simulate geometric brownian motion paths

	S0: initial stock price
	mu: drift coefficient
	sigma: volatility 
	T: time in years
	M: number of paths/simulations
	n: number of time steps
	"""
	S0: float
	mu: float
	sigma: float
	T: float
	num_paths: int
	time_steps: int

	@cached_property
	def St(self) -> np.ndarray:
		"""
		dS = rSdt + σSdZ

		St = S0 * e^((mu - sig^2 / 2)t+sigW)
		"""
		# Time step
		dt = self.T / self.time_steps

		# Wiener process M x n 
		Wt = np.random.normal(0, np.sqrt(dt), size=(self.num_paths, self.time_steps)).T

		# Create M x n matrix of incremental price multipliers 
		# [ 1.03 0.95 ... ]
		# [ 0.97 1.02 ... ]
		#
		# Row 1 -> time = 0
		# Row 2 -> time = 1dt
		St = np.exp(
			(self.mu - 0.5 * self.sigma**2) * dt 
			+ self.sigma * Wt
		)

		# Add initial row of t(0) = 1 to matrix
		St = np.vstack([np.ones(self.num_paths), St])

		# Convert incremental multipliers into total price from starting price S0
		St = self.S0 * St.cumprod(axis=0)
		
		return St

	def plot(self):
		""""
		Plot the simulated paths of the geometric brownian motion
		"""
		t = np.linspace(0, self.T, self.time_steps + 1) 	# [ 0 1dt 2dt ... T ]

		time = np.tile(t, (self.St.shape[1], 1)).T        	# [ 0 	0 	...  0  ]
															# [ 1dt 1dt ... 1dt ]
															# [ 2dt 2dt ... 2dt ]

		plt.plot(time, self.St)
		plt.xlabel("Time (years)")
		plt.ylabel("Stock Price")
		plt.title("Geometric Brownian Motion Simulation")