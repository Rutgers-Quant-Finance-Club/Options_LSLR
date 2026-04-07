from gbm import GeometricBrownianMotion
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class LSM_Engine:
	gbm: GeometricBrownianMotion
	K: float
	r: float

		
	def continuation_value(self, t: int):
		"""
		Implement the cross-sectional regression using a constant and the first three Laguerre
		polynomials to estimate the continuation value.
		"""
		
		# In the money stock prices at time t
		S_itm = self.itm(t)

		# Normalized stock prices
		X = S_itm / self.K

		# Laguerre polynomials
		L0 = np.exp(-X/2)
		L1 = np.exp(-X/2) * (1 - X)
		L2 = np.exp(-X/2) * (1 - 2*X + X**2/2)



	def itm(self, t: int) -> np.ndarray:
		"""
		Returns 1D array of stock prices for in-the-money paths at time step t 

		For a put, in the money means stock price is below the strike price (S < K)

		t: time step index (row in gbm.St)
		"""
		S_t = self.gbm.St[t]

		mask = S_t < self.K

		return S_t[mask]
	
