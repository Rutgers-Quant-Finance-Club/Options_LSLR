from gbm import GeometricBrownianMotion
import matplotlib.pyplot as plt

def main():
	gbm = GeometricBrownianMotion(
		S0=100,
		mu=0,
		sig=0.05,
		T=1,
		num_paths=10,
		time_steps=50
	)

	gbm.plot()
	plt.show()

if __name__ == "__main__":
	main()