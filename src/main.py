from gbm import geometric_brownian_motion
import matplotlib.pyplot as plt

def main():
	gbm = geometric_brownian_motion(
		S0=100,
		mu=0,
		sigma=0.05,
		T=1,
		num_paths=10,
		time_steps=50
	)

	gbm.plot()
	plt.show()

if __name__ == "__main__":
	main()