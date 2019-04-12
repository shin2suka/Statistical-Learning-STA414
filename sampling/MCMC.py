import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(414)


# numerical integration
def p_x_given_theta(x, theta):
	return norm.pdf(x, loc=theta, scale=4)

def f(x, theta):
	return np.sin(5 * (x - theta)) ** 2 / (25 * np.sin(x - theta) ** 2)

def p_g_given_x_theta(x, theta, g=1):
	return f(x, theta) ** g * (1 - f(x, theta)) ** (1 - g)

def numerical_integration(g, theta):
	x = np.linspace(-20,20,10000)
	prod = p_g_given_x_theta(x, theta, g) * p_x_given_theta(x, theta)
	area = np.sum(prod) * 40 / len(x)
	# f1 = plt.figure()
	# ax1 = f1.add_subplot(111)
	# plt.plot(x, prod)
	return area
	
# rejection sampling
def rejection_sampling(iteration=1000):
	samples = []
	i, j = 0, 0
	M = 1
	while i < iteration:
		u = np.random.uniform(0, 1)
		x = np.random.normal(0, 4)
		if u < p_g_given_x_theta(x, theta=0) * 0.8 / M:
			samples.append(x)
			i += 1
		j += 1
	print("fraction of accepted samples: " + str(round(i/j, 2)))
	return samples

# importance sampling
def importance_sampling(iteration=1000):
	i, summation = 0, 0
	while i < iteration:
		x = np.random.normal(0, 4)
		summation += p_g_given_x_theta(x, theta=0, g=0)
		i += 1
	return summation / i

# Q1(d)
def p_x_given_theta(x, theta):
	return 1 / (4 * np.sqrt(2 * np.pi)) * np.exp(-(x - theta) ** 2 / 32)

def cauchy(theta):
	return 1 / (10 * np.pi * (1 + (theta / 10) ** 2))

def p_x_g_theta(theta):
	"""
	the unnormalized density p(x = 1.7, g = 1, θ), as a function of θ
	"""
	return f(1.7, theta) * p_x_given_theta(1.7, theta) * cauchy(theta)

# Metropolis-Hastings sampling
def metropolis_hastings(iteration=10000):
	samples = []
	i, x = 0, 0
	while i < iteration:
		u = np.random.uniform(0, 1)
		x_new = np.random.normal(x, 100)
		func = lambda a, b : p_g_given_x_theta(1.7, a) * p_x_given_theta(1.7, a) * cauchy(a) * norm.pdf(b, loc=a, scale=100)
		fraction = func(x_new, x) / func(x, x_new)
		if u < fraction:
			x = x_new
		
		samples.append(x)
		i += 1
	return samples


if __name__ == '__main__':
	# Q1(a) estimate the fraction of photons that get absorbed on average
	print("estimate of p(g = 0|θ = 0):" + str(round(numerical_integration(g=0, theta=0), 2)))

	# Q1(b) plot rejection sampling
	x = rejection_sampling(iteration=10000)
	f2 = plt.figure()
	ax2 = f2.add_subplot(111)
	count, bins, ignored = plt.hist(x, 100, density=True)
	plt.title("Rejection Sampling")

	# Q1(c) estimatep(g = 0|θ = 0)
	print("the estimate (importance sampling): " + str(round(importance_sampling(), 2)))

	# Q1(d) plot unnormalized density p(x = 1.7, g = 1, θ)
	theta = np.linspace(-20,20,100) # 100 linearly spaced numbers
	y = p_x_g_theta(theta)
	f3 = plt.figure()
	ax3 = f3.add_subplot(111)
	plt.plot(theta, y)
	plt.title("Plot of Unnormalized Density: p(x = 1.7, g = 1, θ)")

	# Q1(e) plot Metropolis-Hastings sampling
	theta_samples = metropolis_hastings(iteration=100000)
	f4 = plt.figure()
	ax4 = f4.add_subplot(111)
	count, bins, ignored = plt.hist(theta_samples, 100, density=True)
	plt.title("Metropolis-Hastings Sampling")
	# plot the true unnormalized density
	f5 = plt.figure()
	ax5 = f5.add_subplot(111)
	theta = np.linspace(-20,20,100)
	y = p_g_given_x_theta(1.7, theta) * p_x_given_theta(1.7, theta) * cauchy(theta)
	plt.plot(theta, y)
	plt.title("The True Unnormalized Density")

	# Q1(f) estimate the posterior probability
	within = 0
	for theta in theta_samples:
		if -3 < theta < 3:
			within +=1
	estimate_posterior = within / len(theta_samples)
	print("estimate of the posterior probability: " + str(round(estimate_posterior, 2)))
	plt.show()







