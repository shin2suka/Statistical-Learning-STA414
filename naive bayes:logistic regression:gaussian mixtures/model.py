from loadMNIST import *
from autograd.scipy.misc import logsumexp
import autograd as ag
import time
# np.random.seed(1)


COLORS = ["indianred", "palegoldenrod", "black", "gray"]


def get_images_by_label(images, labels, query_label):
		"""
		Helper function to return all images in the provided array which match the query label.
		"""
		assert images.shape[0] == labels.shape[0]
		matching_indices = labels == query_label
		return images[matching_indices]


class NaiveBayes:
	"""
	Q1, Naive Bayes model.
	"""
	def __init__(self, train_images, train_labels):
		self.train_images = train_images
		self.train_labels = train_labels

	def map_naive_bayes(self, plot=False):
		"""
		return a matrix 10 * 784 where each row represents a class.
		"""
		theta = np.zeros((10, 784))
		for c in range(10):
		    images = get_images_by_label(self.train_images, self.train_labels, c)
		    theta[c] = np.divide(np.sum(images, axis=0) + 1., images.shape[0] + 2.)
		if plot:
			save_images(theta, "theta_map.png")
		return theta

	def log_likelihood(self, X, y, theta):
		"""
		return a matrix N * 10 where each row represents log likelihood of a data point,
		and each column represents log lilihood of a class.
		"""
		ll = np.zeros((X.shape[0], 10))
		log_p_x = logsumexp(np.log(0.1) + np.dot(X, np.log(theta.T)) + np.dot((1. - X), np.log(1. - theta.T)), axis=1)
		for c in range(10):
			ll[:, c] = np.dot(X, np.log(theta[c])) + np.dot((1. - X), np.log(1. - theta[c])) + np.log(0.1) - log_p_x
		return ll

	def avg_log_likelihood(self, X, y, theta):
		ll = 0
		for c in range(10):
			X_c = get_images_by_label(X, y, c)
			log_p_x = logsumexp(np.log(0.1) + np.dot(X_c, np.log(theta.T)) + np.dot((1. - X_c), np.log(1. - theta.T)), axis=1)
			ll += np.sum(np.dot(X_c, np.log(theta[c])) + np.dot((1. - X_c), np.log(1. - theta[c])) + np.log(0.1) - log_p_x)
		return ll / X.shape[0]

	def predict(self, X, y, theta, train=False, test=False):
		ll = self.log_likelihood(X, y, theta)
		pred = np.argmax(ll, axis=1)
		avg_ll = self.avg_log_likelihood(X, y, theta)
		accuracy = np.mean(pred == y)
		name = "test" if test else "train"
		print("average log-likelihood of naive bayes model on the {} set: ".format(name) + str(avg_ll))
		print("accuracy of naive bayes model on the {} set: ".format(name) + str(accuracy))


class GenerativeNaiveBayes:
	"""
	Q2, Generating from a Naive Bayes Model
	"""
	def __init__(self, theta):
		self.theta = theta

	def sample_plot(self):
		"""
		randomly sample and plot 10 binary images from the marginal distribution, p(x|theta, pi)
		"""
		c = np.random.multinomial(10, [0.1]*10)
		images = np.zeros((10, 784))
		count = 0
		for i in range(10):
			for j in range((c[i])):
				images[count] = np.random.binomial(1, self.theta[i]).reshape((1, 784))
				count += 1
		save_images(images, "samples.png")

	def predict_half(self, X_top):
		"""
		plot the top half the image concatenated with the marginal distribution over each pixel in the bottom half.
		"""
		X_bot = np.zeros((X_top.shape[0], X_top.shape[1]))
		theta_top, theta_bot = self.theta[:, :392].T, self.theta[:, 392:].T
		for i in range(392):
			constant = np.dot(X_top, np.log(theta_top)) + np.dot(1 - X_top, np.log(1 - theta_top))
			X_bot[:, i] = logsumexp(np.add(constant, np.log(theta_bot[i])), axis=1) - logsumexp(constant, axis=1) 
		save_images(np.concatenate((X_top, np.exp(X_bot)), axis=1), "predict_half.png")


class LogisticRegression:
	"""
	Q3, Fitting a simple predictive model using gradient descent. 
	Our model will be multiclass logistic regression.
	"""
	def __init__(self, train_images, train_labels):
		self.train_images = train_images
		self.train_labels = train_labels
		self.W = np.zeros((10, 784))

	def softmax(self, X, W):
		"""
		return a N * 10 vector where each row is a data point
		and each column is the probability of that class.
		"""
		return np.log((np.exp(np.dot(X, W.T)).T / np.exp(logsumexp(np.dot(X, W.T), axis=1))).T)

	def grad_pred_ll(self, X, W, c):
		"""
		This function calculate the gradient of the predictive log-likelihood.
		return a 10 * 784 vector
		"""
		constant = np.exp(logsumexp(np.dot(X, W.T), axis=1))
		return np.sum(X - (X.T * np.divide(np.exp(np.dot(X, W[c])), constant)).T, axis=0)

	def gradient_ascent(self, lr=0.00001, iters=100):
		for _ in range(iters):
			prob = self.softmax(self.train_images, self.W)
			pred = np.argmax(prob, axis=1)
			accuracy = np.mean(pred == self.train_labels)
			print("training accuracy: {}, iterations: {}/{}".format(round(accuracy, 2), _, iters))
			for c in range (10) :
				X_c=get_images_by_label(self.train_images, self.train_labels, c)
				self.W[c] += lr * self.grad_pred_ll(X_c, self.W, c)

	def log_likelihood(self, X, y, W):
		ll = 0
		for c in range(10):
			X_c = get_images_by_label(X, y, c)
			ll += np.sum(np.dot(X_c, W[c]) - logsumexp(np.dot(X_c, W.T), axis=1))
		return ll / X.shape[0]

	def predict(self, X, y, train=False, test=False):
		if train:
			self.gradient_ascent()
			save_images(self.W, "weights.png")
		avg_ll = self.log_likelihood(X, y, self.W)
		pred = np.argmax(self.softmax(X, self.W), axis=1)
		accuracy = np.mean(pred == y)
		name = "test" if test else "train"
		print("average log-likelihood of softmax model on the {} set: ".format(name) + str(avg_ll))
		print("accuracy of softmax model on the {} set: ".format(name) + str(accuracy))


class EM:
	"""
	Q4, EM algorithm for K means and Gaussian mixtures.
	"""
	def __init__(self, initials, c1, c2):
		self.initials = initials
		self.data = np.concatenate((c1, c2), axis=0)
		self.N, self.D = self.data.shape			# Data is a N * D matrix, and here N=400, D=2

		# Initial values for K mean and GMM
		self.miu_hat = np.concatenate((self.initials['MIU1_HAT'], self.initials['MIU2_HAT']), axis=0).reshape((2,2))
		self.clusters = np.concatenate((np.zeros(int(self.N/2)), np.ones(int(self.N/2))), axis=0)
		self.costs_iter = [[], []]

	def plot_clusters(self, km=False, gmm=False):
		"""
		a scatter plot of the data points showing the true cluster assignment of each point. 
		Also plot a scatter plot of K mean or gaussian mixtures.
		"""
		f2 = plt.figure()
		ax2 = f2.add_subplot(111)
		for i in range(self.D):
			plt.scatter(self.data[self.clusters == i][:, 0], self.data[self.clusters == i][:, 1], c=COLORS[i])
			plt.scatter(self.initials['MIU'+str(i+1)][0], self.initials['MIU'+str(i+1)][1], marker='*', c=COLORS[2], s=150)
		plt.title("Scattar Plot of Data Points (Original)")
		if km or gmm:
			name = "K mean" if km else "Gaussian Mixtures"
			plt.scatter(self.miu_hat[:,0], self.miu_hat[:,1], marker='^', c=COLORS[3], s=100)
			plt.title("Scattar Plot of Data Points ({})".format(name))

	def misclassification_error(self):
		return (np.sum(self.clusters[:int(self.N/2)] == 1) + np.sum(self.clusters[int(self.N/2):] == 0)) / self.N 


class KMean(EM):
	def __init__(self, initials, c1, c2):
		super().__init__(initials, c1, c2)

	def cost(self):
		cost = 0
		for i in range(self.D):
			cost += np.sum(np.linalg.norm(self.data[self.clusters == i] - self.miu_hat[i], axis=1) ** 2)
		return cost

	def km_e_step(self):
		distances = np.zeros((self.N, 2))
		for i in range(self.D):
			distances[:,i] = np.linalg.norm(self.data - self.miu_hat[i], axis=1)
		self.clusters = np.argmin(distances, axis=1)

	def km_m_step(self):
		for i in range(self.D):
			self.miu_hat[i] = np.mean(self.data[self.clusters == i], axis=0)

	def train(self, max_iter=100):
		i = 1
		while i <= max_iter:
			self.km_e_step()
			self.km_m_step()
			self.costs_iter[0].append(self.cost())
			self.costs_iter[1].append(i)
			i += 1
		f3 = plt.figure()
		ax3 = f3.add_subplot(111)
		plt.plot(self.costs_iter[1], self.costs_iter[0])
		plt.title("K mean\n Cost vs The number of iterations")
		plt.xlabel("The number of iterations")
		plt.ylabel("Cost")
		self.plot_clusters(km=True)
		print("misclassification error for k mean: " + str(self.misclassification_error()))


class GaussianMixtures(EM):
	def __init__(self, initials, c1, c2):
		super().__init__(initials, c1, c2)
		# Initial values for Gaussian mixtures
		self.simga_hat = [np.eye(self.D)] * 2
		self.pi_hat = [0.5, 0.5]					# Mixing proportions
		self.R = np.zeros((self.N, 2))				# This is the posterior/responsibilities
		self.N_k = []								# The number of data in class K

	def normal_density(self, X, miu, sigma):
		"""
		This is a vectorized normal_densitym, where X is N*D, miu is 1*D, sigma is D*D
		Output is N*1, where each element is a pdf value.
		"""
		constant = 1 / np.sqrt((2 * np.pi) ** self.D * np.linalg.det(sigma))
		return constant * np.diag(np.exp(-0.5 * np.dot(np.dot((X - miu), np.linalg.inv(sigma)), (X - miu).T)))

	def log_likelihood(self):
		normal_sum = np.zeros(self.N)
		for i in range(self.D):
			normal_sum += self.pi_hat[i] * self.normal_density(self.data, self.miu_hat[i], self.simga_hat[i])
		return np.sum(np.log(normal_sum))

	def em_e_step(self):
		for i in range(self.D):
			self.R[:, i] = self.pi_hat[i] * self.normal_density(self.data, self.miu_hat[i].reshape((1,2)), self.simga_hat[i])
		# Normalize R
		self.R = (self.R.T / np.sum(self.R, axis=1)).T
		# assign datapoints to each gaussian 
		self.N_k = np.sum(self.R, axis = 0)
		
	def em_m_step(self):
		for i in range(self.D):
			self.miu_hat[i] = 1. / self.N_k[i] * np.sum(self.R[:, i] * self.data.T, axis=1).T
			diff = self.data - self.miu_hat[i]
			self.simga_hat[i] = 1. / self.N_k[i] * np.dot(np.multiply(diff.T,  self.R[:, i]), diff)
			self.pi_hat[i] = self.N_k[i] / self.N

	def train(self, max_iter=100):
		i = 1
		while i <= max_iter:
			self.em_e_step()
			self.em_m_step()
			self.costs_iter[0].append(self.log_likelihood())
			self.costs_iter[1].append(i)
			i += 1
		f4 = plt.figure()
		ax4 = f4.add_subplot(111)
		plt.plot(self.costs_iter[1], self.costs_iter[0])
		plt.title("Gaussian Mixtures\n log likelihood vs The number of iterations")
		plt.xlabel("The number of iterations")
		plt.ylabel("log likelihood")
		self.clusters = np.argmax(self.R, axis=1)
		self.plot_clusters(gmm=True)
		print("misclassification error for gmm: " + str(self.misclassification_error()))


if __name__ == '__main__':
	start = time.time()
	print("loading data...")
	N_data, train_images, train_labels, test_images, test_labels = load_mnist()
	train_labels = np.argmax(train_labels, axis=1)
	test_labels = np.argmax(test_labels, axis=1)

	print("trainning a Naive Bayes model...")
	nb_model = NaiveBayes(train_images, train_labels)
	theta_map = nb_model.map_naive_bayes(plot=True)
	nb_model.predict(train_images, train_labels, theta_map, train=True)
	nb_model.predict(test_images, test_labels, theta_map, test=True)

	print("training a generative Naive Bayes model...")
	gnb = GenerativeNaiveBayes(theta_map)
	gnb.sample_plot()
	gnb.predict_half(train_images[:20,:392])

	print("training a softmax model...")
	lr_model = LogisticRegression(train_images, train_labels)
	lr_model.predict(train_images, train_labels, train=True)
	lr_model.predict(test_images, test_labels, test=True)

	print("training K mean and GMM algorithms...")
	initials = {'Nk': 200,
				'MIU1': np.array([0.1, 0.1]),
				'MIU2': np.array([6., 0.1]),
				'COV': np.array([[10., 7.], [7., 10.]]),
				'MIU1_HAT': np.array([0., 0.]),
				'MIU2_HAT': np.array([1., 1.])
				}
	# Sampling data from a multivariate guassian distribution
	c1 = np.random.multivariate_normal(initials['MIU1'], initials['COV'], initials['Nk'])
	c2 = np.random.multivariate_normal(initials['MIU2'], initials['COV'], initials['Nk'])
	kmean = KMean(initials, c1, c2)
	kmean.plot_clusters()
	kmean.train()
	gmm = GaussianMixtures(initials, c1, c2)
	gmm.train()
	end = time.time()
	print("running time: {}s".format(round(end - start, 2)))
	plt.show()


