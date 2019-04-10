import scipy.io as sio
import random
import numpy as np
import matplotlib.pyplot as plt


dataset = sio.loadmat('dataset.mat')
data_train_X = dataset['data_train_X']
data_train_y = dataset['data_train_y'][0]
data_test_X = dataset['data_test_X']
data_test_y = dataset['data_test_y'][0]

data_train = (data_train_X, data_train_y)
data_test = (data_test_X, data_test_y)

def shuffle_data(data):
	"""
	we are considering a uniformly random permutation of the training data.

	"""
	n = len(data[0])
	inc = np.arange(n)
	idx = np.random.permutation(inc)

	return data[0][idx], data[1][idx]


def split_data(data, num_folds, fold):
	"""
	num_folds: number of partitions
	fold: selected partition

	"""
	idx = int(data[0].shape[0] / num_folds)
	data_fold = data[0][(fold - 1) * idx:fold * idx], data[1][(fold - 1) * idx:fold * idx]
	data_rest = np.append(data[0][:(fold - 1) * idx], data[0][fold * idx:], axis=0), np.append(data[1][:(fold - 1) * idx], data[1][fold * idx:], axis=0)
	return data_fold, data_rest


def train_model(data, lambd):
	"""
	returns the coefficients of ridge regression with penalty level λ

	"""
	X = data[0]
	Y = data[1]
	beta_map = np.linalg.solve(np.dot(np.transpose(X), X) + lambd * np.identity(X.shape[1]), np.dot(X.T, Y))
	return beta_map


def predict(data, model):
	"""
	returns the predictions based on data and model.

	"""
	predictions = np.dot(data[0], model)
	return predictions


def loss(data, model):
	"""
	returns the average squared error loss based on model.

	"""
	X = data[0]
	y = data[1]
	n = len(y)
	error = np.dot((y - np.dot(X, model)).T, y - np.dot(X, model)) / n
	return error


def cross_validation(data, num_folds, lambd_seq):
	"""
	returns the cross validation error across all λ’s.

	"""
	cv_error = np.zeros(50)
	data_shf = shuffle_data(data)
	for i in range(len(lambd_seq)):
		lambd = lambd_seq[i]
		cv_loss_lmd = 0
		for fold in range(num_folds):
			val_cv, train_cv = split_data(data_shf, num_folds, fold + 1)
			model = train_model(train_cv, lambd)
			cv_loss_lmd += loss(val_cv, model)
		cv_error[i] = cv_loss_lmd / num_folds
	return cv_error


if __name__ == '__main__':
	a = 0.02
	b = 1.5
	lambd_seq = [a + i * (b - a) / 50 for i in range(50)]
	training_errors = []
	test_errors = []

	for lambd in lambd_seq:
		model = train_model(data_train, lambd)
		predictions = predict(data_train, model)
		training_error = loss(data_train, model)
		test_error = loss(data_test, model)
		training_errors.append(training_error)
		test_errors.append(test_error)

	cv_error_5 = cross_validation(data_train, 5, lambd_seq)
	cv_error_10 = cross_validation(data_train, 10, lambd_seq)
	plt.plot(lambd_seq, training_errors)
	plt.plot(lambd_seq, test_errors)
	plt.plot(lambd_seq, cv_error_5)
	plt.plot(lambd_seq, cv_error_10)
	plt.xlabel('Lambda')
	plt.ylabel('Error')
	plt.legend(['training error', 'test error', 'CV5-fold', 'CV10-fold'], loc='upper right')
	plt.show()


