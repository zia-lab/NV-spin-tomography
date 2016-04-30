import numpy as np
import sklearn as sk
from sklearn import svm, preprocessing, cross_validation, cluster, metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
import analysis
import NV_generator

# store an object in a file
def store_obj(obj, fname):
	joblib.dump(obj, fname + ".pkl")

# load an object from a file
def load_obj(fname):
	return joblib.load(fname + ".pkl")

# for a window in tau data around a dip, returns the height of the dip, not from 1 but from the local maxima on either side
def window_height(data, dip_ind, window):
	height_left = data[window[0]]
	height_right = data[window[1]]
	value = data[dip_ind]
	return max(height_left, height_right) - value

# for a window in tau data around a dip, returns the full width half max
def full_width_half_max(data, dip_ind, window):
	left = window[0]
	left_height = window_height(data, left, window)
	max_height = window_height(data, dip_ind, window)
	while left_height < .5 * max_height:
		left = left + 1
		left_height = window_height(data, left, window)
	right = window[1]
	right_height = window_height(data, right, window)
	while right_height < .5 * max_height:
		right = right - 1
		right_height = window_height(data, right, window)
	return right - left

# returns the features used in classification of which dips are likely to be usable in an analysis (which dips can be fit)
def dip_features(i, dip_inds, data, tau, windows):
	di = dip_inds[i]
	M = data[di]
	t = tau[di]
	height = window_height(data, di, windows[i])
	width = windows[i][1] - windows[i][0]
	fwhm = full_width_half_max(data, di, windows[i])
	return [M, t, width, height, fwhm]

# decides if a fitted A and B are close enough to a spin to be counted as a fit
def is_fit(fitted_A, fitted_B, spin, A_acc = .1, B_acc = .5):
	A_acc = analysis.mag * A_acc
	B_acc= analysis.mag * B_acc
	return ((spin[0] >= fitted_A - A_acc) and (spin[0] <= fitted_A + A_acc) and
		(spin[1] >= fitted_B - B_acc) and (spin[1] <= fitted_B + B_acc))

def create_diamonds(diamond_num_list, omega_larmor, num_spins = 450):
	verbose, plots = False, False
	num_subsets = 4
	error_fun = analysis.squared_error
	error_tol = .1/64
	N = 64
	N_vals = np.arange(0, 256, 2)
	tau = analysis.choose_tau_params(N)
	for diamond_num in diamond_num_list:
		print "diamond_num: ", diamond_num
		A, B, r, costheta, sintheta = NV_generator.generate_spins(num_spins)
		def data_func(N, tau, noise_level = .02):
			data = analysis.calc_M(A, B, N, omega_larmor, tau)
			noise = np.random.randn(len(data)) * noise_level
			return data + noise
		data = data_func(N, tau)
		dip_inds, windows = analysis.find_resonances(data, fit_dips_below = None)
		successful_fits, good_fits, phis_list, xs_list, scaled_errors = [], [], [], [], []
		print "number of dips: ", len(dip_inds)
		for dii in range(len(dip_inds)):
			print "dii: ", dii
			dip_ind = dip_inds[dii]
			res_tau = tau[dip_ind]
			N_data = data_func(N_vals, res_tau)
			successful_fit, good_fit = True, True
			try:
				phis, xs, scaled_error = analysis.repeated_spin_fit(N_vals, N_data, error_tol = error_tol, error_fun = error_fun,
					num_subsets = num_subsets, verbose = verbose, plots = plots)
				good_fit = scaled_error <= error_tol
			except analysis.FitError:
				successful_fit, good_fit = False, False
				phis, xs = None, None
				scaled_error = None
			successful_fits.append(successful_fit)
			good_fits.append(good_fit)
			phis_list.append(phis)
			xs_list.append(xs)
			scaled_errors.append(scaled_errors)
		print "create diamond_dict"
		diamond_dict = {"A" : A, "B" : B, "r" : r, "costheta" : costheta, "sintheta" : sintheta, "N" : N,
			"tau" : tau, "data" : data, "N_vals" : N_vals, "N_data" : N_data, "dip_inds" : dip_inds, "windows" : windows,
			"successful_fits" : successful_fits, "good_fits" : good_fits, "error_tol" : error_tol, "num_subsets" : num_subsets,
			"phis_list" : phis_list, "xs_list" : xs_list, "scaled_errors" : scaled_errors}
		print "store diamond_dict"
		store_obj(diamond_dict, "diamonds/diamond_" + str(diamond_num))

# create the training and testing vectors for the dip classifier
def dip_dataset(diamonds, training_percent = .75, pickle = False, suffix = ""):
	train_X = [] # feature vectors
	train_Y = [] # classifications
	test_X = []
	test_Y = []
	for diamond_ind in range(len(diamonds)):
		print "diamond_ind: ", diamond_ind
		diamond = diamonds[diamond_ind]
		dip_inds = diamond["dip_inds"]
		tau = diamond["tau"]
		windows = diamond["windows"]
		data = diamond["data"]
		successful_fits = diamond["successful_fits"]
		good_fits = diamond["good_fits"]
		print "number of dips: ", len(dip_inds)
		for i in range(len(dip_inds)):
			features = dip_features(i, dip_inds, data, tau, windows)
			if diamond_ind < training_percent * len(diamonds):
				train_X.append(features)
				train_Y.append(successful_fits[i])# and good_fits[i])
			else:
				test_X.append(features)
				test_Y.append(successful_fits[i])# and good_fits[i])
	if pickle:
		print "create dataset"
		dataset = {"train_X" : train_X, "train_Y" : train_Y, "test_X" : test_X, "test_Y" : test_Y}
		print "store dataset"
		store_obj(dataset, "datasets/dip_dataset" + suffix)
	return train_X, train_Y, test_X, test_Y

# train the dip classifier svm. Optionally save it to a file.
def train_svm(train_X, train_Y, test_X, test_Y, kernel = 'rbf', C=1.0, verbose = False, pickle = False, suffix = ""):
	scaler = sk.preprocessing.StandardScaler().fit(train_X)
	clf = sk.svm.SVC(kernel = kernel, C=C)# svm with rbf kernel (default)
	clf.fit(scaler.transform(train_X), train_Y)
	if verbose:
		print "training score: ", clf.score(scaler.transform(train_X), train_Y)
		print "testing score: ", clf.score(scaler.transform(test_X), test_Y)
	if pickle:
		store_obj(scaler, "classifiers/scaler_svm_" + kernel + suffix)
		store_obj(clf, "classifiers/clf_svm_" + kernel + suffix)
	return scaler, clf

def train_trees(train_X, train_Y, test_X, test_Y, verbose = False, pickle = False, suffix = ""):
	scaler = sk.preprocessing.StandardScaler().fit(train_X)
	clf = sk.ensemble.ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
	clf.fit(scaler.transform(train_X), train_Y)
	if verbose:
		print "training score: ", clf.score(scaler.transform(train_X), train_Y)
		print "testing score: ", clf.score(scaler.transform(test_X), test_Y)
	if pickle:
		store_obj(scaler, "classifiers/scaler_trees" + suffix)
		store_obj(clf, "classifiers/clf_trees" + suffix)
	return scaler, clf

def guess_dataset(diamonds, omega_larmor, min_dip_ind = 3220, training_percent = .75, pickle = False, suffix = ""):
	train_X = [] # feature vectors
	train_Y = [] # classifications
	test_X = []
	test_Y = []
	error_fun = analysis.squared_error
	for diamond_ind in range(len(diamonds)):
		print "diamond_ind: ", diamond_ind
		diamond = diamonds[diamond_ind]
		tau = diamond["tau"]
		data = diamond["data"]
		N = diamond["N"]
		dip_inds = diamond["dip_inds"]
		windows = diamond["windows"]
		spin_dict = {}
		for dii in range(len(dip_inds)):
			if diamond["good_fits"][dii] and dip_inds[dii] >= min_dip_ind:
				dip_ind = dip_inds[dii]
				res_tau = tau[dip_ind]
				for phi, x in zip(diamond["phis_list"][dii], diamond["xs_list"][dii]):
					for omega_tilde, cosphi in analysis.calc_omega_tilde(phi, x, res_tau, omega_larmor):
						A, B = analysis.calc_A_B(cosphi, res_tau, omega_larmor, omega_tilde)
						if analysis.valid_A_B(A, B):
							if (dip_ind, phi, x) in spin_dict:
								spin_dict[(dip_ind, phi, x)] += [(A, B, cosphi)]
							else:
								spin_dict[(dip_ind, phi, x)] = [(A, B, cosphi)]
		for k in spin_dict.keys():
			err = []
			for A, B, _ in spin_dict[k]:
				err.append(error_fun(analysis.calc_M_single(A, B, N, omega_larmor, tau), data))
			min_err_ind = np.argmin(err)
			best_A, best_B, best_cosphi = spin_dict[k][min_err_ind]
			best_err = err[min_err_ind]
			_, _, x = k
			features = [best_cosphi, x, best_A, best_B, best_err]
			tag = 0
			for spin in set(zip(diamond["A"], np.abs(diamond["B"]))):
				if is_fit(best_A, best_B, spin, A_acc = .5, B_acc = 2):
					tag = 1
					break
			if diamond_ind < training_percent * len(diamonds):
				train_X.append(features)
				train_Y.append(tag)
			else:
				test_X.append(features)
				test_Y.append(tag)
	if pickle:
		print "create dataset"
		dataset = {"train_X" : train_X, "train_Y" : train_Y, "test_X" : test_X, "test_Y" : test_Y}
		print "store dataset"
		store_obj(dataset, "datasets/guess_dataset" + suffix)
	return train_X, train_Y, test_X, test_Y

"""
if __name__ == "__main__":
	B_field = 0.0403555 # Teslas
	gam_c = 67.262 * 10 ** 6 # Gyromagnetic ratio for a single c13 nucleus in rad s-1 T-1
	ms = 1
	omega_larmor = -1 * ms * gam_c * B_field

	create_diamonds, create_dataset, create_dip_classifier, create_guess_dataset, create_guess_classifier = False, False, True, False, False
	suffix = ""

	num_diamonds = 20
	if create_diamonds:
		create_diamonds(num_diamonds, omega_larmor, num_spins = 450)
	if create_dataset:
		diamonds = []
		for diamond_num in range(num_diamonds):
			print "loading diamond ", diamond_num
			diamond = load_obj("diamonds/diamond_" + str(diamond_num))
			diamonds.append(diamond)
		print "diamonds loaded"
		dip_dataset(diamonds, pickle = True, suffix = suffix)
	if create_dip_classifier:
		dataset = load_obj("datasets/dip_dataset" + "_good")
		train_svm(dataset["train_X"], dataset["train_Y"], dataset["test_X"], dataset["test_Y"], verbose = True, pickle = True, suffix = "_dip_good")
	if create_guess_dataset:
		diamonds = []
		for diamond_num in range(num_diamonds):
			print "loading diamond ", diamond_num
			diamond = load_obj("diamonds/diamond_" + str(diamond_num))
			diamonds.append(diamond)
		print "diamonds loaded"
		guess_dataset(diamonds, omega_larmor, pickle = True, suffix = "_limited")
	if create_guess_classifier:
		dataset = load_obj("datasets/guess_dataset" + "_limited")
		print "train svm"
		train_svm(dataset["train_X"], dataset["train_Y"], dataset["test_X"], dataset["test_Y"], verbose = True, pickle = True, suffix = "guess_limited")
"""
"""
	for diamond_num in range(num_diamonds):
		print "loading diamond ", diamond_num
		diamond = load_obj("diamonds/diamond_" + str(diamond_num))
		print "total fits: ", len(diamond["successful_fits"])
		print "successful fits: ", sum(diamond["successful_fits"])
		print "good fits: ", sum(diamond["good_fits"])
"""
"""
# load diamond files that were saved by noisy_dataset.py
def load_diamonds(num_diamonds, diamond_dir):
	diamonds = []
	for i in range(num_diamonds):
		diamond = np.load(diamond_dir + "/diamond_" + str(i) + ".npz")
		#diamonds.append(dict(diamond))
		diamonds.append(diamond)
		#diamond.close()
	return diamonds

# load a classification dataset with training and testing Xs (features) and Ys (labels, 0 or 1)
def load_dataset(fname):
	d = np.load(fname + ".npz")
	return d["train_X"], d["train_Y"], d["test_X"], d["test_Y"]
"""
