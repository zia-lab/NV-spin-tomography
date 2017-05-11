# NV-spin-tomography Michael Scheer mgscheer@gmail.com

# Creating datasets and training classifiers, storing objects on disk

import numpy as np
import sklearn as sk
from sklearn import svm, preprocessing, cross_validation, cluster, metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
import analysis
import NV_generator
import pickle

# store an object in a file
def store_obj(obj, fname):
	joblib.dump(obj, fname + ".pkl")

def store_obj_2(obj, fname):
	pickle.dump(obj, open(fname + ".pkl",'wb'))

# load an object from a file
def load_obj(fname):
	return joblib.load(fname + ".pkl")

def load_obj_2(fname):
	return pickle.load(open(fname + ".pkl",'rb'))

# decides if a fitted A and B are close enough to a spin to be counted as a fit.
# the B tends to be harder to fit so the accuracy demanded is less.
def is_fit(fitted_A, fitted_B, spin, A_acc = .1, B_acc = .5):
	A_acc = analysis.mag * A_acc
	B_acc= analysis.mag * B_acc
	return ((spin[0] >= fitted_A - A_acc) and (spin[0] <= fitted_A + A_acc) and
		(spin[1] >= fitted_B - B_acc) and (spin[1] <= fitted_B + B_acc))

# creates a dataset of diamonds with N measurements already analyzed, saves it to disk
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
			if dii%50==0:
				print "dii: ", dii
			dip_ind = dip_inds[dii]
			res_tau = tau[dip_ind]
			N_data = data_func(N_vals, res_tau)
			successful_fit, good_fit = True, True # did the fit succeed without throwing an error? was the error in the fit small enough?
			try:
				phis, xs, scaled_error = analysis.repeated_spin_fit(N_vals, N_data, error_tol = error_tol, error_fun = error_fun, num_subsets = num_subsets, verbose = verbose, plots = plots)
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
		store_obj_2(diamond_dict, "diamonds/diamond_" + str(diamond_num))

# train and optionally save a support vector machine classifier
# scaler preprocesses the data (scales it) and clf is the classifier
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

# train and optionally save an extra-trees classifier
# scaler preprocesses the data (scales it) and clf is the classifier
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

# creates and optionally saves the dataset used to train the SVM which classifies guesses.
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
		dip_inds = diamond["dip_inds"] # the indices for the dips
		windows = diamond["windows"] # where was this used?
		spin_dict = {} # The keys to the dictionary will be a tuple (dip_ind, phi, x)
		for dii in range(len(dip_inds)):
			if diamond["good_fits"][dii] and dip_inds[dii] >= min_dip_ind:
				dip_ind = dip_inds[dii]
				res_tau = tau[dip_ind]
				for phi, x in zip(diamond["phis_list"][dii], diamond["xs_list"][dii]):
					for omega_tilde, cosphi in analysis.calc_omega_tilde(phi, x, res_tau, omega_larmor):
						A, B = analysis.calc_A_B(cosphi, res_tau, omega_larmor, omega_tilde)
						if analysis.valid_A_B(A, B): # this checks for the upper bound imposed on (A,B)
							if (dip_ind, phi, x) in spin_dict: # if (dip_ind,phi,x) is already a key in the dictionary then add another value to the list that corresponds to it
								spin_dict[(dip_ind, phi, x)] += [(A, B, cosphi)]
							else: # if it is not, then simply initialize the list with the corresponding tuple
								spin_dict[(dip_ind, phi, x)] = [(A, B, cosphi)]
		for k in spin_dict.keys(): #now iterate through they dictionary's keys
			err = []
			for A, B, _ in spin_dict[k]:
				err.append(error_fun(analysis.calc_M_single(A, B, N, omega_larmor, tau), data)) # calculate the errors from assuming data only comes from a single spin
			min_err_ind = np.argmin(err)
			best_A, best_B, best_cosphi = spin_dict[k][min_err_ind]
			best_err = err[min_err_ind]
			_, _, x = k 
			features = [best_cosphi, x, best_A, best_B, best_err] # this is the feature vector
			tag = 0
			for spin in set(zip(diamond["A"], np.abs(diamond["B"]))): #iterate through the true spins in the diamond
				if is_fit(best_A, best_B, spin, A_acc = .5, B_acc = 2): # arbitrary thresholds over here
					tag = 1 # best_A and best_B have been found to correspond to a one of the true spins
					break
			if diamond_ind < training_percent * len(diamonds): # first diamonds for the training set
				train_X.append(features)
				train_Y.append(tag)
			else: # the other ones for the test set
				test_X.append(features)
				test_Y.append(tag)
	if pickle:
		print "create dataset"
		dataset = {"train_X" : train_X, "train_Y" : train_Y, "test_X" : test_X, "test_Y" : test_Y}
		print "store dataset"
		store_obj(dataset, "datasets/guess_dataset" + suffix)
	return train_X, train_Y, test_X, test_Y
