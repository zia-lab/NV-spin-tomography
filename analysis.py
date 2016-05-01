import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize as opt
from scipy import special, fftpack, signal
import operator, heapq, itertools, warnings
import sklearn as sk
import learning

# scale of omega, used for printing paramaters nicely
mag = 2 * np.pi * 1e3

# a special error class that is thrown in this code when a fit does not work
class FitError(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)

# create a plot in the style originally sent us by Tim and Julia.
def initialize_data_plot(figsize = (10,5), xlims = None, ylims = [-1.05,1.05]):
	fig,ax = plt.subplots(figsize=figsize)
	ax.set_ylim(ylims)
	if xlims is not None:
		ax.set_xlim(xlims)
	return fig, ax

# calculates omega_tilde
#def get_omega_tilde(A, B, omega_larmor):
#    return np.sqrt((A + omega_larmor) ** 2 + B ** 2)

# calculates the M function for a single spin A, B
def calc_M_single(A,B,N,omega_larmor,tau):
	tau = np.array(tau)
	omega_tilde = np.sqrt((A + omega_larmor) ** 2 + B ** 2)
	mx = B/omega_tilde
	mz = (A+omega_larmor)/omega_tilde
	alpha = omega_tilde * tau
	beta = omega_larmor * tau
	cos_phi = np.cos(alpha) * np.cos(beta)- mz * np.sin(alpha) * np.sin(beta)
	vec_num = (mx ** 2) * (1-np.cos(alpha)) * (1-np.cos(beta))
	vec_denom = 1.0 + cos_phi
	for i in np.where(vec_denom == 0.0)[0]:
		vec_denom[i] = .0001
	vec_term = vec_num/vec_denom
	angle_term = np.sin(N * np.arccos(cos_phi) / 2.0) ** 2
	return 1 - (vec_term * angle_term)

# calculates the M function corresponding to many spins given by
# A_list, B_list
# equivalent to np.array([calc_M_single(A, B, N, omega_larmor, tau) for A, B in zip(A_list, B_list)]).prod()
def calc_M(A_list, B_list, N, omega_larmor, tau):
	return reduce(lambda accum, next: accum * calc_M_single(next[0], next[1], N, omega_larmor, tau), zip(A_list, B_list), 1.0)

# calculates what A and B must be given the value of cos(phi) as well as omega_tilde
def calc_A_B(cosphi, res_tau, omega_larmor, omega_tilde):
	alpha = omega_tilde * res_tau
	beta = omega_larmor * res_tau
	mz = (np.cos(alpha) * np.cos(beta) - cosphi)/(np.sin(alpha) * np.sin(beta))
	A = mz * omega_tilde - omega_larmor
	B = np.sqrt((1 - mz ** 2)) * omega_tilde
	return A, np.abs(B)

# firm bounds on omega_tilde.
# Julia said they will throw out any diamond with a spin with coupling greater than 150 kHz.
# this implies these bounds on omega.
def omega_bounds(omega_larmor, coupling_bound = 150 * mag):
	upper_omega_bound = omega_larmor + coupling_bound
	lower_omega_bound = omega_larmor - coupling_bound
	return lower_omega_bound, upper_omega_bound

# checks if the A, B pair violates the 150 kHz bound or not
def valid_A_B(A, B, coupling_bound = 150 * mag):
	return A ** 2 + B ** 2 <= coupling_bound ** 2

# for a real sequence y indexed by a sequence x,
# writes y = a0 + a1 * cos(omega_0 * x) + b1 * sin(omega_0 * x) + ...
# with omega_0 = 2 * np.pi/len(x)
# as a "fourier series" (though for discrete y not continuous) and finds (very approximately)
# the a and b coefficients.
# coefs.real = a0, a1, a2, ...
# coefs.imag = 0, -b1, -b2, ... (notice the negative signs)
# see http://www.feynmanlectures.caltech.edu/I_50.html for fourier series and
# http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.fftpack.fft.html for the formula used for FFT.
# notice the 2/N difference between the two formulas for all but the 0 frequency term.
def sinusoid_coefs_fft(x, y):
	ft = fftpack.fft(y)
	freqs = fftpack.fftfreq(len(x), d = (x[1]-x[0])/(2*np.pi))
	# get rid of negative frequencies
	ft, freqs = map(np.array, zip(*filter(lambda p: p[1] >= 0, zip(ft, freqs))))
	coefs = 2.0 * np.array(ft)/len(x)
	coefs[0] = .5 * coefs[0]
	return coefs, freqs

# given the coefs and freqs from sinusoid_coefs_fft, finds the most likely frequencies
# present in the signal. It looks for peaks in both the real and imaginary part of the coefficients.
# it returns a list of frequencies in decreasing order of likelihood, based on the absolute value of the
# fourier spectrum (coefs) at that point.
def find_peaks_from_fourier(coefs, freqs):
	extremal_inds = signal.argrelextrema(coefs.real, np.greater)[0]
	extremal_inds = np.r_[extremal_inds, signal.argrelextrema(coefs.real, np.less)[0]]
	extremal_inds = np.r_[extremal_inds, signal.argrelextrema(coefs.imag, np.greater)[0]]
	extremal_inds = np.r_[extremal_inds, signal.argrelextrema(coefs.imag, np.less)[0]]
	extremal_inds = np.unique(extremal_inds)
	freq_inds = sorted(extremal_inds, key=lambda ind: -np.abs(coefs[ind]))
	return freqs[freq_inds]

# takes a list of frequencies that are peaks in a fourier spectrum, either of the real
# or imaginary part as well as a number of spins to look for. It then approximates the frequencies
#of the underlying product cosine functions that were in the signal.
def gen_freqs_from_fourier(freqs, num_spins):
	# sort them in decreasing order
	freqs = sorted(freqs, key=lambda x:-x)
	if num_spins == 1: # requires 1 freq
		return [freqs[0]]
	elif num_spins == 2: # requires 2 freqs
		f0, f1 = freqs[:2]
		return [.5 * (f0 + f1), .5 * (f0 - f1)]
	elif num_spins == 3: # requires 3 freqs
		f0, f1, f2 = freqs[:3]
		return [.5 * (f1 + f2), .5 * (f0 - f2), .5 * (f0 - f1)]
	elif num_spins == 4: # requires 5 freqs
		phi01sum, phi2, phi3 = gen_freqs_from_fourier(freqs[:4], 3)
		f3, f4 = freqs[3:5]        
		phi0 = .5 * (f3 + f4)
		phi1 = phi01sum - phi0
		return [phi0, phi1, phi2, phi3]
	else:
		return gen_freqs_from_fourier(freqs, 4) # change this if you figure out how to do 5

# the form of the signal as a function of N for a fixed tau
def spin_fit_fun_single(N, phi, x):
	return 1 - x * (np.sin(N * phi/2.0) ** 2)

# the form of the signal as a function of N for a fixed tau with several spins
# there can be any number of spins passed, in the form [phi0, phi1, ..., phin, x0, x1, ..., xn]
# where the function for a single spin is 1 - x * (np.sin(N * phi/2.0) ** 2)
# equivalent to np.array([single_spin_fit_fun(N, phi, x) for phi, x in zip(phis, xs)]).prod()
def spin_fit_fun(N, *args):
	num_spins = len(args)/2
	phis, xs = args[:num_spins], args[num_spins:]
	return reduce(lambda accum, next: accum * spin_fit_fun_single(N, next[0], next[1]), zip(phis, xs), 1.0)

# for two arrays l1 and l2 returns the squared error between them
def squared_error(l1, l2):
	return sum((l1 - l2) ** 2)

def spin_fit_guess(N_vals, N_data, num_spins, param_guess, error_fun = squared_error):
	try:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			popts, pcovs = opt.curve_fit(spin_fit_fun, N_vals, N_data, p0 = param_guess)
		xs = popts[num_spins:]
		error = error_fun(spin_fit_fun(N_vals, *popts), N_data)
		return popts, error, np.all((xs > 0) & (xs <= 2))
	except (RuntimeError, TypeError):
		raise FitError("Failed to fit spins with N data.")

# fit possibly several spins using a spin_fit_fun to N_data
# for reference, see http://stackoverflow.com/questions/13405053/scipy-leastsq-fit-to-a-sine-wave-failing
def spin_fit(N_vals, N_data, error_fun = squared_error, verbose = True, plots = True):
	coefs, freqs = sinusoid_coefs_fft(N_vals, N_data)
	likely_freqs = find_peaks_from_fourier(coefs, freqs)
	params, errors, spins_found = [], [], []
	# the higest number of spins that can be fit with l frequencies
	def highest_spin(l):
		if l <= 3:
			return l
		elif l == 4:
			return 3
		else:
			return 4 # make this 5 if you figure out how to do 5
	for num_spins in range(1,1 + highest_spin(len(likely_freqs))):
		approx_phis = gen_freqs_from_fourier(likely_freqs, num_spins)
		approx_xs = num_spins * [2.0]
		p0s = approx_phis + approx_xs
		try:
			popts, error, valid = spin_fit_guess(N_vals, N_data, num_spins, p0s, error_fun = error_fun)
			if valid:
				params.append(popts)
				errors.append(error)
				spins_found.append(num_spins)
		except FitError:
			pass
	if not params:
		raise FitError("Failed to fit spins with N data.")
	else:
		ind = np.argmin(errors)
		if plots:
			plt.figure(figsize=(10,10))
			plt.plot(freqs, coefs.real, label = 'real part')
			plt.plot(freqs, coefs.imag, label = 'imag part')
			plt.title('Fourier spectrum')
			plt.xlabel('phi')
			plt.ylabel('Fourier coeficient')
			plt.legend()
			plt.show()
			plt.figure(figsize=(10,10))
			plt.plot(N_vals, N_data, '.-k', lw=0.4, label = 'data')
			plt.plot(N_vals, spin_fit_fun(N_vals, *params[ind]), label = 'fit')
			plt.title('Fitted N data')
			plt.xlabel('N')
			plt.ylabel('M')
			plt.legend()
			plt.show()
		return params[ind], spins_found[ind]

def repeated_spin_fit(N_vals, N_data, error_tol = .1/64, error_fun = squared_error, num_subsets = 4, verbose = True, plots = True):
	spin_fits, scaled_errors = [], []
	for subset in [np.arange(int(len(N_vals) * r)) for r in np.linspace(0, 1, 1 + num_subsets)[1:]]:
		try:
			spin_ans = spin_fit(N_vals[subset], N_data[subset], error_fun = error_fun, verbose = verbose, plots = plots)
			spin_fits.append(spin_ans)
			scaled_errors.append(1.0 * error_fun(spin_fit_fun(N_vals, *spin_ans[0]), N_data)/len(N_vals))
		except FitError:
			pass
	if not spin_fits:
		raise FitError("Failed to fit spins with N data.")
	min_ind = np.argmin(scaled_errors)
	popt_spin, num_spins = spin_fits[min_ind]
	if min_ind != len(scaled_errors) - 1:
		try:
			popts, error, valid = spin_fit_guess(N_vals, N_data, num_spins, popt_spin, error_fun = error_fun)
			if valid:
				spin_fits.append((popts, num_spins))
				scaled_errors.append(1.0 * error_fun(spin_fit_fun(N_vals, *popts), N_data)/len(N_vals))
				min_ind = np.argmin(scaled_errors)
				popt_spin, num_spins = spin_fits[min_ind]
			if plots:
				plt.figure(figsize = (10,10))
				plt.plot(N_vals, N_data, '.-k', lw=0.4, label = 'data')
				plt.plot(N_vals, spin_fit_fun(N_vals, *popts), label = 'fit')
				plt.title('Fitted N data')
				plt.xlabel('N')
				plt.ylabel('M')
				plt.legend()
				plt.show()
		except FitError:
			pass
	if verbose:
		print "best: ", min_ind
	scaled_error = scaled_errors[min_ind]
	phis, xs = np.array(popt_spin[:num_spins]) % np.pi, np.array(popt_spin[num_spins:])
	phis, xs = map(np.array, zip(*sorted(zip(phis, xs), key = lambda x:-x[1])))# sort phis in decreasing order by xs
	return phis, xs, scaled_error    

# creates a tau vector with parameters that are suitable for the N value
def choose_tau_params(N):
	if N != 64:
		raise FitError("N should be 64")
	else:
		min_tau = 3.0000000000000001e-06
		max_tau = 2.1999999999999999e-05
		length = 5100
		tau = np.linspace(min_tau, max_tau, length)
		return tau

# finds resonances in a window of data. Assumes the difference between each index corresponds to time_unit seconds.
# it finds all relative minima below fit_dips_below and keeps only the num_dips widest of them
# returns the dip indices and the windows that define each dip, sorted by a measure of quality that accounts for both
# isolation and depth of dip
def find_resonances(data, fit_dips_below = None):
	min_inds = (signal.argrelextrema(data, np.less)[0]).astype(int)
	max_inds = (signal.argrelextrema(data, np.greater)[0]).astype(int)
	widths = max_inds[1:] - max_inds[:-1]
	dip_inds = min_inds[1:] if min_inds[0] < max_inds[0] else min_inds[:-1]
	windows = [(max_inds[i], max_inds[i+1]) for i in range(len(dip_inds)-1)] # should this be 3?
	# a dip must go below fit_dips_below, unless fit_dips_below = None
	filter_fun = lambda p: ((data[p[0]] < fit_dips_below) if (fit_dips_below != None) else True)
	dip_inds, widths, windows = map(np.array, zip(*filter(filter_fun, zip(dip_inds, widths, windows))))
	# sort by width
	sort_inds = np.argsort(-1 * widths)
	return dip_inds[sort_inds], windows[sort_inds]

# Calculates all possible omega_tilde values for a given phi and x at a given res_tau
def calc_omega_tilde(phi, x, res_tau, omega_larmor):
	beta = omega_larmor * res_tau
	cosbeta, cosphi_0 = np.cos(beta), np.cos(phi)
	lower_omega_bound, upper_omega_bound = omega_bounds(omega_larmor)
	omega_cosphi = []
	for cosphi in [cosphi_0, -cosphi_0]:
		prod = x * (1+cosbeta) * (1+cosphi)
		coeff = [1, prod - 2 * cosbeta * cosphi, prod + cosbeta ** 2 + cosphi ** 2 - 1]
		if coeff[1] ** 2 - 4 * coeff[0] * coeff[2] >= 0: # discriminant is nonnegative
			cosalphas = np.roots(coeff)
			for cosalpha in filter(lambda x: np.abs(x) <= 1, cosalphas):
				alpha_0_pi = np.arccos(cosalpha)
				for alpha_base in [alpha_0_pi, 2 * np.pi - alpha_0_pi]:
					lower_int = int(np.ceil((lower_omega_bound * res_tau - alpha_base)/(2 * np.pi)))
					upper_int = int(np.floor((upper_omega_bound * res_tau - alpha_base)/(2 * np.pi)))
					for n in range(lower_int, upper_int + 1):
						omega_cosphi.append(((alpha_base + 2 * np.pi * n)/res_tau, cosphi))
	return omega_cosphi

# for a given dip_ind, measures N data and fits it using analyze_tau. If the fit error is less than error_tol, it
# fits spins_per_dip of the spins to an omega value and therefore to an A and B. If this A and B are within the bounds
# it adds that spin to the spin_dict which was passed in. It returns the spin_dict and a boolean indicating if the fit
# was successful or not.
def analyze_dip(dip_ind, tau, data_func, omega_larmor, spin_dict = {}, error_tol = .1/64,
				N_vals = np.arange(0,256,2), error_fun = squared_error, num_subsets = 4, verbose = False, plots = False):
	res_tau = tau[dip_ind]
	N_data = data_func(N_vals, res_tau)
	try:
		phis, xs, scaled_error = repeated_spin_fit(N_vals, N_data, error_tol = error_tol, error_fun = error_fun,
			num_subsets = num_subsets, verbose = verbose, plots = plots)
	except FitError:
		if verbose:
			print "Failed to fit spins with N data."
		return spin_dict
	if verbose:
		print "res_tau: ", res_tau
		print "scaled_error <= error_tol: ", scaled_error <= error_tol
		print "scaled_error: ", scaled_error
		print "xs, phis, scaled_error: ", xs, phis, scaled_error
	if scaled_error > error_tol:
		return spin_dict
	for phi, x in zip(phis, xs):
		for omega_tilde, cosphi in calc_omega_tilde(phi, x, res_tau, omega_larmor): # all the possible omegas with their associated cosphis
			A, B = calc_A_B(cosphi, res_tau, omega_larmor, omega_tilde)
			if valid_A_B(A, B):
				if (dip_ind, phi, x) in spin_dict:
					spin_dict[(dip_ind, phi, x)] += [(A, B, cosphi)]
				else:
					spin_dict[(dip_ind, phi, x)] = [(A, B, cosphi)]
	return spin_dict

# takes in a dict of the type returned by analyze_dip and for each entry which represents a particular spin found at a particular dip
# it chooses the A, B value that minimizes the squared error from the tau data. It returns these A and B guesses along with the calculated errors.
def choose_spin_guesses(spin_dict, N, omega_larmor, tau, data, classifier, x_min = 1, error_fun = squared_error):
	guess_As, guess_Bs, dataerrs = [], [], []
	all_guess_As, all_guess_Bs, select_As, select_Bs = [], [], [], []
	for k in spin_dict.keys():
		err = []
		for A, B, _ in spin_dict[k]:
			all_guess_As.append(A)
			all_guess_Bs.append(B)
			err.append(error_fun(calc_M_single(A, B, N, omega_larmor, tau), data))
		min_err_ind = np.argmin(err)
		best_A, best_B, best_cosphi = spin_dict[k][min_err_ind]
		select_As.append(best_A)
		select_Bs.append(best_B)
		best_err = err[min_err_ind]
		_, _, x = k
		features = [best_cosphi, x, best_A, best_B, best_err]
		if classifier(features) and x >= x_min:
			guess_As.append(best_A)
			guess_Bs.append(best_B)
			dataerrs.append(best_err)
	return (np.array(guess_As), np.array(guess_Bs), np.array(dataerrs),
		np.array(all_guess_As), np.array(all_guess_Bs), np.array(select_As), np.array(select_Bs))

# given guesses for As and Bs, clusters these guesses using the DBSCAN algorithm.
# for info about DBSCAN see http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# for info about clustering with sklearn see http://scikit-learn.org/stable/modules/clustering.html#clustering
def cluster_spin_guesses(guess_As, guess_Bs, dataerrs, eps = .075, min_samples = 1):
	X = sk.preprocessing.StandardScaler().fit_transform(zip(guess_As, guess_Bs)) # possibly need to be lists?
	db = sk.cluster.DBSCAN(eps = eps, min_samples = min_samples).fit(X)
	labels = db.labels_
	cluster_As, cluster_Bs, cluster_dataerrs = [], [], []
	for k in np.unique(labels):
		if k >= 0:
			label_inds = np.where(labels == k)[0]
			best_guess = np.argmin(dataerrs[label_inds])
			cluster_As.append(guess_As[label_inds][best_guess])
			cluster_Bs.append(guess_Bs[label_inds][best_guess])
			cluster_dataerrs.append(dataerrs[label_inds][best_guess])
	return np.array(cluster_As), np.array(cluster_Bs), np.array(cluster_dataerrs)

# an approximation of the background due to the weakly coupled spins
#A_background = 4 * mag * (np.random.rand(400) - .5)
#B_background = 2 * mag * (np.random.rand(400))
#background_dict = learning.load_obj("background_A_B")
#A_background, B_background = background_dict["A_background"], background_dict["B_background"]

#Ab, Bb, rb, costhetab, sinthetab = NV_generator.generate_spins(500)
#background_dict = {"A" : Ab, "B" : Bb, "r" : rb, "costheta" : costhetab, "sintheta" : sinthetab}
#learning.store_obj(background_dict, "datasets/background_spins_" + len(Ab))
background_dict = learning.load_obj("datasets/background_spins_472")
A_background, B_background = background_dict["A"], background_dict["B"]

# given guess_As and guess_Bs, this function considers all ways of removing num_remove spins from the guess list
# it compares all of these possibilities along with not taking anything out in terms of the error from this subset to the data
# the error function considers the spins along with 400 weakly coupled spins which approximates the background.
# it returns the optimal subset as well as a number to remove next time: the same number if a spin was removed, otherwise the number
# considered for removal this time plus one (considering taking several out at the same time)
def remove_spins(guess_As, guess_Bs, N, omega_larmor, tau, data, num_remove = 1, error_fun = squared_error, verbose = False):
	# creates all subsets of s with size between lower and upper
	def subset_size_range(s, lower, upper):
		ans = []
		for i in range(lower, upper+1):
			for j in itertools.combinations(s, i):
				ans.append(j)
		return ans
	guess_As, guess_Bs = np.array(guess_As), np.array(guess_Bs)
#	background_inds = np.where(A_background ** 2 + B_background ** 2 < min(guess_As ** 2 + guess_Bs ** 2))[0]
#	if verbose:
#		print "background inds start with: ", min(background_inds)
#		print "number of background spins included: ", len(background_inds)
	min_background_ind = max(50, len(guess_As))
	M_background = calc_M(A_background[min_background_ind:], B_background[min_background_ind:], N, omega_larmor, tau)
	err, As, Bs = [], [], []
	for subset in subset_size_range(range(len(guess_As)), len(guess_As) - num_remove, len(guess_As) - 1):
		subset = np.array(subset)
		err.append(error_fun(M_background * calc_M(guess_As[subset], guess_Bs[subset], N, omega_larmor, tau), data))
		As.append(guess_As[subset])
		Bs.append(guess_Bs[subset])
	best_ind = np.argmin(err)
	orig_error = error_fun(M_background * calc_M(guess_As, guess_Bs, N, omega_larmor, tau), data)
	if verbose:
		print "new error, old error: ", err[best_ind], orig_error
	if orig_error > err[best_ind]:
		return As[best_ind], Bs[best_ind], num_remove, M_background
	else:
		return guess_As, guess_Bs, num_remove + 1, M_background

def analyze_diamond(data_func, N, omega_larmor, verbose = False, plots = False):
	tau = choose_tau_params(N)
	data = data_func(N, tau)
	dip_inds, windows = find_resonances(data, fit_dips_below = None)
	spin_dict = {}
	for dii in range(len(dip_inds)):
		dip_ind = dip_inds[dii]
		if dip_ind >= 3220: # 15 microseconds and on
			spin_dict = analyze_dip(dip_ind, tau, data_func, omega_larmor, spin_dict, N_vals = np.arange(0,256,2),
				error_tol = .1/64, verbose = verbose, plots = plots)
	guess_scaler = learning.load_obj("classifiers/scaler_svm_rbf_di3220_29diamonds_cxABe")
	guess_clf = learning.load_obj("classifiers/clf_svm_rbf_di3220_29diamonds_cxABe")
	def guess_classifier(features):
		return guess_clf.predict(guess_scaler.transform([features]))
	guess_As, guess_Bs, dataerrs, all_guess_As, all_guess_Bs, select_As, select_Bs = choose_spin_guesses(spin_dict, N, omega_larmor, tau, data, guess_classifier, x_min = 1, error_fun = squared_error)
	cluster_As, cluster_Bs, cluster_dataerrs = cluster_spin_guesses(guess_As, guess_Bs, dataerrs, eps = .075, min_samples = 1)
	As, Bs, num_remove = cluster_As, cluster_Bs, 1
	while num_remove <= 2:
		As, Bs, num_remove, M_background = remove_spins(As, Bs, N, omega_larmor, tau, data, num_remove = num_remove, error_fun = squared_error, verbose=verbose)
	return As, Bs, all_guess_As, all_guess_Bs, select_As, select_Bs, guess_As, guess_Bs, cluster_As, cluster_Bs, M_background

"""

# calculates the right side of equation 10 of the supplement of Taminiau et al 2012 minus the left side as a function of omega (omega_tilde).
# anywhere where mz is more than 1 in magnitude, the result is -1-x where x is the left side.
# it also returns whether or not the calculated curve goes through 0
def omega_root_fun(omega, res_tau, cosphi, x, omega_larmor):
	omega = np.array(omega)
	beta = omega_larmor * res_tau
	alpha = omega * res_tau
	mz = (np.cos(alpha) * np.cos(beta) - cosphi)/(np.sin(alpha) * np.sin(beta))
	x_approx = (1 - mz ** 2) * (1-np.cos(alpha)) * (1-np.cos(beta))/(1+cosphi)
	bad_inds = np.where(mz ** 2 > 1)[0]
	x_approx[bad_inds] = -1
	return x_approx - x, np.all(x_approx < x) or np.all(x_approx > x)


# fits the N data at a particular tau with spins (trying several subsets of N_vals and taking whichever fits best)
# finds the correct value for cosphi (finds its sign correctly) and also the omege root array for that value of cosphi
# returns the xs along with their cosphis and root arrays as well as the error level that was achieved in the spin fit
def analyze_tau(tau_fixed, N_data, omega_larmor, omega_array, N_vals, plots = False):
	spin_fits = []
	scaled_errors = []
	for subset in [np.arange(len(N_vals)/2), np.arange(len(N_vals) * 3/4), np.arange(len(N_vals))]: # different N ranges
		try:
			spin_ans = spin_fit(N_vals[subset], N_data[subset], verbose = False, plots = plots)
			spin_fits.append(spin_ans)
			scaled_errors.append(1.0 * spin_ans[1]/len(subset))
		except FitError:
			pass
	if not spin_fits:
		raise FitError("Failed to fit spin with N data.")
	popt_spin, spin_error, num_spins = spin_fits[np.argmin(scaled_errors)]
	xs = popt_spin[num_spins:]
	phis, xs = map(np.array, zip(*sorted(zip(popt_spin[:num_spins], xs), key = lambda x:-x[1])))# sort phis in decreasing order by xs
	cosphis = np.cos(phis) # could be positive or negative, if this is wrong it gets corrected below
	corrected_cosphis, root_arrays = [], []
	for x, cosphi in zip(xs, cosphis):
		correct_cosphi = cosphi
		root_array, wrong_sign = omega_root_fun(omega_array, tau_fixed, correct_cosphi, x, omega_larmor)
		if wrong_sign:
			correct_cosphi = -cosphi
			root_array, wrong_sign = omega_root_fun(omega_array, tau_fixed, correct_cosphi, x, omega_larmor)
		root_arrays.append(root_array)
		corrected_cosphis.append(correct_cosphi)
	return xs, corrected_cosphis, root_arrays, min(scaled_errors)

# for a given dip_ind, measures N data and fits it using analyze_tau. If the fit error is less than error_tol, it
# fits spins_per_dip of the spins to an omega value and therefore to an A and B. If this A and B are within the bounds
# it adds that spin to the spin_dict which was passed in. It returns the spin_dict and a boolean indicating if the fit
# was successful or not.
def analyze_dip(dip_ind, tau, data_func, omega_larmor, spin_dict = {}, error_tol = .1/64, omega_acc = .0001 * mag,
				spins_per_dip = 2, N_vals = np.arange(0,256,2), verbose = False, plots = False):
	lower_bound, upper_bound = omega_bounds(omega_larmor)
	omega_array = np.linspace(lower_bound, upper_bound, (upper_bound - lower_bound)/omega_acc)
	tau_fixed = tau[dip_ind]
	N_data = data_func(N_vals, tau_fixed)
	try:
		xs, cosphis, root_arrays, scaled_error = analyze_tau(tau_fixed, N_data, omega_larmor, omega_array, N_vals = N_vals, plots = plots)
		if verbose:
			print "xs, cosphis, scaled_error: ", xs, cosphis, scaled_error
			print "scaled_error > error_tol: ", scaled_error > error_tol
		if scaled_error > error_tol:
			return spin_dict, False
		for x, cosphi, root_array in zip(xs, cosphis, root_arrays)[:spins_per_dip]: # find the roots of the root_array
			sign_diffs = np.diff(np.sign(root_array))
			zero_inds = np.concatenate((np.where(sign_diffs == 2.0)[0] + 1, np.where(sign_diffs == -2.0)[0]))
			for omega in omega_array[zero_inds]: # all the possible omegas
				A, B = calc_A_B(cosphi, omega, omega_larmor, tau_fixed)
				if valid_A_B(A, B):
					if (dip_ind, x, cosphi) in spin_dict:
						spin_dict[(dip_ind, x, cosphi)] += [(A, B)]
					else:
						spin_dict[(dip_ind, x, cosphi)] = [(A, B)]
		return spin_dict, True
	except FitError:
		return spin_dict, False



# takes in a dict of the type returned by analyze_dip and for each entry which represents a particular spin found at a particular dip
# it chooses the A, B value that minimizes the squared error from the tau data. It returns these A and B guesses along with the calculated errors.
def choose_spin_guesses(spin_dict, N, omega_larmor, tau, data, x_min = 1.5, error_fun = squared_error):
	guess_As, guess_Bs, dataerrs = [], [], []
	for k in spin_dict.keys():
		if k[2] >= x_min: # the x value
			err, all_B_abs = [], []
			for A, B in spin_dict[k]:
				err.append(error_fun(calc_M_single(A, B, N, omega_larmor, tau), data))
				all_B_abs.append(np.abs(B))
			best_A, best_B = spin_dict[k][np.argmin(err)]
			if best_B >= mag: # in case it's just choosing the one with smallest B since it is usually 1
				guess_As.append(best_A)
				guess_Bs.append(best_B)
				dataerrs.append(min(err))
	return np.array(guess_As), np.array(guess_Bs), np.array(dataerrs)


def analyze_diamond(data_func, N, omega_larmor, verbose = False, plots = False):
	tau = choose_tau_params(N)
	data = data_func(N, tau)
	dip_inds, windows = find_resonances(data, fit_dips_below = None)
	dip_scaler, dip_clf = learning.load_obj("classifiers/dip_scaler_rbf"), learning.load_obj("classifiers/dip_clf_rbf")
	def classify_dii(dii):
		features = learning.dip_features(dii, dip_inds, data, tau, windows)
		return dip_clf.predict(dip_scaler.transform([features]))
	clf_dii = filter(classify_dii, range(len(dip_inds)))
	spin_dict = {}
	for dii in clf_dii:
		dip_ind = dip_inds[dii]
		spin_dict = analyze_dip(dip_ind, tau, data_func, omega_larmor, spin_dict, spins_per_dip = 2, N_vals = np.arange(0,256,2),
			error_tol = .1/64, verbose = verbose, plots = plots)
	guess_As, guess_Bs, dataerrs = choose_spin_guesses(spin_dict, N, omega_larmor, tau, data, x_min = 1.5, error_fun = squared_error)
	cluster_As, cluster_Bs, cluster_dataerrs = cluster_spin_guesses(guess_As, guess_Bs, dataerrs, eps = .075, min_samples = 1)
	As, Bs, num_remove = cluster_As, cluster_Bs, 1
	while num_remove < 3:
		As, Bs, num_remove = remove_spins(As, Bs, N, omega_larmor, tau, data, num_remove = num_remove, error_fun = squared_error)
	return As, Bs
"""
