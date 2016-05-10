import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import os,sys,inspect,shutil
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import analysis
import NV_generator
figures_path = "../../../Writing/Thesis/Figures/"

# Constants
delta = 2 * np.pi * 2.88 * 10 ** 9 # zero field splitting is 2.88 GHz in real frequency
B_field = 0.0403555 # Teslas
gam_el = -1.760859 * 10 ** 11 # Gyromagnetic ratio for a single electron in rad s-1 T-1
gam_c = 67.262 * 10 ** 6 # Gyromagnetic ratio for a single c13 nucleus in rad s-1 T-1
hbar = 1.05457173 * 10 ** (-34)
h = hbar * 2 * np.pi
mu0 = 4 * np.pi * 10 ** (-7)
omega_larmor = -1 * gam_c * B_field
tau = analysis.choose_tau_params(64)

A, B, r, costheta, sintheta = 59891.5741878, 610776.610967, 6.67885843539e-10, 0.617213399848, 0.786795792469
print A/(2 * np.pi * 1e3), B/(2 * np.pi * 1e3), r

# Spin 1/2 states for NV center
S0 = qt.basis(2,0)
S1 = qt.basis(2,1)

# Spin 1/2 operators for c13
Ii = qt.qeye(2)
Ix = qt.jmat(.5, 'x')
Iy = qt.jmat(.5, 'y')
Iz = qt.jmat(.5, 'z')

# Hamiltonian in angular frequency units
H = qt.tensor(S0 * S0.dag(), omega_larmor * Iz) + qt.tensor(S1 * S1.dag(), (A + omega_larmor) * Iz + B * Ix)

# Pi pulse on NV spin between ms = 0 and ms = 1 states
# I'm still assuming this is "infinitely" fast, only involves the NV, and is a complete population inversion
# because I don't know what hamiltonian to use realistically for the microwave field
pi_pulse = qt.tensor(-1j * qt.sigmax(), Ii)

analytical_P = .5 * (1+analysis.calc_M_single(A, B, 64, omega_larmor, tau))
displayed_inds = np.where(np.logical_and(tau>=3*1e-6, tau<=5*1e-6))[0]
res_ind = displayed_inds[np.argmin(analytical_P[displayed_inds])]
t_step = tau[100]-tau[99]
print tau[res_ind], analytical_P[res_ind]

N_vals = np.arange(0,128,2)
analytical_P = .5 * (1+analysis.calc_M_single(A, B, N_vals, omega_larmor, tau[res_ind]))
res_N = N_vals[np.argmin(analytical_P[:15])]
print res_N, analytical_P[res_N/2]

speed = 12
num_pts = 2 * res_ind * res_N / speed
print num_pts, res_ind, res_N, speed, res_ind % speed
U = (-1j * t_step * speed * H).expm() 

rho_init = qt.tensor(qt.ket2dm(S0), .5 * (Ii + qt.sigmax()))
rho = rho_init
rho_s, rho_c = [rho.ptrace(0)], [rho.ptrace(1)]
rotation_vector = [[0,0,-1]] 
counter = 0
first_pulse = True
s_up = True
omega_tilde = np.sqrt((A + omega_larmor) ** 2 + B ** 2)
for ind in range(num_pts):
	if (first_pulse and counter == (res_ind/speed)) or ((not first_pulse) and counter == 2 * (res_ind/speed)):
		UorP = pi_pulse
		print "pulse ", counter, res_ind/speed
		counter = 0
		first_pulse = False
		s_up = not s_up
	else:
		UorP = U
		counter += 1
	rho = UorP * rho * UorP.dag()
	rho_s.append(rho.ptrace(0))
	rho_c.append(rho.ptrace(1))
	if s_up:
		rotation_vector.append([0,0,-1])
	else:
		rotation_vector.append([B/omega_tilde, 0, (A+omega_larmor)/omega_tilde])


movie_dir = 'movie_dd_NV0_Cpx'
for ind in range(len(rho_s)):
	b = qt.Bloch()
	b.view = [-40,30]
	b.add_states(rho_c[ind], kind='vector')
	b.add_states(rho_s[ind], kind='vector')
	b.add_vectors(rotation_vector[ind])
	b.save(dirc=movie_dir+'/tmp')
	shutil.move(movie_dir+'/tmp/bloch_0.png', movie_dir+'/bloch_' + str(ind) + '.png')