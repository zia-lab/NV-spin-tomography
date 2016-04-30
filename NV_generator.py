import numpy as np
import itertools

# Constants
# lattice constant of diamond
a0 = 3.567 * 10 ** (-10) # in meters
# Hyperfine coupling related constants
gam_el = -1.760859 * 10 ** 11 # Gyromagnetic ratio for a single electron in rad s-1 T-1
gam_n = 67.262 * 10 ** 6 # Gyromagnetic ratio for a single c13 nucleus in rad s-1 T-1
hbar = 1.05457173 * 10 ** (-34)
pi = np.pi
h = hbar * 2 * pi
mu0 = 4 * pi * 10 ** (-7)

# Carbon Lattice Definition
# Basis vectors
# i, j, k are basis vectors of Face Centered Cubic Bravais lattice
i = a0/2*np.array([0,1,1])
j = a0/2*np.array([1,0,1])
k = a0/2*np.array([1,1,0])
# a and b are the relative positions of the nitrogen and vacancy
a = np.array([0,0,0])
b = a0/4*np.array([1,1,1])
# rotate all the vectors so that the symmetry axis (b) of the NV center is along the z direction
# rotation around z axis by pi/4 puts b in yz plane
Rz = np.array([[np.cos(pi/4),-np.sin(pi/4),0],[np.sin(pi/4),np.cos(pi/4),0],[0,0,1]])
sqrt2angle = np.arctan(np.sqrt(2))
# rotation around x axis by sqrt2angle moves b from yz plane to z axis
Rx = np.array([[1,0,0],[0,np.cos(sqrt2angle),-np.sin(sqrt2angle)],[0,np.sin(sqrt2angle),np.cos(sqrt2angle)]])
b = Rx.dot(Rz).dot(b)
i = Rx.dot(Rz).dot(i)
j = Rx.dot(Rz).dot(j)
k = Rx.dot(Rz).dot(k)

hyperfine_prefactor = -1 * mu0 * gam_el * gam_n * hbar/(4 * pi) # Only one hbar instead of hbar ** 2 since we convert to angular frequency units in the hamiltonian.

# Takes Carbon Concentration, and gridsize (N) as inputs
# Returns a list of 2D numpy arrays with hyperfine couplings of c13 nuclei to the NV center in angular frequency Hz
# If sphere is true the c13 nuclei inside a sphere are returned instead of those inside a cube.	
def generate_NV(c13_concentration=0.011,N=25,sphere = True):
	# define position of NV in middle of the grid
	center = round(N/2)
	NVPos = center * i + center * j + center * k

	spin_list = []
	#Calculate Hyperfine strength for all gridpoints
	for n, m, l in itertools.product(range(N), repeat = 3):
		if (n,m,l) != (center, center, center): # all lattice pairs except the nitrogen and vacancy
			lattice_position = n * i + m * j + l * k - NVPos # relative position of a lattice point to the NV center
			for pos in [lattice_position, lattice_position + b]: # the two lattice positions at this grid location
				x, y, z = pos
				r = np.sqrt(pos.dot(pos))
				costheta = z/r
				sintheta = np.sqrt(x ** 2 + y ** 2)/r
				A = hyperfine_prefactor * (3 * costheta ** 2 - 1)/(r ** 3)
				B = hyperfine_prefactor * 3 * costheta * sintheta /(r ** 3)
				spin_list.append((r, A, B, costheta, sintheta))

	if sphere:
		spin_list.sort() # sort by r value from smallest to largest
		spin_list = spin_list[:len(spin_list)/2] # keep the closest half of the spins

	# for each lattice point, let it be a c13 with probability c13_concentration 
	c13_inds = np.where(np.random.rand(len(spin_list))<c13_concentration)[0]
	c13_spins = np.array(spin_list)[c13_inds]
	r, A, B, costheta, sintheta = zip(*c13_spins)
	return np.array(A), np.array(B), np.array(r), np.array(costheta), np.array(sintheta)

def generate_spins(num_spins, enforce_bound = True, coupling_bound = 150 * 2 * pi * 1e3, c13_concentration = 0.011, N = None, abs_val = False, verbose = False):
	if N == None:
		N = int(np.ceil(((num_spins*1.0)/c13_concentration) ** (1.0/3)))
	if verbose:
		print "N:", N
	A, B, r, costheta, sintheta = generate_NV(N = N, c13_concentration = c13_concentration)
	while enforce_bound and (np.any(A ** 2 + B ** 2 > (coupling_bound) ** 2)):
		A, B, r, costheta, sintheta = generate_NV(N = N, c13_concentration = c13_concentration)
	if abs_val:
		B = np.abs(B)
	return A, B, r, costheta, sintheta



