import numpy as np
from scipy.signal import lfilter

# input signal/shma eisodou
N = 1000  # Number of samples/arithmos deigmaton
x = np.random.randn(N)  # White Gaussian noise with 0 mean values and dispersion of 1/leukos thorivos opos zitite

# Simulating the Unknown System/prosomiosi tou agnostou sistimatos
h = [1, -0.4, -4, 0.5]  # Coefficients of H(z)/sidelestes pou dinode
d = lfilter(h, [1], x)  # Output signal of the system/eksodon d(n)

# Forming the Input matrix/mhtroo eisodou
L = len(h)  # Filter length/mhkos filtrou
X = np.zeros((N, L))
for i in range(L):
    X[i:, i] = x[:N - i]

# Computing  Autocorrelation matrix/ipologismos matrhou autosisxetisis
R_x = (X.T @ X) / N

#  Computing Cross-Correlation Vector/ ipologismos dianismatos autosisxetishs
p = (X.T @ d) / N

# Solving the wiener hopf equation/ lisi eksisosis
w = np.linalg.solve(R_x, p)


print("Estimated Coefficients of H(z):")
print(w)
print("Actual Coefficients of H(z):")
print(h)
