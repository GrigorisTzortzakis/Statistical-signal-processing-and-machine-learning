import numpy as np
from scipy.signal import lfilter

# Step 1: Generate Input Signal
N = 1000  # Length of the signal
x = np.random.randn(N)  # White Gaussian noise

# Step 2: Simulate the Unknown System
h = [1, -0.4, -4, 0.5]  # Coefficients of H(z)
d = lfilter(h, [1], x)  # Output signal using H(z)

# Step 3: Form the Input Matrix
L = len(h)  # Filter length
X = np.zeros((N, L))
for i in range(L):
    X[i:, i] = x[:N - i]

# Step 4: Compute Autocorrelation Matrix
R_x = (X.T @ X) / N  # Autocorrelation matrix

# Step 5: Compute Cross-Correlation Vector
p = (X.T @ d) / N  # Cross-correlation vector

# Step 6: Solve for Wiener Filter Coefficients
w = np.linalg.solve(R_x, p)  # Solve Wiener-Hopf equation

# Step 7: Display the Results
print("Estimated Coefficients of H(z):")
print(w)
print("Actual Coefficients of H(z):")
print(h)
