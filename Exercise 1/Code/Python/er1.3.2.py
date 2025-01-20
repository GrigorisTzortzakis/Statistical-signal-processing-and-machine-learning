import numpy as np


N = 1000  # Number of samples/arithmos deigmaton
x = np.random.randn(N)  # Input signal/sima eisodou
h = np.array([1, -0.4, -4, 0.5])  # Given coefficients/H(z)
d = np.convolve(x, h, mode='full')[:N]  # d(n)

L = 4  # 4 coefficients/4 sidelestes
w = np.zeros(L)  # Initial coefficients of lms are zero/miden arxikoi sidelestes

# Computing the autocorrelation matrix Rx/Ipologismos mitroou Rx
X_full = np.zeros((L, N - L + 1))
for i in range(L):
    X_full[i, :] = x[i:N - L + i + 1]
R_x = np.dot(X_full, X_full.T) / (N - L)

# Maximum step size/Prosdiorismos μ
lambda_max = np.max(np.linalg.eigvals(R_x))
mu_max = 2 / lambda_max
mu = 0.1 * mu_max

# Lms implementation/ilopoihsh lms
for n in range(L, N):

    X = np.array([x[n], x[n - 1], x[n - 2], x[n - 3]])

    # filter output/eksodos filtrou
    y = np.dot(w, X)

    # Error of filter output signal compared to the desired signal/sfalma eksodou
    e = d[n] - y

    # New filter coefficients/Enimerosi sideleston me kathe epanalipsi
    w += mu * e * X


print("Estimated LMS Filter Coefficients/Lms apotelesmata:")
print(w)
print("Actual System Coefficients/Pragmatika apotelesmata:")
print(h)
