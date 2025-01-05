import matplotlib

matplotlib.use("TkAgg")  # Set a compatible backend for interactive plotting
import numpy as np
import matplotlib.pyplot as plt


N = 1000  # Samples/deigmata
L = 4  # 4 coefficients/4 sidelestes
np.random.seed(0)
x = np.random.randn(N)  # Input signal/sima eisodou
h = np.array([1, -0.4, -4, 0.5])  # Coefficients/Sidelestes
d = np.convolve(x, h, mode='full')[:N]  # d(n)

# Computing the autocorrelation matrix Rx/Ipologismos Rx
X_full = np.zeros((L, N - L + 1))
for i in range(L):
    X_full[i, :] = x[i:N - L + i + 1]
R_x = np.dot(X_full, X_full.T) / (N - L)

# Maximum step size μ/ Megisti timh tou μ
lambda_max = np.max(np.linalg.eigvals(R_x))
mu_max = 2 / lambda_max

# Step sizes required/thetoume to μ pou zitite
mu_values = np.array([0.001, 0.01, 0.1, 0.5]) * mu_max  # Step sizes

# Loop for all step sizes/kanoume loop oste na treksoun ola ta μ
final_coefficients = np.zeros((len(mu_values), L))
coeff_evolution = []

for i, mu in enumerate(mu_values):
    w = np.zeros(L)
    coeffs_over_time = []

# Lms implementation/ilopoihsh lms
    for n in range(L, N):

        X = np.array([x[n], x[n - 1], x[n - 2], x[n - 3]])

        # filter output/eksodos filtrou
        y = np.dot(w, X)

        # Error of filter output signal compared to the desired signal/sfalma eksodou
        e = d[n] - y

        # New filter coefficients/Enimerosi sideleston me kathe epanalipsi
        w += mu * e * X


        coeffs_over_time.append(w.copy())

    #  Save coefficient evolution/ekseliksei sideleston
    final_coefficients[i] = w
    coeff_evolution.append(np.array(coeffs_over_time))


for i, mu in enumerate(mu_values):
    print(f"Step size (mu): {mu:.6f}")
    print("Final LMS Filter Coefficients:")
    print(final_coefficients[i])
    print("-" * 40)


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, mu in enumerate(mu_values):
    ax = axes[i]
    coeffs = coeff_evolution[i]
    for j in range(L):
        ax.plot(coeffs[:, j], label=f"w{j + 1}")
    ax.set_title(f"Coefficient Evolution for μ = {mu:.6f}")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Coefficient Value")
    ax.legend()
plt.tight_layout()
plt.show()  
