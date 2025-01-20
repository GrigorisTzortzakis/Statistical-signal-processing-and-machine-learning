import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Change backend to TkAgg (compatible with interactive plotting)
import matplotlib.pyplot as plt

N = 1000  # Samples/deigmata
L = 4  # 4 coefficients/4 sidelestes
mu = 0.01  # Step size
np.random.seed(0)

# Input signal/sima eisodou
x = np.random.normal(0, 1, N)

# Lms implementation/ilopoihsh lms
def lms_time_varying(x, b_of_n, mu, N, L):
    """
    LMS Algorithm for Time-Varying Systems.
    Returns:
        - e_vec: Error at each iteration
        - w_evol: Weight evolution over iterations
    """
    w_current = np.zeros(L)  # Initialize weights
    e_vec = np.zeros(N)  # Error
    w_evol = np.zeros((N, L))  # Weight evolution

    for n in range(L, N):
        # Compute the time-varying system output d(n)
        b_n = b_of_n(n)  # Coefficient b(n) at time n
        d_n = b_n * x[n] - 0.4 * x[n - 1] - 4.0 * x[n - 2] + 0.5 * x[n - 3]

        # Input vector for the filter
        x_vec = x[n:n - L:-1]  # x[n], x[n-1], ..., x[n-(L-1)]

        # Filter output
        y_n = np.dot(w_current, x_vec)

        # Error
        e_n = d_n - y_n
        e_vec[n] = e_n

        # LMS weight update
        w_current += mu * e_n * x_vec

        # Store the weights
        w_evol[n, :] = w_current

    return e_vec, w_evol


# Smooth Variation: b(n) = 1 / (1 + exp(-0.02 * n))
def b_smooth(n):
    return 1.0 / (1.0 + np.exp(-0.02 * n))


# Abrupt Variation: b(n) = 100 for n <= 500, 0 for n > 500
def b_abrupt(n):
    if n <= 500:
        return 100.0
    else:
        return 0.0


# Run LMS Algorithm for Both Variations
e_smooth, w_smooth = lms_time_varying(x, b_smooth, mu, N, L)
e_abrupt, w_abrupt = lms_time_varying(x, b_abrupt, mu, N, L)

# Compute Mean Squared Error (MSE)
mse_smooth = e_smooth**2
mse_abrupt = e_abrupt**2

# Normalize MSE for comparison
mse_smooth /= np.max(mse_smooth)  # Normalize smooth MSE
mse_abrupt /= np.max(mse_abrupt)  # Normalize abrupt MSE

# Plot Learning Curves
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(mse_smooth) / np.arange(1, N + 1), label="Smooth Variation (2.1)", color="blue")
plt.plot(np.cumsum(mse_abrupt) / np.arange(1, N + 1), label="Abrupt Variation (2.2)", color="red")
plt.title("Learning Curves for LMS Algorithm with Time-Varying Impulse Responses")
plt.xlabel("Iterations")
plt.ylabel("Normalized Mean Squared Error (MSE)")
plt.legend()
plt.grid()
plt.show()

# Plot Weight Evolution
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(w_smooth)
axes[0].set_title("Weight Evolution for Smooth Variation (2.1)")
axes[0].set_xlabel("Iterations")
axes[0].set_ylabel("Weight Values")
axes[0].legend([f"w_{i}" for i in range(L)])
axes[0].grid()

axes[1].plot(w_abrupt)
axes[1].set_title("Weight Evolution for Abrupt Variation (2.2)")
axes[1].set_xlabel("Iterations")
axes[1].set_ylabel("Weight Values")
axes[1].legend([f"w_{i}" for i in range(L)])
axes[1].grid()

plt.tight_layout()
plt.show()

# Display Final Weights
print("Final Weights for Smooth Variation:")
print(w_smooth[-1, :])

print("\nFinal Weights for Abrupt Variation:")
print(w_abrupt[-1, :])
