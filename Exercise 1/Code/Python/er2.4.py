import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Change backend to TkAgg (compatible with interactive plotting)
import matplotlib.pyplot as plt

# Step 1: Parameters
N = 1000  # Number of samples
L = 4  # Number of filter coefficients
mu = 0.01  # Step size
num_realizations = 20  # Number of realizations

# Step 2: Function to calculate LMS for a single realization
def lms_algorithm(x, b_smooth):
    w = np.zeros(L)  # Initialize filter coefficients
    e_vec = np.zeros(N)  # Squared error for this realization

    for n in range(L, N):
        # Generate d(n) using the smooth b_smooth
        x_vec = x[n : n - L : -1]  # Input vector
        d_n = b_smooth[n] * x[n] - 0.4 * x[n - 1] - 4 * x[n - 2] + 0.5 * x[n - 3]
        y_n = np.dot(w, x_vec)  # Filter output
        e_n = d_n - y_n  # Error
        w += mu * e_n * x_vec  # LMS weight update
        e_vec[n] = e_n**2  # Squared error

    return e_vec

# Step 3: Perform the experiment for 20 realizations
all_squared_errors = np.zeros((num_realizations, N))

for realization in range(num_realizations):
    x = np.random.randn(N)  # Generate random input signal
    b_smooth = 1.0 / (1.0 + np.exp(-0.02 * np.arange(N)))  # Smooth variation
    all_squared_errors[realization, :] = lms_algorithm(x, b_smooth)

# Step 4: Compute average and standard deviation of squared error
avg_squared_error = np.mean(all_squared_errors, axis=0)
std_squared_error = np.std(all_squared_errors, axis=0)

# Step 5: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(avg_squared_error, label="Average Squared Error", color="blue")
plt.fill_between(
    np.arange(N),
    avg_squared_error - std_squared_error,
    avg_squared_error + std_squared_error,
    color="blue",
    alpha=0.3,
    label="Standard Deviation",
)
plt.title("Average Squared Error over 20 Realizations")
plt.xlabel("Iterations")
plt.ylabel("Average Squared Error (e^2)")
plt.grid(True)
plt.legend()
plt.show()
