import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt


N = 1000  # Samples/deigmata
np.random.seed(0)
x = np.random.randn(N)  # Input signal/sima eisodou
h = np.array([1, -0.4, -4, 0.5])  # Coefficients/Sidelestes
d = np.convolve(x, h, mode='full')[:N]  # d(n)
mu = 0.01  # Step size/vima μ

# Filter length/mhkos filtrou
filter_lengths = [3, 5]


learning_curves = []

for L in filter_lengths:
    w = np.zeros(L)  # Initialize LMS filter coefficients
    mse = np.zeros(N)  # To track squared error

# Loop for each length/epanalipseis gia to kathe mhkos
# Lms implementation/ilopoihsh lms
    for n in range(L, N):

        X = x[n:n - L:-1]

        # filter output/eksodos filtrou
        y = np.dot(w, X)

        # Error of filter output signal compared to the desired signal/sfalma eksodou
        e = d[n] - y

        # New filter coefficients/Enimerosi sideleston me kathe epanalipsi
        w += mu * e * X


        mse[n] = e ** 2


    cumulative_mse = np.cumsum(mse) / np.arange(1, N + 1)
    learning_curves.append(cumulative_mse)


plt.figure(figsize=(10, 6))
for idx, L in enumerate(filter_lengths):
    plt.plot(learning_curves[idx], label=f'L = {L}')
plt.title('Learning Curves for LMS Algorithm')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()  
