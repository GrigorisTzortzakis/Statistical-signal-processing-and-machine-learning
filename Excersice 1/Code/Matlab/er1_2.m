% Step 1: Generate Input Signal
N = 1000; % Length of the signal
x = randn(1, N); % White Gaussian noise

% Step 2: Simulate the Unknown System
h = [1, -0.4, -4, 0.5]; % Coefficients of H(z)
d = filter(h, 1, x); % Output signal using H(z)

% Step 3: Form the Input Matrix
L = length(h); % Filter length
X = zeros(N, L);
for i = 1:L
    X(i:end, i) = x(1:end-i+1);
end

% Step 4: Compute Autocorrelation Matrix
R_x = (X' * X) / N; % Autocorrelation matrix

% Step 5: Compute Cross-Correlation Vector
p = (X' * d') / N; % Cross-correlation vector

% Step 6: Solve for Wiener Filter Coefficients
w = R_x \ p; % Solve Wiener-Hopf equation

% Step 7: Display the Results
disp('Estimated Coefficients of H(z):');
disp(w');
disp('Actual Coefficients of H(z):');
disp(h);
