%  input signal/shma eisodou
N = 1000; % Number of samples/arithmos deigmaton
x = randn(1, N); % White Gaussian noise with 0 mean values and dispersion of 1/leukos thorivos opos zitite

%  Simulating the Unknown System/prosomiosi tou agnostou sistimatos
h = [1, -0.4, -4, 0.5]; % Coefficients of H(z)/sidelestes pou dinode
d = filter(h, 1, x); % Output signal of the system/eksodon d(n)

% Forming the Input matrix/mhtroo eisodou
L = length(h); % Filter length/mhkos filtrou
X = zeros(N, L);
for i = 1:L
    X(i:end, i) = x(1:end-i+1);
end

%  Computing  Autocorrelation matrix/ipologismos matrhou autosisxetisis
R_x = (X' * X) / N; 

%  Computing Cross-Correlation Vector/ ipologismos dianismatos
%  autosisxetisis
p = (X' * d') / N; 

% Solving the wiener hopf equation/ lisi eksisosis
w = R_x \ p;


disp('Estimated Coefficients of H(z):');
disp(w');
disp('Actual Coefficients of H(z):');
disp(h);
