
N = 1000; % Number of samples/arithmos deigmaton
x = randn(1, N); % Input signal/sima eisodou
h = [1, -0.4, -4, 0.5]; % Given coefficients/H(z)
d = filter(h, 1, x); % d(n)

L = 4; % 4 coefficients/4 sidelestes
w = zeros(1, L); % Initial coefficients of lms are zero/miden arxikoi sidelestes

% Computing the autocorrelation matrix Rx/Ipologismos mitroou Rx
X_full = zeros(L, N-L+1); 
for i = 1:L
    X_full(i, :) = x(i:N-L+i); 
end
R_x = (X_full * X_full') / (N - L); 

% Maximum step size/Prosdiorismos μ
lambda_max = max(eig(R_x)); % Largest eigenvalue of R_x
mu_max = 2 / lambda_max;
mu = 0.1 * mu_max; 

% Lms implementation/ilopoihsh lms
for n = L:N
    
    X = [x(n), x(n-1), x(n-2), x(n-3)];
    
    % filter output/eksodos filtrou
    y = w * X'; 
    
    % Error of filter output signal compared to the desired signal/sfalma
    % eksodou
    e = d(n) - y; 
    
    % New filter coefficients/Enimerosi sideleston me kathe epanalipsi
    w = w + mu * e * X; 
end


disp('Estimated LMS Filter Coefficients/Lms apotelesmata:');
disp(w);
disp('Actual System Coefficients/Pragmatika apotelesmata:');
disp(h);
