
N = 1000; % Samples/deigmata
x = randn(1, N); % Input signal/sima eisodou
L = 4; % 4 coefficients/4 sidelestes
mu = 0.01; % Step size


mse_smooth = zeros(1, N);
mse_abrupt = zeros(1, N);


weights_smooth = zeros(N, L); % For smooth variation
weights_abrupt = zeros(N, L); % For abrupt variation

% Smooth Variation/Omalh metavolh
b_smooth = 1 ./ (1 + exp(-0.02 * (1:N))); % b(n) 
h_smooth = zeros(L, N);
for n = 1:N
    h_smooth(:, n) = b_smooth(n) * [1; -0.4; -4; 0.5]; 
end

% d(n)
d_smooth = zeros(1, N);
for n = L:N
    d_smooth(n) = h_smooth(:, n)' * flip(x(n-L+1:n))'; 
end

% Lms implementation/ilopoihsh lms
w = zeros(1, L); 
for n = L:N
    X = x(n:-1:n-L+1); 
    y = w * X'; % filter output/eksodos filtrou
    e = d_smooth(n) - y; % Error of filter output signal compared to the desired signal/sfalma eksodou
    w = w + mu * e * X; % New filter coefficients/Enimerosi sideleston me kathe epanalipsi
    weights_smooth(n, :) = w; 
    mse_smooth(n) = e^2; 
end

% Abrupt Variation/akariaia metavolh
b_abrupt = [ones(1, 500) * 100, zeros(1, 500)]; % b(n) 
h_abrupt = zeros(L, N);
for n = 1:N
    h_abrupt(:, n) = b_abrupt(n) * [1; -0.4; -4; 0.5]; 
end

% d(n)
d_abrupt = zeros(1, N);
for n = L:N
    d_abrupt(n) = h_abrupt(:, n)' * flip(x(n-L+1:n))'; % Transpose flip result
end

% Lms implementation/ilopoihsh lms
w = zeros(1, L); 
for n = L:N
    X = x(n:-1:n-L+1); 
    y = w * X'; % filter output/eksodos filtrou
    e = d_abrupt(n) - y; % Error of filter output signal compared to the desired signal/sfalma eksodou
    w = w + mu * e * X; % New filter coefficients/Enimerosi sideleston me kathe epanalipsi
    weights_abrupt(n, :) = w; 
    mse_abrupt(n) = e^2; 
end

% Normalize MSE for comparison
mse_smooth = mse_smooth / max(mse_smooth); 
mse_abrupt = mse_abrupt / max(mse_abrupt); 


figure;
plot(1:N, cumsum(mse_smooth) ./ (1:N), 'b', 'LineWidth', 1.5); hold on;
plot(1:N, cumsum(mse_abrupt) ./ (1:N), 'r', 'LineWidth', 1.5);
title('Learning Curves for LMS Algorithm with Time-Varying Impulse Responses');
xlabel('Iterations');
ylabel('Normalized Mean Squared Error (MSE)');
legend('Smooth Variation (2.1)', 'Abrupt Variation (2.2)');
grid on;


figure;
subplot(2, 1, 1);
plot(weights_smooth);
title('Weight Evolution for Smooth Variation (2.1)');
xlabel('Iterations');
ylabel('Weight Values');
legend('w_0', 'w_1', 'w_2', 'w_3');
grid on;

subplot(2, 1, 2);
plot(weights_abrupt);
title('Weight Evolution for Abrupt Variation (2.2)');
xlabel('Iterations');
ylabel('Weight Values');
legend('w_0', 'w_1', 'w_2', 'w_3');
grid on;


disp('Final Weights for Smooth Variation:');
disp(weights_smooth(end, :));

disp('Final Weights for Abrupt Variation:');
disp(weights_abrupt(end, :));
