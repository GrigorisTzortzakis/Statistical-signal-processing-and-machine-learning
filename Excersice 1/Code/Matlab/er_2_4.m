
N = 1000; % Samples/deigmata
L = 4; % 4 coefficients/4 sidelestes
mu = 0.01; % Step size
num_realizations = 20; % 20 different inputs/20 ilopoihseis


all_squared_errors = zeros(num_realizations, N);

% Lms implementation/ilopoihsh lms
for realization = 1:num_realizations
    % New Input signal each time for 20 times/Neo sima eisodou 20 fores
    x = randn(1, N); 

    % Smooth Variation/Omalh metavolh
    b_smooth = 1 ./ (1 + exp(-0.02 * (1:N))); % b(n) 
    h_smooth = zeros(L, N);
    for n = 1:N
        h_smooth(:, n) = b_smooth(n) * [1; -0.4; -4; 0.5]; 
    end

    %  d(n)
    d_smooth = zeros(1, N);
    for n = L:N
        d_smooth(n) = h_smooth(:, n)' * flip(x(n-L+1:n))'; 
    end

   
    w = zeros(1, L); 
    squared_error = zeros(1, N); 
    for n = L:N
        X = x(n:-1:n-L+1); 
        y = w * X'; % filter output/eksodos filtrou
        e = d_smooth(n) - y; % Error of filter output signal compared to the desired signal/sfalma eksodou
        w = w + mu * e * X; % New filter coefficients/Enimerosi sideleston me kathe epanalipsi
        squared_error(n) = e^2; 
    end

    
    all_squared_errors(realization, :) = squared_error;
end

% Average squared error/mesos oros sfalmatos
avg_squared_error = mean(all_squared_errors, 1);


figure;
plot(1:N, avg_squared_error, 'b', 'LineWidth', 1.5);
title('Average Squared Error over 20 Realizations');
xlabel('Iterations');
ylabel('Average Squared Error (e^2)');
grid on;
