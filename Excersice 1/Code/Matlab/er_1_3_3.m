
N = 1000; % Samples/deigmata
x = randn(1, N); % Input signal/sima eisodou
h = [1, -0.4, -4, 0.5]; % Coefficients/Sidelestes
d = filter(h, 1, x); % d(n)

L = 4; % 4 coefficients/4 sidelestes

% Computing the autocorrelation matrix Rx/Ipologismos Rx
X_full = zeros(L, N - L + 1); 
for i = 1:L
    X_full(i, :) = x(i:N - L + i); 
end
R_x = (X_full * X_full') / (N - L); 

% Maximum step size μ/ Megisti timh tou μ
lambda_max = max(eig(R_x)); 
mu_max = 2 / lambda_max; 

% Step sizes required/thetoume to μ pou zitite
mu_values = [0.001, 0.01, 0.1, 0.5] * mu_max;


coeff_evolution = zeros(length(mu_values), N, L);

% Loop for all step sizes/kanoume loop oste na treksoun ola ta μ
final_coefficients = zeros(length(mu_values), L); 
for i = 1:length(mu_values)
    mu = mu_values(i); 
    w = zeros(1, L); 
    
   
    coeffs_over_time = zeros(N, L);
    
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
        
        
        coeffs_over_time(n, :) = w;
    end
    
    % Save coefficient evolution/ekseliksei sideleston
    coeff_evolution(i, :, :) = coeffs_over_time;
    
    % Save coefficients/sidelestes
    final_coefficients(i, :) = w;
end


for i = 1:length(mu_values)
    mu = mu_values(i);
    disp(['Step size μ: ', num2str(mu)]);
    disp('Final Coefficients:');
    disp(final_coefficients(i, :)); 
    disp('------------------------------------');
end


figure;
for i = 1:length(mu_values)
    subplot(2, 2, i);
    plot(squeeze(coeff_evolution(i, :, :)));
    title(['Coefficient Evolution for μ = ', num2str(mu_values(i))]);
    xlabel('Iterations');
    ylabel('Coefficient Value');
    legend('w1', 'w2', 'w3', 'w4');
end
