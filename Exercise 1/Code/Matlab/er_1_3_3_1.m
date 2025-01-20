
N = 1000; % Samples/deigmata
x = randn(1, N); % Input signal/sima eisodou
h = [1, -0.4, -4, 0.5]; % Coefficients/Sidelestes
d = filter(h, 1, x); % d(n)
mu = 0.01; % Step size/vima μ

% Filter length/mhkos filtrou
filter_lengths = [3, 5];


learning_curves = zeros(length(filter_lengths), N);

% Loop for each length/epanalipseis gia to kathe mhkos
for idx = 1:length(filter_lengths)
    L = filter_lengths(idx); 
    w = zeros(1, L); 
    mse = zeros(1, N); 

    % Lms implementation/ilopoihsh lms
    for n = L:N
       
        X = x(n:-1:n-L+1); 
        
        % filter output/eksodos filtrou
        y = w * X'; 
        
       % Error of filter output signal compared to the desired signal/sfalma
    % eksodou
        e = d(n) - y; 
        
        %  New filter coefficients/Enimerosi sideleston me kathe epanalipsi
        w = w + mu * e * X; 
        
        
        mse(n) = e^2;
    end

   
    learning_curves(idx, :) = cumsum(mse) ./ (1:N);
end


figure;
for idx = 1:length(filter_lengths)
    plot(1:N, learning_curves(idx, :), 'LineWidth', 1.5); hold on;
end
title('Learning Curves');
xlabel('Iterations');
ylabel('Mean Squared Error (MSE)');
legend(arrayfun(@(L) sprintf('L = %d', L), filter_lengths, 'UniformOutput', false));
grid on;
