%% Simulates data and then fits the model to the data

data = model_simulate([0.4, 5.5]);
f = @(theta)(model_neg_log_likelihood(data, theta));
theta_start = [0, 0];
theta_opt = fminunc(f, theta_start);
theta_opt(2) = exp(theta_opt(2));
