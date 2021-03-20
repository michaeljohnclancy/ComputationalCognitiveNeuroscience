function nll = model_neg_log_likelihood(data, theta)
    eps = theta(1); rho = theta(2); V = [0, 0];
    rho = exp(rho); num_trials = numel(data.choices);
    choice_probabilities = nan(num_trials, 1);
    for t = 1:num_trials
        c = data.choices(t); c_alt = 1 + (c==1);
        p_choose_c = 1 / (1 + exp(-(V(c) - V(c_alt))));
        choice_probabilities(t) = p_choose_c;
        V(c) = V(c) + eps * (rho * data.rewards(t) - V(c));
    end
    nll = -sum(log(choice_probabilities));
end
