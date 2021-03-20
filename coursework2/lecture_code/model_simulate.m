function data = model_simulate(theta)
    eps = theta(1); rho = theta(2); V = [0, 0];
    data.choices = []; data.rewards = [];
    for t = 1:100
        p_choose_B = 1 / (1 + exp(-(V(2) - V(1))));
        c = 1 + (p_choose_B > rand);
        if c==1, r = (0.8 > rand); end
        if c==2, r = (0.2 > rand); end
        V(c) = V(c) + eps * (rho * r - V(c));
        data.choices(t) = c; data.rewards(t) = r;
    end
end
