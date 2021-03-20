import numpy as np


def wiener_diffusion(W_t_old, drift_rate_mean, drift_rate_variance, time_step):
    return (
        W_t_old
        + (drift_rate_mean * time_step)
        + (drift_rate_variance * float(np.random.normal(0, np.sqrt(time_step), 1)))  # type: ignore
    )


def simulate(
    time_step,
    starting_point,
    drift_rate_mean,
    drift_rate_variance,
    boundary_separation,
    max_steps=2000,
):

    W_t = []
    W_t.append(starting_point)

    for i in range(max_steps - 1):
        W_t.append(
            wiener_diffusion(
                W_t_old=W_t[i],
                drift_rate_mean=drift_rate_mean,
                drift_rate_variance=drift_rate_variance,
                time_step=time_step,
            )
        )
        if W_t[i + 1] < 0:
            return "h_neg", W_t
        elif W_t[i + 1] > boundary_separation:
            return "h_pos", W_t

    return "", W_t


def simulate_many(
    num_simulations,
    time_step,
    starting_point,
    drift_rate_mean,
    drift_rate_variance,
    boundary_separation,
    max_steps=2000,
):
    results = []
    for i in range(num_simulations):
        hypothesis, W_t = simulate(
            time_step=time_step,
            starting_point=starting_point,
            drift_rate_mean=drift_rate_mean,
            drift_rate_variance=drift_rate_variance,
            boundary_separation=boundary_separation,
            max_steps=max_steps,
        )

        results.append((hypothesis, W_t))

    return results
