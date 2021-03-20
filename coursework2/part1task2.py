import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from part1helpers import simulate_many

np.random.seed(0)
drift_rate_variance = 0.04  # s
time_step = 0.001  # dt
max_steps = 2000

drift_rate_mean_range = np.arange(-0.10, 0.10, 0.02)
boundary_separation_range = np.arange(0.05, 0.50, 0.05)

combinations = np.array(np.meshgrid(drift_rate_mean_range, boundary_separation_range)).T.reshape(-1, 2)

accuracies = []
average_response_times = []

for drift_rate_mean, boundary_separation in combinations:
    results = simulate_many(num_simulations=100,
                            time_step=time_step,
                            starting_point=boundary_separation/2,
                            drift_rate_mean=drift_rate_mean,
                            drift_rate_variance=drift_rate_variance,
                            boundary_separation=boundary_separation,
                            max_steps=max_steps
                            )

    accuracy = sum(hypothesis == "h_pos" for hypothesis, W_t in results)/len(results)
    average_response_time = sum(len(W_t) for hypothesis, W_t in results)/len(results)

    accuracies.append(accuracy)
    average_response_times.append(average_response_time)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(combinations[:, 0], combinations[:, 1], accuracies)
ax.plot_trisurf(combinations[:, 0], combinations[:, 1], accuracies, linewidth=0, antialiased=False)
ax.set_xlabel("Drift rate mean")
ax.set_ylabel("Boundary separation")
ax.set_zlabel("Accuracy")
plt.show()
