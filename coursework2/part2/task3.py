import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from part2.part2helpers import simulate_trials

num_simulations = 100

learning_rate_range = np.arange(0, 1.1, 0.1)
inverse_temperature_range = np.arange(0, 16, 1)

learning_rates, inverse_temperatures = np.meshgrid(learning_rate_range, inverse_temperature_range)

average_received_rewards = np.zeros(shape=(len(inverse_temperature_range), len(learning_rate_range)))
for i in range(len(inverse_temperature_range)):
    for j in range(len(learning_rate_range)):
        num_received_rewards = []
        for _ in range(num_simulations):
            stimulus_values, choices, reward_received = simulate_trials(learning_rate=learning_rates[i,j],
                                                                        inverse_temperature=inverse_temperatures[i,j])
            num_received_rewards.append(reward_received.sum())

        average_received_rewards[i,j] = (sum(num_received_rewards)/num_simulations)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# average_received_rewards = np.array(average_received_rewards).T.reshape(combinations[0].shape)

# ax.scatter(combinations[0], combinations[1], average_received_rewards)
myplot = ax.plot_surface(learning_rates, inverse_temperatures, average_received_rewards, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
ax.set_xlabel("Learning rate")
ax.set_ylabel("Inverse Temperature")
ax.set_zlabel("Average number of rewards received")
fig.colorbar(myplot, shrink=0.4, aspect=8)
plt.show()
