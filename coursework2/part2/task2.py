import numpy as np
from part2.part2helpers import simulate_trials

# Model parameters
learning_rate = 0.35
inverse_temperature = 5.5

stimulus_values, choices, reward_received = simulate_trials(learning_rate=learning_rate, inverse_temperature=inverse_temperature)
