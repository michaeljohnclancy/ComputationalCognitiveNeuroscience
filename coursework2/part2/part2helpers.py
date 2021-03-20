import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.optimize import minimize


def get_new_stimulus_value(current_stimulus_value, learning_rate, reward_received: bool):
    return current_stimulus_value + (learning_rate * (int(reward_received) - current_stimulus_value))


def get_probability_of_stimulus_a(stimulus_values: Dict[str, float], inverse_temperature: float):
    return np.exp(inverse_temperature * stimulus_values['A']) \
           / (np.exp(inverse_temperature * stimulus_values['A']) + np.exp(inverse_temperature * stimulus_values['B']))


def simulate_single_trial(stimulus_values: Dict[str, float], reward_probabilities: Dict[str, float], learning_rate: float, inverse_temperature: float):
    prob_a = get_probability_of_stimulus_a(stimulus_values, inverse_temperature=inverse_temperature)
    choice = np.random.choice(['A', 'B'], p=(prob_a, 1-prob_a))

    reward_received = np.random.choice([True, False],
                                       p=(reward_probabilities[choice],
                                          1-reward_probabilities[choice]))

    stimulus_values[choice] = get_new_stimulus_value(stimulus_values[choice],
                                                     learning_rate=learning_rate,
                                                     reward_received=reward_received)

    return stimulus_values, choice, reward_received


def simulate_trials(learning_rate: float, inverse_temperature: float) -> Tuple[pd.DataFrame, np.array, np.array]:
    reward_probabilities_1 = {'A': 0.4, 'B': 0.85}
    reward_probabilities_2 = {'A': 0.65, 'B': 0.30}

    stimulus_values = {'A': 0, 'B': 0}

    all_choices = []
    all_reward_received = []
    all_stimulus_values = []
    for i in range(5):
        for _ in range(24):
            stimulus_values, choice, reward_received = simulate_single_trial(stimulus_values, reward_probabilities_1, learning_rate, inverse_temperature)
            all_choices.append(choice)
            all_reward_received.append(reward_received)
            all_stimulus_values.append(stimulus_values.copy())

        for _ in range(24):
            stimulus_values, choice, reward_received = simulate_single_trial(stimulus_values, reward_probabilities_2, learning_rate, inverse_temperature)
            all_choices.append(choice)
            all_reward_received.append(reward_received)
            all_stimulus_values.append(stimulus_values.copy())

    return pd.DataFrame(all_stimulus_values, columns=['A', 'B']), np.array(all_choices), np.array(all_reward_received)

def get_negative_log_likelihood(parameters: np.ndarray, choices: pd.Series, rewards_received: pd.Series):
    learning_rate = parameters[0]
    inverse_temperature = parameters[1]
    num_trials = choices.shape[0]
    V = [0,0]
    choice_probabilities = np.empty(shape=num_trials)
    for i in range(num_trials):
        choice_index = choices[i] - 1
        # choice_probabilities[i] = 1 / (1 + np.exp(-(V[choice_index] - V[not choice_index])))
        choice_probabilities[i] = np.exp(inverse_temperature * V[choice_index]) \
                    / (np.exp(inverse_temperature * V[choice_index]) + np.exp(inverse_temperature * V[int(not choice_index)]))
        V[choice_index] += (learning_rate * (rewards_received[i] - V[choice_index]))
    return -np.sum(np.log(choice_probabilities))

def get_total_negative_log_likelihood(parameters: np.ndarray):
    choices = pd.read_csv('data/choices.csv', header=None)
    rewards = pd.read_csv('data/rewards.csv', header=None)
    return sum(get_negative_log_likelihood(parameters=parameters, choices=choices.iloc[i], rewards_received=rewards.iloc[i]) for i in range(choices.shape[0]))


def get_individual_parameter_estimates(choices, rewards, initial_parameters=np.array([0.5, 5])):
    optimal_params_list = []
    for i in range(choices.shape[0]):
        optimal_params_list.append(
            minimize(get_negative_log_likelihood, method="Nelder-Mead", x0=initial_parameters, args=(choices.iloc[i], rewards.iloc[i])).x)

    return pd.DataFrame(optimal_params_list, columns=['learning_rate', 'inverse_temperature'])
