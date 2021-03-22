import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy.optimize import minimize

def get_new_stimulus_value(current_stimulus_value, learning_rate, reward_received: bool, reward_sensitivity: float):
    return current_stimulus_value + (learning_rate * ((int(reward_received) * reward_sensitivity) - current_stimulus_value))


def get_probability_of_stimulus_a(stimulus_values: Dict[int, float]):
    return np.exp(stimulus_values[1]) \
           / (np.exp(stimulus_values[1]) + np.exp(stimulus_values[2]))


def simulate_single_trial(stimulus_values: Dict[int, float], reward_probabilities: Dict[int, float], learning_rate: float, reward_sensitivity: float):
    prob_a = get_probability_of_stimulus_a(stimulus_values)
    choice = np.random.choice([1,2], p=(prob_a, 1-prob_a))

    reward_received = np.random.choice([True, False],
                                       p=(reward_probabilities[choice],
                                          1-reward_probabilities[choice]))

    stimulus_values[choice] = get_new_stimulus_value(stimulus_values[choice],
                                                     learning_rate=learning_rate,
                                                     reward_received=reward_received,
                                                     reward_sensitivity=reward_sensitivity)

    return stimulus_values, choice, reward_received


def simulate_trials(learning_rate: float, reward_sensitivity: float, num_48_trial_batches=5, V_A=0, V_B=0) -> Tuple[pd.DataFrame, np.array, np.array]:
    reward_probabilities_1 = {1: 0.4, 2: 0.85}
    reward_probabilities_2 = {1: 0.65, 2: 0.30}

    stimulus_values = {}
    stimulus_values[1] = V_A
    stimulus_values[2] = V_B

    all_choices = []
    all_reward_received = []
    all_stimulus_values = []
    for i in range(num_48_trial_batches):
        for _ in range(24):
            stimulus_values, choice, reward_received = simulate_single_trial(stimulus_values, reward_probabilities_1, learning_rate, reward_sensitivity)
            all_choices.append(choice)
            all_reward_received.append(reward_received)
            all_stimulus_values.append(stimulus_values.copy())

        for _ in range(24):
            stimulus_values, choice, reward_received = simulate_single_trial(stimulus_values, reward_probabilities_2, learning_rate, reward_sensitivity)
            all_choices.append(choice)
            all_reward_received.append(reward_received)
            all_stimulus_values.append(stimulus_values.copy())

    return pd.DataFrame(all_stimulus_values, columns=['A', 'B']), np.array(all_choices), np.array(all_reward_received)

def get_negative_log_likelihood(parameters: np.ndarray, choices: pd.Series, rewards_received: pd.Series):
    learning_rate = parameters[0]
    reward_sensitivity = parameters[1]
    num_trials = choices.shape[0]
    try:
        V = [parameters[2], parameters[3]]
    except:
        V = [0,0]
    choice_probabilities = np.empty(shape=num_trials)
    for i in range(num_trials):
        choice_index = choices[i] - 1
        choice_probabilities[i] = np.exp(V[choice_index]) \
                                  / (np.exp(V[choice_index]) + np.exp(V[int(not choice_index)]))
        V[choice_index] += (learning_rate * ((rewards_received[i] * reward_sensitivity) - V[choice_index]))
    return -np.sum(np.log(choice_probabilities))

def get_individual_parameter_estimates_model2(choices, rewards, initial_parameters=np.array([0.5, 5])):
    optimal_params_list = []
    for i in range(choices.shape[0]):
        optimal_params_list.append(
            minimize(get_negative_log_likelihood, method="Nelder-Mead", x0=initial_parameters, args=(choices.iloc[i], rewards.iloc[i])).x)

    return pd.DataFrame(optimal_params_list, columns=['learning_rate', 'inverse_temperature'])

def get_individual_parameter_estimates_model3(choices, rewards, initial_parameters=np.array([0.5, 5, 0, 0])):
    optimal_params_list = []
    for i in range(choices.shape[0]):
        optimal_params_list.append(
            minimize(get_negative_log_likelihood, method="Nelder-Mead", x0=initial_parameters, args=(choices.iloc[i], rewards.iloc[i])).x)

    return pd.DataFrame(optimal_params_list, columns=['learning_rate', 'inverse_temperature', 'V_A', 'V_B'])

def get_total_negative_log_likelihood(parameters: pd.DataFrame, choices=pd.read_csv('data/choices.csv', header=None), rewards = pd.read_csv('data/rewards.csv', header=None)):
    return sum(get_negative_log_likelihood(parameters=parameters[i], choices=choices.iloc[i], rewards_received=rewards.iloc[i]) for i in range(choices.shape[0]))

def get_aic(nll, p):
    return (2*nll) + (2*p)

def get_bic(nll, p, n):
    return (2*nll) + (p * np.log(n))