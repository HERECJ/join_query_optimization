"""
    Config dict for vanilla DQN(SIMPLE_CONFIG), DDQN with Priority Replay(DOUBLE_PRIO) and PPO(PPO_CONFIG)

̣__author__ = "Jonas Heitz"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

from gym.spaces import Discrete, Box, Dict

from ray.tune.registry import register_env

import ray
from ray import tune
from ray.tune import grid_search
import json
from queryoptimization.evaluate_database import custom_eval_function


OPTIMIZER_SHARED_CONFIGS = [
    "buffer_size", "prioritized_replay", "prioritized_replay_alpha", "prioritized_replay_beta",
    "final_prioritized_replay_beta",
    "prioritized_replay_eps",
    "train_batch_size",
    "learning_starts"
]

eval_fn = custom_eval_function

SIMPLE_CONFIG = {
    # === Model ===
    # Number of atoms for representing the distribution of return. When
    # this is greater than 1, distributional Q-learning is used.
    # the discrete supports are bounded by v_min and v_max
    "num_atoms": 0,
    "v_min": -10.0,
    "v_max": 10.0,
    # Whether to use noisy network
    "noisy": False,
    # control the initial value of noisy nets
    "sigma0": 0.5,
    # Whether to use dueling dqn
    "dueling": False,
    # Whether to use double dqn
    "double_q": False,
    # Postprocess model outputs with these hidden layers to compute the
    # state and action values. See also the model config in catalog.py.
    "hiddens": [],
    # N-step Q learning
    "n_step": 1,

    # === Evaluation ===
    # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    # "evaluation_interval": None,
    "evaluation_interval" : None,
    # Number of episodes to run per evaluation period.
    "evaluation_num_episodes": 200,
    # "custom_eval_function": eval_fn,

    # === Exploration ===
    # Max num timesteps for annealing schedules. Exploration is annealed from
    # 1.0 to exploration_fraction over this number of timesteps scaled by
    # exploration_fraction
    # "schedule_max_timesteps": 100000,
    # Number of env steps to optimize for before returning
    "timesteps_per_iteration": 1000,
    # Fraction of entire training period over which the exploration rate is
    # annealed
    # "exploration_fraction": 0.1,
    # Final value of random action probability
    # "exploration_final_eps": 0.02,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 500,
    # Use softmax for sampling actions. Required for off policy estimation.
    # "soft_q": False,
    # Softmax temperature. Q values are divided by this value prior to softmax.
    # Softmax approaches argmax as the temperature drops to zero.
    # "softmax_temp": 1.0,
    # If True parameter space noise will be used for exploration
    # See https://blog.openai.com/better-exploration-with-parameter-noise/
    # "parameter_noise": False,
    "explore": True,
    

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": 500,
    # If True prioritized replay buffer will be used.
    "prioritized_replay": False,#True,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Fraction of entire training period over which the beta parameter is
    # annealed
    # "beta_annealing_fraction": 0.2,
    # Final value of beta
    "final_prioritized_replay_beta": 0.4,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,
    # Whether to LZ4 compress observations
    "compress_observations": False,

    # === Optimization ===
    # Learning rate for adam optimizer
    "lr": 5e-4,
    # Adam epsilon hyper parameter
    "adam_epsilon": 1e-8,
    # If not None, clip gradients during optimization at this value
    "grad_clip": None,#40,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1000,
    # Update the replay buffer with this many samples at once. Note that
    # this setting applies per-worker if num_workers > 1.
    # "sample_batch_size": 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 32,

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Optimizer class to use.
    # "optimizer_class": "SyncReplayOptimizer",
    # Whether to use a distribution of epsilons across workers for exploration.
    # "per_worker_exploration": False,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,

}

# SIMPLE_CONFIG = {
#      # === Model ===
#     # Number of atoms for representing the distribution of return. When
#     # this is greater than 1, distributional Q-learning is used.
#     # the discrete supports are bounded by v_min and v_max
#     "num_atoms": 1,
#     "v_min": -10.0,
#     "v_max": 10.0,
#     # Whether to use noisy network
#     "noisy": False,
#     # control the initial value of noisy nets
#     "sigma0": 0.5,
#     # Whether to use dueling dqn
#     "dueling": True,
#     # Dense-layer setup for each the advantage branch and the value branch
#     # in a dueling architecture.
#     "hiddens": [],
#     # Whether to use double dqn
#     "double_q": True,
#     # N-step Q learning
#     "n_step": 1,

#     # === Exploration Settings ===
#     "exploration_config": {
#         # The Exploration class to use.
#         "type": "EpsilonGreedy",
#         # Config for the Exploration class' constructor:
#         "initial_epsilon": 1.0,
#         "final_epsilon": 0.02,
#         "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.

#         # For soft_q, use:
#         # "exploration_config" = {
#         #   "type": "SoftQ"
#         #   "temperature": [float, e.g. 1.0]
#         # }
#     },
#     # Switch to greedy actions in evaluation workers.
#     "evaluation_config": {
#         "explore": False,
#     },

#     # Minimum env steps to optimize for per train call. This value does
#     # not affect learning, only the length of iterations.
#     "timesteps_per_iteration": 1000,
#     # Update the target network every `target_network_update_freq` steps.
#     "target_network_update_freq": 500,


#     # === Replay buffer ===
#     # Size of the replay buffer. Note that if async_updates is set, then
#     # each worker will have a replay buffer of this size.
#     "buffer_size": 50000,
#     # The number of contiguous environment steps to replay at once. This may
#     # be set to greater than 1 to support recurrent models.
#     "replay_sequence_length": 1,
#     # If True prioritized replay buffer will be used.
#     "prioritized_replay": True,
#     # Alpha parameter for prioritized replay buffer.
#     "prioritized_replay_alpha": 0.6,
#     # Beta parameter for sampling from prioritized replay buffer.
#     "prioritized_replay_beta": 0.4,
#     # Final value of beta (by default, we use constant beta=0.4).
#     "final_prioritized_replay_beta": 0.4,
#     # Time steps over which the beta parameter is annealed.
#     "prioritized_replay_beta_annealing_timesteps": 20000,
#     # Epsilon to add to the TD errors when updating priorities.
#     "prioritized_replay_eps": 1e-6,

#     # Whether to LZ4 compress observations
#     "compress_observations": False,
#     # Callback to run before learning on a multi-agent batch of experiences.
#     "before_learn_on_batch": None,
#     # If set, this will fix the ratio of replayed from a buffer and learned on
#     # timesteps to sampled from an environment and stored in the replay buffer
#     # timesteps. Otherwise, the replay will proceed at the native ratio
#     # determined by (train_batch_size / rollout_fragment_length).
#     "training_intensity": None,

#     # === Optimization ===
#     # Learning rate for adam optimizer
#     "lr": 5e-4,
#     # Learning rate schedule
#     "lr_schedule": None,
#     # Adam epsilon hyper parameter
#     "adam_epsilon": 1e-8,
#     # If not None, clip gradients during optimization at this value
#     "grad_clip": 40,
#     # How many steps of the model to sample before learning starts.
#     "learning_starts": 1000,
#     # Update the replay buffer with this many samples at once. Note that
#     # this setting applies per-worker if num_workers > 1.
#     "rollout_fragment_length": 4,
#     # Size of a batch sampled from replay buffer for training. Note that
#     # if async_updates is set, then each worker returns gradients for a
#     # batch of this size.
#     "train_batch_size": 32,

#     # === Parallelism ===
#     # Number of workers for collecting samples with. This only makes sense
#     # to increase if your environment is particularly slow to sample, or if
#     # you"re using the Async or Ape-X optimizers.
#     "num_workers": 0,
#     # Whether to compute priorities on workers.
#     "worker_side_prioritization": False,
#     # Prevent iterations from going lower than this time span
#     "min_iter_time_s": 1,
# }