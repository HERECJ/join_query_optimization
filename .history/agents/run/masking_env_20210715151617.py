from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import gym

from gym.spaces import Discrete, Box, Dict

from ray.tune.registry import register_env

import ray
from ray import tune
from ray.tune import grid_search



class CrossVal(gym.Env):
    def __init__(self, config):
        fold = config['fold_idx']
        # proc = config['process'] # 0 : Train 1: Eval
        if config['process'] == 0:
            proc = 'Train'
        elif config['process'] == 1:
            proc = 'Eval'
        else:
            raise ValueError(
                "Not supported process, Please check the config file!!!")
        print("Env : ", proc)
        env_name = 'PGsql-{}-Join-Job-v{}'.format(proc, fold)
        self.wrapped = gym.make(env_name)
        self.action_space = self.wrapped.action_space
        self.reward_range = self.wrapped.reward_range

        self.update_avail_actions()
        self.observation_space = Dict({})
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.wrapped.action_space.n, )),
            "db": self.wrapped.observation_space,
        })
        self.counter = 0
        self.scounter = 0

    def update_avail_actions(self):
        self.validActions = self.wrapped.getValidActions()
        self.action_mask = np.array([0.] * self.action_space.n)
        for i in self.validActions:
            self.action_mask[i] = 1

    def reset(self):
        self.counter += 1
        self.scounter = 0
        obs = self.wrapped.reset()
        self.update_avail_actions()
        return {
            "action_mask": self.action_mask,
            "db": obs,
        }

    def step(self, action):
        self.scounter += 1
        if action not in self.validActions:
            #self.failurecounter+=1
            print("failure steps:")

        orig_obs, rew, done, info = self.wrapped.step(action)
        # import pdb; pdb.set_trace()
        self.update_avail_actions()
        obs = {
            "action_mask": self.action_mask,
            "db": orig_obs,
        }
        return obs, rew, done, info