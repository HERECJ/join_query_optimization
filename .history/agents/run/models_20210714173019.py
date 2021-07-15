"""
        CustomModels:
            Define the NN including the action masking layer


̣__author__ = "Jonas Heitz"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import gym
# from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from gym.spaces import Discrete, Box, Dict

# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# from ray.rllib.models.misc import normc_initializer
from ray.tune.registry import register_env
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX

import torch
import torch.nn as nn


import ray
from ray import tune
from ray.tune import grid_search
import json


class CustomModel(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        # nn.Module.__init__()
        # self.fcnet = nn.Linear(3024, num_outputs)
        # action_space.n  the shape of the action space
        combined_space = Box(0, 1, shape=(4004,), dtype=np.float32)
        self.fcnet = FullyConnectedNetwork(combined_space, action_space, num_outputs, model_config, name)
        self.cnt_for = 0
        print("======Finish Initialize=====")
        
    def forward(self, input_dict, state, seq_lens):
        print("((((((  Start Model Forward  ))))), ", self.cnt_for)
        self.cnt_for += 1
        action_mask = input_dict["obs"]["action_mask"]
        if action_mask.sum() < 1:
            # import pdb; pdb.set_trace()
            print("********* All Invalid Actions *********")
        input_dict["obs"] = input_dict["obs"]["db"]
        output,state = self.fcnet(input_dict)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        masked_logits = inf_mask + output
        return masked_logits, state



    


# class CustomModel(Model):
#     def _build_layers_v2(self, input_dict, num_outputs, options):
#         action_mask = input_dict["obs"]["action_mask"]
#         self.obs_space = Box(0, 1, shape=(3024,), dtype=np.float32)  # 28*108
#         input_dict["obs"] = input_dict["obs"]["db"]


#         self.fcnet = FullyConnectedNetwork(input_dict, self.obs_space,
#                                            self.action_space, num_outputs,
#                                            options)

#         label = "fc_out2"

#         output = slim.fully_connected(
#             self.fcnet.last_layer,
#             num_outputs,
#             weights_initializer=normc_initializer(0.01),
#             activation_fn=None,
#             scope=label)

#         inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)



#         masked_logits = inf_mask + output
#         return masked_logits, self.fcnet.last_layer,

# class CustomModel2(Model):
#     def _build_layers_v2(self, input_dict, num_outputs, options):
#         print(options)
#         print(num_outputs)
#         print(input_dict)
#         action_mask = input_dict["obs"]["action_mask"]
#         self.obs_space = Box(0, 1, shape=(3024,), dtype=np.float32) #28*108
#         input_dict["obs"] = input_dict["obs"]["db"]

#         options["fcnet_hiddens"] = [num_outputs * 2 * 2 * 2, num_outputs * 2]

#         self.fcnet = FullyConnectedNetwork(input_dict, self.obs_space,
#                                            self.action_space, num_outputs,
#                                             options)
#         last_layer = self.fcnet.last_layer
#         label = "fc_out2"
#         output = slim.fully_connected(
#             last_layer,
#             num_outputs,
#             weights_initializer=normc_initializer(0.01),
#             activation_fn=None,
#             scope=label)

#         inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

#         masked_logits = inf_mask + output
#         return masked_logits, last_layer,
