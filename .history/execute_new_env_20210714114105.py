"""
    Executes series of experiments

̣__author__ = "Jonas Heitz"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models import ModelCatalog


from ray.tune.registry import register_env

import ray
from ray import tune

from agents.run.models import CustomModel
from agents.run.masking_env import CrossVal
from agents.run.configs_new_env import SIMPLE_CONFIG


if __name__ == "__main__":

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)



    # register_env("CM1-postgres-card-job-masking-v0", lambda config : CrossVal(config))


    iteration = range(0,6)
    cross = range(0,1)

    model = "my_model"

    CONFIGS = [SIMPLE_CONFIG]
    

    for cfg in CONFIGS:
        for j in cross: #crossval
            env_name = 
            for i in iteration: #range(0, 5):
                print(i)
                tune.run(
                    "DQN",
                    checkpoint_at_end=True,
                    stop={
                        "timesteps_total": 200000,
                        "training_iteration": 32,
                    },
                    config=dict({
                        "framework": "torch",
                        # "num_gpus": 1,
                        # "num_gpus_per_worker": 1,
                        "num_workers": 4,
                        "env": "CM1-postgres-card-job-masking-v"+str(j),
                        "model": {
                            "custom_model": model,
                        },
                    }, **cfg),

                )
        model = "my_model_deep"

    # cfg = PPO_CONFIG
    # for j in cross: #crossval
    #    for i in iteration: #range(0, 5):
    #         print(i)
    #         tune.run(
    #             "PPO",
    #             checkpoint_at_end=True,
    #             resources_per_trial={"cpu": 4, "gpu": 0},
    #             stop={
    #                 "timesteps_total": 20000*2*5,
    #             },
    #             config=dict({
    #                 "env": "CM1-postgres-card-job-masking-v"+str(j),
    #                 "model": {
    #                     "custom_model": "my_model",
    #                 },
    #             }, **cfg),)
