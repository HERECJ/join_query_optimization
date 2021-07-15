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
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)



    # register_env("CM1-postgres-card-job-masking-v0", lambda config : CrossVal(config))


    iteration = range(0,6)
    cross = range(0,2)

    model = "my_model"

    CONFIGS = [SIMPLE_CONFIG]
    

    for cfg in CONFIGS:
        for j in cross: #crossval
            env_name = "PG-Join-Cross-Masking-v{}".format(j)
            register_env(env_name, lambda config: CrossVal(config))
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
                        # "log_level": "DEBUG",
                        "num_gpus" : 1,
                        "framework": "torch",
                        "env": env_name,
                        "env_config": {
                            "fold_idx": j,
                            "process":0,
                        },
                        "evaluation_config": {
                            "env_config": {
                                "fold_idx": j,
                                "process":1,
                            }
                        },
                        "model": {
                            "custom_model": model,
                            # "custom_model_config": {}
                        },
                    }, **cfg),

                )