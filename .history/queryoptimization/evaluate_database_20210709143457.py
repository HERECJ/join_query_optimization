import ray
from ray.rllib.evaluation.metrics import collect_episodes
from ray.rllib.utils.test_utils import check_learning_achieved

def custom_eval_function(trainer, eval_workers):
    