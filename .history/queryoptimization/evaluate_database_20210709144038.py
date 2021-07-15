import ray
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.utils.test_utils import check_learning_achieved

def custom_eval_function(trainer, eval_workers):
    worker_ = eval_workers.remote_workers()
    for i in range(3):
        print("***********Custom evaluation round ", i)
        