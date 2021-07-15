import ray
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.utils.test_utils import check_learning_achieved

def custom_eval_function(trainer, eval_workers):
    import pdb; pdb.set_trace()
    worker_ = eval_workers.remote_workers()
    for i in range(3):
        print("***********Custom evaluation round ", i)
        ray.get([w.sample.remote() for w in worker_])
    
    episodes, _ = collect_episodes(remote_workers=worker_, timeout_seconds=99999)
    metrics = summarize_episodes(episodes)
    import pdb; pdb.set_trace()
    return metrics