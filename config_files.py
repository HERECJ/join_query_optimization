# Policy configs
from testt_model import EPS_START


POLICY_CONFIG = {
    # Exploration
    "EPS_START" : 0.9,
    "EPS_END" : 0.05,
    "EPS_DECAY" : 200,
    "TARGET_UPDATE" : 20,
    "CAPACITY" : 10000,
    'learning_rate' : 0.001,
    'optim' : 'adam',
    "GAMMA" : 0.000,
    "Test" : 100,
    'loss': 'l1'
}

# Model configs
MODEL_CONFIG = {
    "emb_dim" : 32,
    "emb_bias": False,
    "emb_init_std" : 0.01,
    "graph_dim" : 32,
    "activation" : "RELU",
    "dropout" : 0.5,
    "graph_bias" : True,
    "graph_pooling":'SUM',
}
