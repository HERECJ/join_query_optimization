# Policy configs

POLICY_CONFIG = {
    # Exploration
    "EPS_START" : 0.9,
    "EPS_END" : 0.05,
    "EPS_DECAY" : 200*10,
    "TARGET_UPDATE" : 20,
    "CAPACITY" : 50000,
    'optim' : 'adam',
    "Test" : 100,
    'loss': 'mse',
    "prioritized" : False,
    "Lambda1" : 0.000,
    "Lambda2" : 0.000,
    "Normalization" : 'softmax',
}

# Model configs
MODEL_CONFIG = {

    "emb_bias": False,
    "emb_init_std" : 0.01,
    "activation" : "RELU",
    "dropout" : 0.5,
    "graph_bias" : True,
    "graph_pooling":'mean',
}
