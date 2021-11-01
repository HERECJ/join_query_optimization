import torch
import numpy as np
file_path = 'table_emb_deepwalk/out_deepwalk_16.embeddings'
emb = []
with open(file_path, 'r') as fn:
    for line in fn.readlines():
        row = line.strip().split(' ')
        emb.append(row[1:])
embs = np.asarray(emb)
a = 1