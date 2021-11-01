import numpy as np
import torch
from torch._C import dtype
# import torch

from env.pg_tree_env import Train_Join_Tree, Test_Join_Tree
class Env():
    def __init__(self, config):
        file_path = config['file_name']
        print(config['process'])
        if config['process'].lower() in ['train']:
            self.wrapped = Train_Join_Tree(file_path, config['_tree_flag_'])
        elif config['process'].lower() in ['eval']:
            self.wrapped = Test_Join_Tree(file_path, config['_tree_flag_'])
        else:
            raise ValueError('Invalid Input')
        
    
    def update_avail_actions(self):
        self.validActions = self.wrapped.getValidActions()
        self.action_mask = torch.zeros(len(self.wrapped.action_list))
        for i in self.validActions:
            self.action_mask[i] = 1

    def reset(self):
        obs = self.wrapped.reset()
        self.update_avail_actions()
        obs = torch.tensor(obs, dtype=torch.float)
        return {
            "action_mask" : self.action_mask,
            "db" : obs
        }, self.wrapped.query.table_num
        # return self.action_mask, obs, self.wrapped.query.table_num
    
    def step(self, action):
        if action not in self.validActions:
            print("Invalid Actions!!! Failure steps.")
        
        obs, reward, done, tree_list, join_num = self.wrapped.step(action)
        self.update_avail_actions()
        obs = torch.tensor(obs, dtype=torch.float)
        obs = {
            "action_mask" : self.action_mask,
            "db" : obs
        }
        # return self.action_mask, obs, reward, done, tree_list, join_num
        return obs, reward, done, tree_list, join_num

class Tree_Env(Env):
    def __init__(self, config):
        super().__init__(config)
    
    def reset(self):
        _ = self.wrapped.reset()
        self.update_avail_actions()
        
        link_mtx = self.wrapped.query.link_mtx
        self.link_mtx = torch.tensor(link_mtx, dtype=torch.float)
        return {
            "action_mask" : self.action_mask,
            "query_lst" : None,
            "link_mtx" : self.link_mtx
        }, self.wrapped.query.table_num
    
    def step(self, action):
        if action not in self.validActions:
            print("Invalid Actions!!! Failure steps.")
        
        _, reward, done, tree_list, join_num = self.wrapped.step(action)
        self.update_avail_actions()
        obs = {
            "action_mask" : self.action_mask,
            "query_lst" : tree_list,
            "link_mtx" : self.link_mtx
        }
        return obs, reward, done, [], join_num
