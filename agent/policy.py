import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import math
import random
import numpy as np
from agent.model import DQ_Net, Tree_Net
from torch.optim.lr_scheduler import StepLR

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38

Experience=namedtuple('Experience',('old_obs', 'action_num', 'obs', 'cost', 'done'))

steps_done=0

class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
    def push (self,prio,*args):
        self.memory.append(Experience(*args))
    def sample (self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)


# class PriorityReplayMemory(ReplayMemory):
#     def __init__(self,capacity):
#         super().__init__(capacity=capacity)
#         self.priorities = deque([], maxlen=capacity)
#         self.join = deque([], maxlen=capacity)

#     def push (self,prio,*args):
#         self.memory.append(Experience(*args))
#         self.priorities.append(prio)
#         self.join.append(prio)
#         #print(self.priorities)

#     def sample (self,batch_size, priority_scale=1.0):
#         sample_size = batch_size
#         sample_probs = self.get_probabilities(priority_scale)
#         memory_pool = [ i for i in range(len(self.memory))]
#         sample_indices = random.choices(memory_pool, k=sample_size, weights=sample_probs)
#         batch = []
#         for i in range(len(sample_indices)):
#             batch.append(self.memory[sample_indices[i]])
#         #print(len(batch))
#         return batch,sample_indices

#     def get_probabilities(self, priority_scale):
#         scaled_priorities = np.array(self.priorities) ** priority_scale
#         sample_probabilities = scaled_priorities / sum(scaled_priorities)
#         return sample_probabilities

#     def set_priorities(self, indices, errors, offset=0):
#         for i, e in zip(indices, errors):
#             self.priorities[i] = abs(e)*self.join[i] + offset

class BasePolicy(object):
    def __init__(self, config):
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.GAMMA = config['GAMMA']
        self.EPS_START = config['EPS_START']
        self.EPS_END = config['EPS_END']
        self.EPS_DECAY = config['EPS_DECAY']

        self.device = torch.device(config['device'])
        self.prioritized = config['prioritized']

        self.policy_net = DQ_Net(config['num_col'], config['num_actions'], config).to(self.device)
        self.target_net = DQ_Net(config['num_col'], config['num_actions'], config).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.prioritized :
            # self.memory = PriorityReplayMemory(config['CAPACITY'])
            raise NotImplementedError

        else:
            self.memory = ReplayMemory(config['CAPACITY'])


        # optimizer
        if config['optim'] == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay']) #

        elif config['optim'] == 'rms':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])

        else:
            raise ValueError('Not supported Loss')
        
        # loss funcction
        if config['loss'] == 'l1':
            self.criterion = nn.SmoothL1Loss()
        elif config['loss'] == 'huber':
            self.criterion = nn.HuberLoss()
        elif config['loss'] == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = None
            #raise ValueError('Not supported Loss')

        self.num_actions = config['num_actions']

    def select_action(self, obs, is_train=True):
        global steps_done
        if is_train:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * steps_done / self.EPS_DECAY)
            steps_done += 1
            if random.random() > eps_threshold:
                with torch.no_grad():
                    action_num = self.policy_net(obs).max(0)[1]
            else:
                action_num = torch.multinomial(obs['action_mask'], 1)
        else:
            with torch.no_grad():
                action_num = self.policy_net(obs).max(0)[1]
        return int(action_num)

    
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 99999

        if self.prioritized:
            # transitions, sample_indices = self.memory.sample(self.BATCH_SIZE)
            raise NotImplementedError
        else:
            transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Experience(*zip(*transitions))



        non_final_mask = tuple(map(lambda s: s is not None,
                                                batch.obs))

        if True not in non_final_mask:
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        else :
            non_final_mask = torch.tensor(non_final_mask, device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s["db"].unsqueeze(0) for s in batch.obs
                                               if s is not None], dim=0)
            non_final_next_states_mask = torch.cat([s["action_mask"].unsqueeze(0) for s in batch.obs if s is not None ], dim=0)


            next_states_dict = {}

            next_states_dict["db"] = non_final_next_states
            next_states_dict["action_mask"] = non_final_next_states_mask

            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            #next_state_values[non_final_mask] = self.target_net(next_states_dict).max(1)[0].detach()

            #next_state_values[non_final_mask] = self.target_net(next_states_dict).min(1)[0].detach()
            # print(self.policy_net(next_states_dict).shape)
            # print(self.policy_net(next_states_dict).argmax(1))
            next_state_index_in_policy  = self.policy_net(next_states_dict).argmax(1).unsqueeze(1).detach()
            next_state_values[non_final_mask] = self.target_net(next_states_dict).gather(1, next_state_index_in_policy).squeeze(1).detach()



        states_mask = torch.cat([s["action_mask"].unsqueeze(0) for s in batch.old_obs
                                               ], dim=0)

        state_batch = torch.cat([s["db"].unsqueeze(0) for s in batch.old_obs], dim=0)


      

        
        states_dict = {}
        states_dict["db"] = state_batch
        states_dict["action_mask"] = states_mask


        action_batch = torch.tensor(batch.action_num, device=self.device).unsqueeze(-1)
        reward_batch = torch.tensor(batch.cost, device=self.device)

        state_action_values = self.policy_net(states_dict).gather(1, action_batch)
        # state_action_values = self.policy_net(states_dict)[action_batch]

        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch


        # if self.prioritized :
        #     error = expected_state_action_values.unsqueeze(1)-state_action_values
        #     error2=error.squeeze().detach().clone().cpu().tolist()
        #     self.memory.set_priorities(sample_indices,error2)

        loss = self.criterion(state_action_values,
                                  expected_state_action_values.unsqueeze(1))
        

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            torch.nn.utils.clip_grad_norm_(param, 10, norm_type=2)

        self.optimizer.step()
        return loss.item()

class Tree_Cus(BasePolicy):
    def __init__(self, config):
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.GAMMA = config['GAMMA']
        self.EPS_START = config['EPS_START']
        self.EPS_END = config['EPS_END']
        self.EPS_DECAY = config['EPS_DECAY']

        self.device = torch.device(config['device'])
        self.prioritized = config['prioritized']

        self.policy_net = Tree_Net(config['num_col'], config['num_actions'], config).to(self.device)
        self.target_net = Tree_Net(config['num_col'], config['num_actions'], config).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.prioritized :
            # self.memory = PriorityReplayMemory(config['CAPACITY'])
            raise NotImplementedError

        else:
            self.memory = ReplayMemory(config['CAPACITY'])


        # optimizer
        if config['optim'] == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay']) #

        elif config['optim'] == 'rms':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])

        else:
            raise ValueError('Not supported Loss')
        
        # loss funcction
        if config['loss'] == 'l1':
            self.criterion = nn.SmoothL1Loss()
        elif config['loss'] == 'huber':
            self.criterion = nn.HuberLoss()
        elif config['loss'] == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = None
            #raise ValueError('Not supported Loss')

        self.num_actions = config['num_actions']
    
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 99999

        if self.prioritized:
            # transitions, sample_indices = self.memory.sample(self.BATCH_SIZE)
            raise NotImplementedError
        else:
            transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Experience(*zip(*transitions))



        non_final_mask = tuple(map(lambda s: s is not None,
                                                batch.obs))
        
        if True not in non_final_mask:
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        else:
            non_final_mask = torch.tensor(non_final_mask, device=self.device, dtype=torch.bool)

            tree_states = []
            mask_states = []
            link_matrices = []
            for s in batch.obs:
                if s is not None:
                    mask_states.append(s['action_mask'])
                    tree_states.append(self.policy_net.tree_encoding(s['query_lst'])) #这边确认一下是policy 还是 target
                    # tree_states.append(s['db'])
                    link_matrices.append(s['link_mtx'])
            


            # next_link_mtx = torch.stack(
            #     [s["link_mtx"] for key, s in enumerate(batch.old_obs)
            #      if non_final_mask.tolist()[key] is not False], dim=0) 
            non_final_next_states = torch.stack(tree_states, dim=0)
            non_final_next_states_mask = torch.stack(mask_states, dim=0)
            next_link_mtx = torch.stack(link_matrices, dim=0)


            next_states_dict = {}
            next_states_dict["db"] = non_final_next_states
            next_states_dict["action_mask"] = non_final_next_states_mask
            next_states_dict["link_mtx"] = next_link_mtx

            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

            next_state_index_in_policy  = self.policy_net(next_states_dict).argmax(1).unsqueeze(1).detach()
            next_state_values[non_final_mask] = self.target_net(next_states_dict).gather(1, next_state_index_in_policy).squeeze(1).detach()

        states_mask = torch.stack([s["action_mask"] for s in batch.old_obs], dim =0)
        state_batch = torch.stack([self.policy_net.tree_encoding(s['query_lst']) for s in batch.old_obs], dim=0)
        # state_batch = torch.stack([s['db'] for s in batch.old_obs], dim=0)
        link_mtx = torch.stack([s["link_mtx"] for s in batch.old_obs], dim=0)
        
        states_dict = {}

        states_dict["action_mask"] = states_mask
        states_dict["db"] = state_batch
        states_dict["link_mtx"] = link_mtx
        action_batch = torch.tensor(batch.action_num, device=self.device).unsqueeze(-1)
        reward_batch = torch.tensor(batch.cost, device=self.device)

        state_action_values = self.policy_net(states_dict).gather(1, action_batch)
        # state_action_values = self.policy_net(states_dict)[action_batch]

        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch


        # if self.prioritized :
        #     error = expected_state_action_values.unsqueeze(1)-state_action_values
        #     error2=error.squeeze().detach().clone().cpu().tolist()
        #     self.memory.set_priorities(sample_indices,error2)

        loss = self.criterion(state_action_values,
                                  expected_state_action_values.unsqueeze(1))
        

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            torch.nn.utils.clip_grad_norm_(param, 10, norm_type=2)

        self.optimizer.step()
        return loss.item()

        






            

    
    