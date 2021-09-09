import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import scipy.special as sp
import math
import random
import numpy as np
from ..run.model import DQN,Net1,Net2, FLOAT_MAX, FLOAT_MIN


Experience=namedtuple('Experience',('old_obs', 'action_num', 'obs', 'cost', 'done'))

steps_done=0
class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
    def push (self,*args):
        self.memory.append(Experience(*args))
    def sample (self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)

class agent(object):
    def __init__(self,BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TARGET_UPDATE, num_col, num_rel, num_actions,memory_size):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        #self.in_put_size=in_put_size
        #self.out_put_size=out_put_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Experience = namedtuple('Experience',('old_obs', 'action_num', 'obs', 'cost', 'done'))
        #self.model_list = model_list[DQN, Net1]
        #self.idx = 1
        #model_list[1]()
        self.model_state='train'

        self.policy_net = Net2(num_col, num_rel, num_actions).to(self.device)
        self.target_net = Net2(num_col, num_rel, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory=ReplayMemory(memory_size)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def select_action(self,obs,is_train):

        global steps_done
        if is_train =='train':
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * steps_done / self.EPS_DECAY)
            steps_done += 1

            if sample > eps_threshold:  #将0换成 eps_threshold
                with torch.no_grad():
                    probs = self.policy_net(obs).cpu()
                    # action_mask = obs["action_mask"]
                    # action_mask = np.clip(np.log(action_mask), -np.inf, np.inf)
                    # #+ action_mask
                    final_probs = sp.softmax(probs)
                    action_num = np.argmax(final_probs)

            else:
                action_total = len(obs["action_mask"])
                candidate_action = np.arange(0, action_total)
                probs = np.ones(action_total, dtype=np.float)
                action_mask = obs["action_mask"]
                action_mask = np.clip(np.log(action_mask), -np.inf, np.inf)
                final_probs = sp.softmax(probs + action_mask)
                action_num = int(np.random.choice(a=candidate_action, size=1, p=final_probs))


        elif is_train =='eval':
            with torch.no_grad():
                probs = self.policy_net(obs).cpu()
                # action_mask = obs["action_mask"]
                # action_mask = np.clip(np.log(action_mask), -np.inf, np.inf)
                # #+ action_mask
                final_probs = sp.softmax(probs)
                action_num = np.argmax(final_probs)

        return action_num


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Experience(*zip(*transitions))


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.obs)), device=self.device, dtype=torch.bool)

        non_final_next_states_mask = torch.cat([torch.FloatTensor([s["action_mask"]]) for s in batch.obs
                                           if s is not None], dim=0)

        non_final_next_states = torch.cat([torch.FloatTensor([s["db"]]) for s in batch.obs
                                           if s is not None], dim=0)

        next_tree =  [s["tree"] for s in batch.obs if s is not None]


        #print(batch.obs)

        next_link_mtx =   torch.stack(
            [s["link_mtx"] for key, s in enumerate(batch.old_obs)
                                           if s is not None and non_final_mask.tolist()[key] is not False ], dim=0)  #是少的


        next_table_embed = [s["all_table_embed"] for key, s in enumerate(batch.old_obs)
             if s is not None and non_final_mask.tolist()[key] is not False]  # 是少的

        next_states_dict={}
        next_states_dict["db"]= non_final_next_states
        next_states_dict["action_mask"] = non_final_next_states_mask
        next_states_dict["tree"] = next_tree
        next_states_dict["link_mtx"] = next_link_mtx
        next_states_dict["all_table_embed"] = next_table_embed



        non_final_states_mask = torch.cat([torch.FloatTensor([s["action_mask"]]) for s in batch.old_obs
                                                if s is not None], dim=0)

        state_batch = torch.cat([torch.FloatTensor([s["db"]]) for s in batch.old_obs
                                 if s is not None], dim=0)

        tree = [s["tree"] for s in batch.old_obs]

        link_mtx = torch.stack(
            [s["link_mtx"] for key, s in enumerate(batch.old_obs)
             if s is not None ], dim=0)  # 是少的

        table_embed = [s["all_table_embed"] for key, s in enumerate(batch.old_obs)
             if s is not None ] # 是少的


        states_dict = {}
        states_dict["db"] = state_batch
        states_dict["action_mask"] = non_final_states_mask
        states_dict["tree"] = tree
        states_dict["link_mtx"] = link_mtx
        states_dict["all_table_embed"] = table_embed

        action_batch = torch.cat(batch.action_num).unsqueeze(1)

        reward_batch = torch.cat(batch.cost)

        state_action_values = self.policy_net(states_dict).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        next_state_values[non_final_mask] = self.target_net( next_states_dict).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class BasePolicy(object):
    def __init__(self, config, model:nn.Module):
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.GAMMA = config['GAMMA']
        self.EPS_START = config['EPS_START']
        self.EPS_END = config['EPS_END']
        self.EPS_DECAY = config['EPS_DECAY']

        self.device = torch.device(config['device'])
       

        self.model_state = 'train'
        self.policy_net = model(config['num_col'], config['num_rel'], config['num_actions'], config).to(self.device)
        self.target_net = model(config['num_col'], config['num_rel'], config['num_actions'], config).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.memory = ReplayMemory(config['CAPACITY'])
        if config['optim'] == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        elif config['optim'] == 'rms':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=config['learning_rate'])
        else:
            raise ValueError('Not supported Loss')

        if config['loss'] == 'l1':
            self.criterion = nn.SmoothL1Loss()
        elif config['loss'] == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError('Not supported Loss')

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
                probs = torch.ones(self.num_actions).to(self.device)
                action_mask = torch.tensor(obs["action_mask"]).to(self.device)
                action_mask = torch.clip(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
                probs = torch.softmax(probs + action_mask, dim=-1)
                action_num = torch.multinomial(probs, 1)
        else:
            with torch.no_grad():
                action_num = self.policy_net(obs).max(0)[1]
        
        return int(action_num)
    
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 99999

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Experience(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.obs)), device=self.device, dtype=torch.bool)

        non_final_next_states_mask = torch.cat([torch.FloatTensor([s["action_mask"]]) for s in batch.obs
                                           if s is not None], dim=0)

        non_final_next_states = torch.cat([torch.FloatTensor([s["db"]]) for s in batch.obs
                                           if s is not None], dim=0)

        next_tree =  [s["tree"] for s in batch.obs if s is not None]


        #print(batch.obs)

        next_link_mtx =   torch.stack(
            [s["link_mtx"] for key, s in enumerate(batch.old_obs)
                                           if s is not None and non_final_mask.tolist()[key] is not False ], dim=0)  #是少的


        next_table_embed = [s["all_table_embed"] for key, s in enumerate(batch.old_obs)
             if s is not None and non_final_mask.tolist()[key] is not False]  # 是少的

        next_states_dict={}
        next_states_dict["db"]= non_final_next_states
        next_states_dict["action_mask"] = non_final_next_states_mask
        next_states_dict["tree"] = next_tree
        next_states_dict["link_mtx"] = next_link_mtx
        next_states_dict["all_table_embed"] = next_table_embed



        non_final_states_mask = torch.cat([torch.FloatTensor([s["action_mask"]]) for s in batch.old_obs
                                                if s is not None], dim=0)

        state_batch = torch.cat([torch.FloatTensor([s["db"]]) for s in batch.old_obs
                                 if s is not None], dim=0)

        tree = [s["tree"] for s in batch.old_obs]

        link_mtx = torch.stack(
            [s["link_mtx"] for key, s in enumerate(batch.old_obs)
             if s is not None ], dim=0)  # 是少的

        table_embed = [s["all_table_embed"] for key, s in enumerate(batch.old_obs)
             if s is not None ] # 是少的


        states_dict = {}
        states_dict["db"] = state_batch
        states_dict["action_mask"] = non_final_states_mask
        states_dict["tree"] = tree
        states_dict["link_mtx"] = link_mtx
        states_dict["all_table_embed"] = table_embed

        action_batch = torch.cat(batch.action_num).unsqueeze(1)

        reward_batch = torch.cat(batch.cost)

        state_action_values = self.policy_net(states_dict).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        next_state_values[non_final_mask] = self.target_net( next_states_dict).max(1)[0].detach()

        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch
        loss = self.criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()