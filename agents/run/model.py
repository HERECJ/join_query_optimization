import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from queryoptimization.QueryGraph import Relation, Query
from queryoptimization.subplan2tree import Tree
from collections import namedtuple, deque
import scipy.special as sp
import math
import random
import numpy as np

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38



class Q_Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        """
            Define model parameters
        """
        pass

    def forward(self, observation):
        raise NotImplementedError




class DQN(nn.Module):
    def __init__(self,num_col, num_rel, num_actions, device, **kwargs):
        super(DQN, self).__init__()

        linear_input_size =  num_col*num_rel

        self.head = nn.Linear(linear_input_size, num_actions)
        self.device = torch.device(device)

    def forward(self,obs):
        new_obs=obs["db"]
        new_obs =torch.FloatTensor(new_obs).to(self.device)
        probs= self.head(new_obs)
        action_mask = torch.tensor(obs["action_mask"]).to(self.device)
        action_mask = torch.clip(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        probs = probs + action_mask
        return probs


class Net1(Q_Net):
    def __init__(self, num_col, num_rel, num_actions, config=dict(), Granularity=4, **kwargs):
        super().__init__()
        emb_dim = config['emb_dim']
        emb_bias = config['emb_bias']
        emb_init_std = config['emb_init_std']
        graph_dim = config['graph_dim']
        dropout = config['dropout']
        graph_bias = config['graph_bias']
        graph_pooling = config['graph_pooling']
        device = config['device']

        act = config['activation'].lower()
        if act == 'relu':
            activation = F.relu
        elif act == 'tanh':
            activation = F.tanh
        else:
            raise ValueError('Relu or Tanh activation functions are supported')

        self.device = torch.device(device)
        # Embedding for Columns and Tables
        self._Column_Emb = nn.Embedding(num_col, emb_dim)
        self._Rel_Emb = nn.Embedding(num_rel, emb_dim)
        nn.init.normal_(self._Column_Emb.weight, std=emb_init_std)
        nn.init.normal_(self._Rel_Emb.weight, std=emb_init_std)
        self._Column_Select_Emb = nn.Linear(Granularity+1, emb_dim, bias=emb_bias)
        nn.init.normal_(self._Column_Select_Emb.weight, std=emb_init_std)


        # Graph parameters
        self.graph_weight = nn.Parameter(torch.randn(emb_dim, graph_dim))
        nn.init.normal_(self.graph_weight, std=emb_init_std)
        self.graph_bias = None
        if graph_bias is True:
            self.graph_bias = nn.Parameter(torch.zeros(graph_dim))

        self.activate = activation
        self.dropout = dropout
        self.graph_pooling = graph_pooling
        self.Granularity = Granularity

        # Variables for Tree Structures
        # self.tree_att_k = nn.Parameter(torch.randn(4, emb_dim))
        # self.tree_att_q = nn.Parameter(torch.randn(4, emb_dim))
        # self.tree_att_v = nn.Parameter(torch.randn(4, emb_dim))
        # nn.init.normal_(self.tree_att_k, std = emb_init_std)
        # nn.init.normal_(self.tree_att_q, std = emb_init_std)
        # nn.init.normal_(self.tree_att_v, std = emb_init_std)
        self.tree_att_k = nn.Linear(emb_dim, emb_dim, bias=emb_bias)
        self.tree_att_q = nn.Linear(emb_dim, emb_dim, bias=emb_bias)
        self.tree_att_v = nn.Linear(emb_dim, emb_dim, bias=emb_bias)
        nn.init.normal_(self.tree_att_k.weight, std=emb_init_std)
        nn.init.normal_(self.tree_att_q.weight, std=emb_init_std)
        nn.init.normal_(self.tree_att_v.weight, std=emb_init_std)
        self.att_dim = emb_dim

        self.fc_out = nn.Linear(graph_dim + emb_dim,
                                num_actions, bias=emb_bias)

        self.dqn=DQN(num_col, num_rel, num_actions, self.device).to(self.device)


    def forward(self, obs):
        sub_trees=obs['tree']
        link_mtx=obs['link_mtx']
        all_table_embed=obs['all_table_embed']

        if sub_trees is None:
            return self.dqn(obs)

        if type(all_table_embed) is dict:
            all_table_embed=[all_table_embed]
        # Graph
        out_graph = self.graph_forward(link_mtx,all_table_embed)

        type_opm=0
        for i in sub_trees:
            if type(i)== list:
                type_opm=1

        if type_opm:
            tree_all=[]
            for tree in sub_trees:
                if tree != None:
                    tree_emb_list = torch.zeros(len(tree), self.att_dim)
                    for idx, tree in enumerate(tree):
                        tree_emb_list[idx] = self.tree_forward(tree)
                    temp_tree = torch.sum(tree_emb_list, dim=0).to(self.device)
                    tree_all.append(temp_tree)
                else :
                    tree_all.append(torch.zeros(self.att_dim).to(self.device))
            tree_emb=torch.stack(tree_all, dim=0).to(self.device)

        else:
            tree_emb_list = torch.zeros(len(sub_trees), self.att_dim)
            #print("sub_trees",tree_emb_list)
            for idx, subtree in enumerate(sub_trees):
                tree_emb_list[idx] = self.tree_forward(subtree)
            #print("sub_trees2",tree_emb_list)
            tree_emb = torch.sum(tree_emb_list, dim=0).to(self.device)
            #print(tree_emb)


        # Tree_Att
        # Add processing for the initial state
        # if sub_trees is None:
            # tree_emb = None

        if len(out_graph.shape)>1:
            out_put = self.fc_out(torch.cat([out_graph, tree_emb],dim=1))
        else :
            out_put = self.fc_out(torch.cat([out_graph, tree_emb]))

        action_mask = torch.tensor(obs['action_mask'], device=self.device)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        masked_logits = inf_mask + out_put
        #print(masked_logits)
        return masked_logits

    def graph_forward(self, link_mtx,all_table_embed):
        # Graph
        # A * X * W
        # Here X is the embedding of tables, further can be transferred as the customed embeddings
        # A is the Adjacency matrix recording the neighbours of the nodes(relations)
        # W is the parameters
        # Further we can fit into advanced GCNs.
        rel_embs = self.table_emb(all_table_embed)
        support = torch.matmul(link_mtx, rel_embs)
        out_graph = self.activate(
            torch.matmul(support, self.graph_weight) + self.graph_bias)
        if self.graph_pooling.lower() in ['sum']:
            if len(out_graph.shape)>2:
                out_graph_pooling = torch.sum(out_graph, dim=1)
            else:
                out_graph_pooling = torch.sum(out_graph, dim=0)
        elif self.graph_pooling.lower() in ['mean', 'average', 'avg']:
            if len(out_graph.shape)>2:
                out_graph_pooling = torch.mean(out_graph, dim=1)
            else:
                out_graph_pooling = torch.mean(out_graph, dim=0)

        return out_graph_pooling

    def tree_forward(self, tree):
        if type(tree.l_name) is str and type(tree.r_name) is str:
            l_table_emb = self.get_table_emb(tree.l_table_id)
            r_table_emb = self.get_table_emb(tree.r_table_id)
        elif type(tree.l_name) is str and isinstance(tree.r_name, Tree):

            l_table_emb = self.get_table_emb(tree.l_table_id)
            r_table_emb = self.tree_forward(tree.r_name)
        elif isinstance(tree.l_name, Tree) and type(tree.r_name) is str:
            l_table_emb = self.tree_forward(tree.l_name)
            r_table_emb = self.get_table_emb(tree.r_table_id)
        elif isinstance(tree.l_name, Tree) and isinstance(tree.r_name, Tree):
            l_table_emb = self.tree_forward(tree.l_name)
            r_table_emb = self.tree_forward(tree.r_name)
        else:
            raise ValueError("Invalid subtrees for subqueries")

        assert len(tree.l_column_id) > 0, ValueError("Cross Join")
        assert len(tree.l_column_id) == len(tree.r_column_id), ValueError(
            "The joined columns are not consistent")

        if self.graph_pooling.lower() in ['sum']:
            l_col_emb = torch.sum(self.get_col_emb(
                tree.l_column_id), dim=0, keepdim=True)
            r_col_emb = torch.sum(self.get_col_emb(
                tree.r_column_id), dim=0, keepdim=True)
        elif self.graph_pooling.lower() in ['mean', 'average', 'avg']:
            l_col_emb = torch.mean(self.get_col_emb(
                tree.l_column_id), dim=0, keepdim=True)
            r_col_emb = torch.mean(self.get_col_emb(
                tree.r_column_id), dim=0, keepdim=True)
       # print(l_table_emb, l_col_emb, r_col_emb, r_table_emb)
        query_emb = torch.cat(
            [l_table_emb, l_col_emb, r_col_emb, r_table_emb], dim=0)  # 4 * emb_dim

        return self.att_forward(query_emb)

    def att_forward(self, query_emb):

        val_k = self.tree_att_k(query_emb)
        val_q = self.tree_att_q(query_emb)
        val_v = self.tree_att_v(query_emb)
        scores = val_q.mm(val_k.T) / self.att_dim
        return F.softmax(scores, dim=-1).mm(val_v).sum(0, keepdim=True)

    def table_emb(self,all_table_embed):

        # Utilize the col emb for table embs
        return self._Rel_Emb.weight

    def get_table_emb(self, table_id):
        return self._Rel_Emb(torch.LongTensor([table_id]).to(self.device))

    def get_col_emb(self, col_id):
        return self._Column_Emb(torch.LongTensor(col_id).to(self.device))


class Net2(Net1):
    def __init__(self, num_col, num_rel, num_actions, config=dict(), Granularity=4, **kwargs):
        super().__init__(num_col, num_rel, num_actions, config=config, Granularity=Granularity)
        emb_dim = config['emb_dim']
        emb_bias = config['emb_bias']
        emb_init_std = config['emb_init_std']
        self._Table_Emb = nn.Linear(4, emb_dim, bias=emb_bias)
        nn.init.normal_(self._Table_Emb.weight, std=emb_init_std)
        self.num_rel=num_rel
        self.emb_dim=emb_dim


    def tree_forward(self, tree):
        if type(tree.l_name) is str and type(tree.r_name) is str:
            l_table_emb = self.get_table_emb(tree.l_table_embed)
            r_table_emb = self.get_table_emb(tree.r_table_embed)
        elif type(tree.l_name) is str and isinstance(tree.r_name, Tree):

            l_table_emb = self.get_table_emb(tree.l_table_embed)
            r_table_emb = self.tree_forward(tree.r_name)
        elif isinstance(tree.l_name, Tree) and type(tree.r_name) is str:
            l_table_emb = self.tree_forward(tree.l_name)
            r_table_emb = self.get_table_emb(tree.r_table_embed)
        elif isinstance(tree.l_name, Tree) and isinstance(tree.r_name, Tree):
            l_table_emb = self.tree_forward(tree.l_name)
            r_table_emb = self.tree_forward(tree.r_name)
        else:
            raise ValueError("Invalid subtrees for subqueries")

        assert len(tree.l_column_id) > 0, ValueError("Cross Join")
        assert len(tree.l_column_id) == len(tree.r_column_id), ValueError(
            "The joined columns are not consistent")

        if self.graph_pooling in ['SUM', 'sum']:
            l_col_emb = torch.sum(self.get_col_emb(
                tree.l_column_embed), dim=0, keepdim=True)

            r_col_emb = torch.sum(self.get_col_emb(
                tree.r_column_embed), dim=0, keepdim=True)
        elif self.graph_pooling in ['MEAN', 'mean', 'AVERAGE', 'average', 'AvG', 'avg']:
            l_col_emb = torch.mean(self.get_col_emb(
                tree.l_column_embed), dim=0, keepdim=True)
            r_col_emb = torch.mean(self.get_col_emb(
                tree.r_column_embed), dim=0, keepdim=True)
        query_emb = torch.cat(
            [l_table_emb, l_col_emb, r_col_emb, r_table_emb], dim=0)  # 4 * emb_dim

        return self.att_forward(query_emb)

    def table_emb(self,all_table_embed):
        table_embed_all=[]
        for num in range(len(all_table_embed)):
            table_embed = []
            for id in range(self.num_rel):
                if id not in all_table_embed:
                    table_embed.append(torch.zeros(1, self.emb_dim).to(self.device))
                else:
                    table_embed.append(
                        torch.sum(torch.FloatTensor(table_embed).to(self.device), dim=0, keepdim=True))
            table_embed_all.append(torch.cat(table_embed, dim=0))
        if len(all_table_embed)==1:
            return table_embed_all[0]
        else:
            return torch.stack(table_embed_all, dim=0)

    def get_table_emb(self, table_embed):
        table_embed=self._Table_Emb(torch.sum(torch.FloatTensor(table_embed).to(self.device),dim=0,keepdim=True))
        return table_embed

    def get_col_emb(self, column_embed):
        column_embed_all=[]
        for col in column_embed:
            join_selectivity_embed = [0 for _ in range(self.Granularity+1)]
            join_selectivity_embed[0]=col[2]
            if col[1]==1:
                join_selectivity_embed[self.Granularity]=1
            elif col[1]==0:
                join_selectivity_embed[1] = 1
            else:
                num = math.ceil(col[1] % (1 / self.Granularity))
                join_selectivity_embed[num] = 1

            join_selectivity_embed=self._Column_Select_Emb(torch.FloatTensor([join_selectivity_embed]).to(self.device))
            column_id_embed = self._Column_Emb(torch.LongTensor([col[0]]).to(self.device))
            column_embed_all.append(join_selectivity_embed)
            column_embed_all.append(column_id_embed)
        column_embed_all=torch.cat(column_embed_all,dim=0)
        #print(column_embed_all)

        return column_embed_all



class RTOS(Net1):
    """
        Liang's work
    """
    def __init__(self, num_col, num_rel, num_actions, config, Granularity, **kwargs):
        super().__init__(num_col, num_rel, num_actions, config=config, Granularity=Granularity)
        raise NotImplementedError



    def forward(self, observation):
        obs, sub_trees, link_mtx = observation
        out_graph = link_mtx.flatten()

        raise NotImplementedError


class DQ(Net1):
    """
        DQ, baseline methods
    """

    def __init__(self, num_col, num_rel, num_actions, config, Granularity, **kwargs):
        super().__init__(num_col, num_rel, num_actions, config=config, Granularity=Granularity)
        raise NotImplementedError

    def forward(self, observation):
        raise NotImplementedError
