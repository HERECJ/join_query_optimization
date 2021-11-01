import torch
import torch.nn as nn
import torch.nn.functional as F


from env.QueryGraph import Query, Relation
import numpy as np
import os
FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38


class Base_Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, obs):
        raise NotImplementedError

class DQ_Net(Base_Net):
    def __init__(self, num_col, num_actions, config, **kwargs):
        super().__init__()
        self.MLP = MultiLayer(4*num_col, config["emb_dim"], num_actions)
        self.device = config['device']
    
    def forward(self, obs):
        out_put = self.MLP(obs['db'].to(self.device))
        mask = obs['action_mask'].to(device=self.device)
        inf_mask = torch.clamp(torch.log(mask), FLOAT_MIN, FLOAT_MAX)
        return inf_mask + out_put


class MultiLayer(nn.Module):
    def __init__(self, input_size, hidden_dim, out_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_size),
        )

    def forward(self, embed):
        out = self.mlp(embed)
        return out

class Tree_Net(Base_Net):
    def __init__(self, num_col, num_actions, config, **kwargs):
        super().__init__()
        emb_dim = config['emb_dim']
        emb_bias = config['emb_bias']
        emb_init_std = config['emb_init_std']
        graph_dim = config['emb_dim']
        dropout = config['dropout']
        graph_bias = config['graph_bias']
        graph_pooling = config['graph_pooling']
        device = config['device']

        self.Granularity = 4
        act = config['activation'].lower()
        if act == 'relu':
            activation = F.relu
        elif act == 'tanh':
            activation = F.tanh
        else:
            raise ValueError('Relu or Tanh activation functions are supported')
        
        self.device = torch.device(device)

        self._Column_Emb = nn.Embedding(num_col, emb_dim)
        nn.init.normal_(self._Column_Emb.weight, std=emb_init_std)

        self._Column_Select_Emb = nn.Linear(self.Granularity+1, emb_dim, bias=emb_bias)
        nn.init.normal_(self._Column_Select_Emb.weight, std=emb_init_std)

        self.cat_col_Emb = nn.Linear(emb_dim*2, emb_dim, bias=emb_bias)
        nn.init.normal_(self.cat_col_Emb.weight, std=emb_init_std)

        if config['rel_pretrain']:
            emb_file = 'table_emb_deepwalk/out_deepwalk_{}.embeddings'.format(emb_dim)
            if os.path.exists(emb_file):
                emb = []
                with open(emb_file, 'r') as fn:
                    for line in fn.readlines():
                        row = line.strip().split(' ')
                        emb.append(row[1:])
                embs = np.asarray(emb).astype(float)
                self._Rel_Emb = nn.Embedding.from_pretrained(torch.tensor(embs, dtype=torch.float, device=self.device), freeze=True)
        
        else:
            self._Rel_Emb = nn.Embedding(config['num_rel'], emb_dim)
        
        self.Graph = Graph_Net(emb_dim, graph_dim, graph_bias, graph_pooling, emb_init_std, activation)

        # Tree Structure Net
        self.tree_att_k = nn.Linear(emb_dim, emb_dim, bias=emb_bias)
        self.tree_att_q = nn.Linear(emb_dim, emb_dim, bias=emb_bias)
        self.tree_att_v = nn.Linear(emb_dim, emb_dim, bias=emb_bias)
        self.att_dim = emb_dim
        nn.init.normal_(self.tree_att_k.weight, std=emb_init_std)
        nn.init.normal_(self.tree_att_q.weight, std=emb_init_std)
        nn.init.normal_(self.tree_att_v.weight, std=emb_init_std)

        self.num_actions = num_actions
        self.emb_dim = emb_dim

        self.fc_out = nn.Linear(graph_dim + emb_dim, num_actions, bias=emb_bias)
            
    
    def forward(self, obs):
        observe = obs['db']
        link_mtx =obs['link_mtx'].to(device=self.device)
        out_graph = self.Graph(link_mtx, self._Rel_Emb.weight)
        mask = obs['action_mask'].to(device=self.device)
        inf_mask = torch.clamp(torch.log(mask), FLOAT_MIN, FLOAT_MAX)
        if observe is None:
            out_put = self.fc_out(torch.cat([out_graph, torch.zeros(self.emb_dim, device=self.device)]))
            return inf_mask + out_put
        else:
            out_put = self.fc_out(torch.cat([out_graph, observe.to(self.device)], dim=-1))
            return inf_mask + out_put
        

    def get_table_emb(self, idx):
        return self._Rel_Emb(torch.LongTensor([idx]).to(device=self.device))




    def tree_encoding(self, query):
        if query is None:
            return torch.zeros(self.emb_dim, device=self.device)
        assert type(query) is list
        tree_emb  = []
        for subquery in query:
            assert isinstance(subquery, Query)
            tree_emb.append(self._tree_encoding(subquery))
        tree_emb = torch.cat(tree_emb, dim=0)
        return torch.mean(tree_emb, dim=0)


    def _tree_encoding(self, query):
        if type(query.left) is Relation and type(query.right) is Relation:
            l_table_emb = self.get_table_emb(query.left.id)
            r_table_emb = self.get_table_emb(query.right.id)
        elif type(query.left) is Query and type(query.right) is Relation:
            l_table_emb = self._tree_encoding(query.left)
            r_table_emb = self.get_table_emb(query.right.id)
        elif type(query.left) is Relation and type(query.right) is Query:
            l_table_emb = self.get_table_emb(query.left.id)
            r_table_emb = self._tree_encoding(query.right)
        elif type(query.left) is Query and type(query.right) is Query:
            l_table_emb = self._tree_encoding(query.left)
            r_table_emb = self._tree_encoding(query.right)
        
        left_join_col, right_join_col, left_colum_emb, right_colum_emb = query.get_joined_column_state()
        
        l_col_emb = torch.zeros(len(left_join_col),  self.emb_dim, device=self.device)
        r_col_emb = torch.zeros(len(left_join_col),  self.emb_dim, device=self.device)
        for i in range(len(left_join_col)):
            l_col_emb[i] = self.get_col_emb(left_join_col[i], left_colum_emb[i])
            r_col_emb[i] = self.get_col_emb(right_join_col[i], right_colum_emb[i])
        
        left_colum_emb = torch.mean(l_col_emb, dim=0)
        right_col_emb = torch.mean(r_col_emb, dim=0)
        query_emb = torch.cat([l_table_emb, left_colum_emb.unsqueeze(0), right_col_emb.unsqueeze(0), r_table_emb], dim=0)

        return self.att_forward(query_emb)
    
    def att_forward(self, query_emb):

        val_k = self.tree_att_k(query_emb)
        val_q = self.tree_att_q(query_emb)
        val_v = self.tree_att_v(query_emb)
        scores = val_q.mm(val_k.T) / self.att_dim
        return F.softmax(scores, dim=-1).mm(val_v).sum(0, keepdim=True)


    def get_col_emb(self, col_id, col_emb):
        col_id = self._Column_Emb(torch.LongTensor([col_id]).to( device=self.device))
        col_emb = self._Column_Select_Emb(torch.tensor(col_emb, device=self.device)).unsqueeze(0)
        return self.cat_col_Emb(torch.cat([col_id, col_emb], dim=1))


        

class Graph_Net(nn.Module):
    def __init__(self, emb_dim, graph_dim, graph_bias, graph_pooling, emb_init_std, activation):
        super().__init__()
        
        self.graph_weight = nn.Parameter(torch.randn(emb_dim, graph_dim))
        nn.init.normal_(self.graph_weight, std=emb_init_std)
        self.graph_bias = None
        if graph_bias is True:
            self.graph_bias = nn.Parameter(torch.zeros(graph_dim))
        

        self.activate = activation
        self.graph_pooling = graph_pooling

    def forward(self, link_mtx, rel_embs):
        # Graph
        # A * X * W
        # Here X is the embedding of tables, further can be transferred as the customed embeddings
        # A is the Adjacency matrix recording the neighbours of the nodes(relations)
        # W is the parameters
        # Further we can fit into advanced GCNs.
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

