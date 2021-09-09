import torch
from agents.run.masking_env_tree import CrossVal
from queryoptimization.utils import setup_seed
from agents.run.agents import agent
import psycopg2
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
in_put_size=4004
out_put_size=756
memory_size=10000
episodes=0
Test=100
is_train="train"

def main(env_config, device='cpu'):
    global  episodes
    device = torch.device(device)
    env = CrossVal(env_config)
    obs = env.reset()
    action_total = env.action_space.n
    agent_a = agent(BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE, env.wrapped.num_of_columns, env.wrapped.num_of_relations, action_total,memory_size)
    first_ep=0

    for i in range(1000000):
        if  first_ep==0:
            first_ep=1
            obs['tree'] = None
            obs['link_mtx']=torch.tensor(env.wrapped.query.link_mtx, device=device, dtype=torch.float32)
            obs['all_table_embed'] = env.wrapped.table_embeds

        else:
            obs['link_mtx'] = old_obs['link_mtx']
            obs['all_table_embed'] = old_obs['all_table_embed'].copy()

        action_num = agent_a.select_action(obs,is_train)
        old_obs = obs.copy()
        obs, cost, done, sub_trees = env.step(action_num)
        cost = torch.tensor([cost]).cuda()
        action_num = torch.tensor([action_num]).cuda()

        if not done:
            obs['tree'] = sub_trees
            agent_a.memory.push(old_obs, action_num, obs, cost, done)     #old_obs: mask,db , tree ,link_mtx,all_table_embed
                                                                          #    obs: mask,db , tree
        else:
            obs = None
            agent_a.memory.push(old_obs, action_num, obs, cost, done)

        agent_a.optimize_model()


        if done is True:
            if episodes % TARGET_UPDATE == 0:
                agent_a.target_net.load_state_dict(agent_a.policy_net.state_dict())
            if episodes %Test==0:
                eval(agent_a,episodes)
            if episodes==10000:
                break

            obs = env.reset()
            episodes += 1
            first_ep = 0


def eval(agent_a,episodes):
    env_config = {"fold_idx": 0, "process": 1}
    device = torch.device(device='cuda')
    env = CrossVal(env_config)
    for sql in range(33):
        obs = env.reset()
        obs['tree'] = None
        done = False
        while done is False:
            obs['link_mtx'] = torch.tensor(env.wrapped.query.link_mtx, device=device, dtype=torch.float32)
            obs['all_table_embed'] = env.wrapped.table_embeds
            action_num = agent_a.select_action(obs, 'eval')
            obs, cost, done, sub_trees = env.step(action_num)
            obs['tree'] = sub_trees

        if done is True:
            sql = env.wrapped.sql
            sql_id = env.wrapped.sql_id
            estimatedRows ,estimatedcosts = get_cost_rows(sql)

            # with  open(file=r"/data/ygy/code_list/join2/result"+"/episodes"+str(episodes)+".txt", mode='a')as f:
            #     f.write(sql_id+'|'+sql+'estimatedRows:'+estimatedRows+'estimatedcosts:'+estimatedcosts+'\n')


def get_cost_rows(sql):
    try:
        conn = psycopg2.connect(
            database='exp', user='imdb', password='ych19960128', host='127.0.0.1', port='5433')
    except:
        print("I am unable to connect to the database")
    cursor = conn.cursor()
    cursor.execute(""" EXPLAIN """ + sql)
    rows = cursor.fetchall()

    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedRows = row0[1].replace("rows=", "")

    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedcosts = row0[0].split("..")[1]
    return estimatedRows ,estimatedcosts





if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    env_config = {"fold_idx": 0, "process":0}
    setup_seed(10)
    main(env_config, device='cuda')


