from posixpath import join
from agent.model import DQ_Net
from utils import pg_utils
from agent.policy import BasePolicy
from env.masking_env import Env
import argparse
import torch
import numpy as np
import os
from configs import POLICY_CONFIG, MODEL_CONFIG

def main(config, log_dir):
    file_name = "/data/ygy/code_list/join_mod/agents/queries/crossval_sens/job_queries_simple_crossval_{}".format(config['fold'])
    env = Env({"file_name":file_name + '_train.txt', 'process': 'train', "_tree_flag_": False})
    eval_env = Env({"file_name":file_name + '_test.txt', 'process':'eval', "_tree_flag_": False})
    device = config['device']
    obs, table_num = env.reset()
    config['num_col'] = env.wrapped.num_of_columns
    config['num_rel'] = env.wrapped.num_of_relations
    config['num_actions'] = len(env.wrapped.action_list)

    agent_a = BasePolicy(config)


    first_ep = True
    loss_0 = 0.0
    epi = 0
    for i in range(config['episodes']):
        action_num = agent_a.select_action(obs, True)
        old_obs = obs.copy()
        obs, cost, done, _, join_num = env.step(action_num)
        
        # cost = torch.tensor([cost]).to(device)
        # action_num = torch.tensor([action_num]).to(device)




        # if config['Normalization'] == 'sum':
        #     w1 = join_num/np.sum(np.array([i for i in range(1,table_num)]))
        #     np.arange(1,table_num)
        #     w2 = table_num/103

        # elif config['Normalization'] == 'softmax':

        #     w1 = torch.tensor([i for i in range(1,table_num)])
        #     w1 = sp.softmax(w1)[join_num-1].item()
        #     w2 = torch.tensor([0,0,0,0,4,5,6,7,8,9,10,11,12,0,14,0,0,17])
        #     w2 = sp.softmax(w2)[table_num].item()

        # priority = (1 + config['Lambda1']*w1) * (1 + config['Lambda2']*w2)
        priority = 1.

        if not done:
            agent_a.memory.push(priority, old_obs,  action_num, obs, cost, done)     #old_obs: mask,db , tree ,link_mtx,all_table_embed
                                                                          #    obs: mask,db , tree
        else:
            agent_a.memory.push(priority, old_obs,  action_num, None, cost, done)



        if done is True:
            loss0 = agent_a.optimize_model()
             # print("loss",loss0)
            loss_0 += loss0
            if epi % config['TARGET_UPDATE'] == 0:
                agent_a.target_net.load_state_dict(agent_a.policy_net.state_dict())
            if epi % config['Test'] ==0:
                logger.info('Episode {}, Loss : {:.4f}'.format(epi, loss_0))
                loss_0 = 0.0
                with torch.no_grad():
                    avg_rows, avg_cost = eval(agent_a, epi, eval_env, device, log_dir)
                logger.info('Episode {}/{}, Avg Rows {}, Avg Cost {}'.format(i, epi, avg_rows, avg_cost))
            # if  epi % 200 ==0:
            #     PATH = r'/data/ygy/code_list/join_mod/save_model_dict/' + str(epi) + '_' + str(
            #         config['BATCH_SIZE']) + '_' + str(model_name)+ str(policy_name) + '_para.pth'
            #     torch.save({'policy_net_state_dict': agent_a.policy_net.state_dict(),
            #                 'optimizer': agent_a.optimizer.state_dict()}, PATH)
            obs, table_num = env.reset()

            first_ep = True
            epi += 1

def eval(agent_a, episodes, env, device, log_dir):
    test_num = env.wrapped.num_test
    estimate_rows = []
    estimate_cost = []
    for idx in range(test_num):
        obs, table_num = env.reset()
        done = False
        while done is False:
            action_num = agent_a.select_action(obs, False)
            

            obs, cost, done, _, _ = env.step(action_num)
            # print(action_num, cost)

        if done is True:
            sql = env.wrapped.hint_sql
            sql_id = env.wrapped.query.sql_id
            estimatedRows, estimatedcosts = get_cost_rows(sql)
            estimate_rows.append(float(estimatedRows))
            estimate_cost.append(float(estimatedcosts))

            file_name = os.path.join(log_dir, 'test_episodes_{}.txt'.format(episodes))
            with open(file_name, mode='a') as f:
                f.write(sql_id+'|'+sql+'estimatedRows:'+estimatedRows+ '|' + 'estimatedcosts:'+estimatedcosts+'\n')

    return sum(estimate_rows)/len(estimate_rows), sum(estimate_cost)/len(estimate_cost)
            
def get_cost_rows(sql):
    #print(sql)
    cursor = pg_utils.init_pg()
    cursor.execute(""" EXPLAIN """ + sql)
    #print(sql)
    rows = cursor.fetchall()

    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedRows = row0[1].replace("rows=", "")

    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedcosts = row0[0].split("..")[1]
    return estimatedRows, estimatedcosts

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('-b', '--BATCH_SIZE', default=16, type=int, help='BATCH_SIZE for training')
    parser.add_argument('-f', '--fold', default=2, type=int, help='data file')
    parser.add_argument('-ed', '--emb_dim', default=16, type=int)
    parser.add_argument('-e', '--episodes', default=50000, type=int)
    parser.add_argument('-vis', '--CUDA_VISIBLE_DEVICES', default='1', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--log_path', default='logs', type=str)
    parser.add_argument('-s', '--SEED', default=10, type=int)
    parser.add_argument('-p', '--policy_name', default='DQN', type=str)
    parser.add_argument('-lr', '--learning_rate', default=3e-3, type=float)
    parser.add_argument('-ga', '--GAMMA', default=0.999, type=float)
    parser.add_argument('-wd', '--weight_decay', default=0.01, type=float)

    config = vars(parser.parse_args())
    config = dict(**config, **POLICY_CONFIG, **MODEL_CONFIG)
    model_name = 'DQ'
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
    policy_name = config['policy_name']  # "DQN" # can modify here

    import os, datetime
    if not os.path.exists(config['log_path']):
        os.makedirs(config['log_path'])

    pg_utils.setup_seed(config["SEED"])

    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((model_name, policy_name, timestamp))
    log_dir = os.path.join(config['log_path'], loglogs)
    os.makedirs(log_dir)
    log_file_name = os.path.join(log_dir, "running_log")
    logger = pg_utils.get_logger(log_file_name)
    logger.info(config)
    main(config, log_dir)

