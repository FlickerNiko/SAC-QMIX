from smac.env import StarCraft2Env
import numpy as np
import MQ
import JAL
import torch
from VDN.vdn_agent import VDNAgent
from episode_buffer import EpisodeBuffer
from runnner import Runner
from Learner.learner import Learner
from Controller.controller import Controller

class Args:
    pass

def main():

    args = Args()

    args.msg_dim = 32
    args.rnn_hidden_dim = 128
    
    env = StarCraft2Env(map_name="3m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    args.n_agents = n_agents
    args.n_actions = n_actions
    args.input_dim = env_info['obs_shape']
    #args.input_dim = env_info['state_shape']
    args.hub_hidden_dim = 128
    args.gamma = 0.99
    args.lr = 1e-3
    args.target_update = 10
    args.epsilon = 0.1
    args.test_every = 10
    args.device = 'cuda'
    args.hidden_dim = 256
    args.solo_explore = True
    sys_agent = MQ.MQAgent(args)
    sys_agent.cuda()

    ctrler = Controller(sys_agent,args)
    runner = Runner(env,ctrler,args)
    learner = Learner(sys_agent, args)
    scheme = {}
    scheme['obs'] = {'shape':(n_agents,args.input_dim), 'dtype': torch.float32}
    #scheme['obs'] = {'shape':(args.input_dim,), 'dtype': torch.float32}
    scheme['valid'] = {'shape':(), 'dtype': torch.int32}
    scheme['actions'] = {'shape':(n_agents,), 'dtype': torch.int32}
    scheme['avail_actions'] = {'shape':(n_agents, n_actions), 'dtype': torch.int32}
    scheme['reward'] = {'shape':(), 'dtype': torch.float32}
    scheme['to_learn'] = {'shape':(n_agents,), 'dtype': torch.int32}
    buffer = EpisodeBuffer(scheme,128, env_info['episode_limit'])
    
    print("Init MQ Success")

    n_episodes = 1000

    for e in range(n_episodes):

        test_mode = True if e % args.test_every == 0 else False
        data,episode_reward =  runner.run(test_mode=test_mode)
        
        buffer.add_episode(data)
        print("Total reward in episode {} = {}".format(e, episode_reward))

        data = buffer.sample(8)
        
        if data:    
            learner.train(data)

    env.close()


main()