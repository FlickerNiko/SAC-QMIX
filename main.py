from smac.env import StarCraft2Env
import numpy as np
import torch
from Agent import VDNAgent,MQAgent,JALAgent
from episode_buffer import EpisodeBuffer
from runnner import Runner
from learner import Learner
from controller import Controller

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
    args.test_count = 10
    args.device = 'cuda'
    args.hidden_dim = 512
    args.explore_type = 'independent'    #solo, independent, sync
    args.learn_mask = False
    args.agent_index = True
    args.explore_action = True
    args.n_batch = 8
    args.len_buffer = 256
    sys_agent = MQAgent(args)
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
    scheme['explores'] = {'shape':(n_agents,), 'dtype': torch.int32}
    scheme['learns'] = {'shape':(n_agents,), 'dtype': torch.int32}

    buffer = EpisodeBuffer(scheme,args.len_buffer, env_info['episode_limit'])
    
    print("Init MQ Success")

    n_episodes = 1000
    loss = None
    for e in range(n_episodes):

        
        data, episode_reward =  runner.run()        
        buffer.add_episode(data)        
        data = buffer.sample(args.n_batch)
        
        if data:    
            loss = learner.train(data)
        print("Episode {}, reward = {}， loss = {}".format(e, episode_reward, loss))
        if e % args.test_every == 0:
            for i in range(args.test_count):
                _, episode_reward =  runner.run(test_mode=True)        
                print("Test reward {}".format(episode_reward))
    env.close()


main()