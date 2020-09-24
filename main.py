from smac.env import StarCraft2Env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from Agent import VDNAgent,MQAgent,JALAgent
from episode_buffer import EpisodeBuffer
from runnner import Runner
from learner import Learner
from controller import Controller
from writter_util import WritterUtil

class Args:
    pass

def main(args):

      
    env = StarCraft2Env(map_name=args.map_name)
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    
    args.n_agents = n_agents
    args.n_actions = n_actions
    args.input_dim = env_info['obs_shape']
    args.episode_limit = env_info['episode_limit']
    sys_agent = MQAgent(args)
    if args.device == 'cuda':        
        sys_agent.cuda()
    
    writter = SummaryWriter('runs/'+ args.run_name)
    w_util = WritterUtil(writter,args)

    ctrler = Controller(sys_agent,args)
    runner = Runner(env,ctrler,args)
    learner = Learner(sys_agent, args)
    scheme = {}
    scheme['obs'] = {'shape':(n_agents,args.input_dim), 'dtype': torch.float32}
    scheme['valid'] = {'shape':(), 'dtype': torch.int32}
    scheme['actions'] = {'shape':(n_agents,), 'dtype': torch.int32}
    scheme['avail_actions'] = {'shape':(n_agents, n_actions), 'dtype': torch.int32}
    scheme['reward'] = {'shape':(), 'dtype': torch.float32}
    scheme['explores'] = {'shape':(n_agents,), 'dtype': torch.int32}
    scheme['learns'] = {'shape':(n_agents,), 'dtype': torch.int32}

    buffer = EpisodeBuffer(scheme, args)
    
    loss = None
    for e in range(args.n_episodes):

        
        data, episode_reward =  runner.run()        
        buffer.add_episode(data)        
        data = buffer.sample(args.n_batch)        
        w_util.WriteScalar('train/reward', episode_reward, e)
        if data:
            loss = learner.train(data)                        
            w_util.WriteScalar('train/loss', loss, e)
        print("Episode {}, reward = {}， loss = {}".format(e, episode_reward, loss))
        if e % args.test_every == 0:
            win_count = 0
            reward_avg = 0
            for i in range(args.test_count):
                _, episode_reward =  runner.run(test_mode=True)
                if episode_reward > 18:
                    win_count += 1
                reward_avg += episode_reward                
            win_rate = win_count/args.test_count
            reward_avg /= args.test_count
            w_util.WriteScalar('test/reward_avg', reward_avg, e)
            w_util.WriteScalar('test/win_rate', win_rate, e)
            
            print('Test reward = {}, win_rate = {}'.format(reward_avg, win_rate))

        if e % args.log_every == 0:
            w_util.WriteModel('model', sys_agent, e)
            state_dict = {}
            state_dict['args'] = args
            state_dict['model_state'] = sys_agent.state_dict()
            state_dict['optim_state'] = learner.optimizer.state_dict()
            state_dict['episode'] = e            
            state_dict['buffer_state'] = buffer.state_dict()
            torch.save(state_dict, 'checkpoints/'+args.run_name+'.tar')

    env.close()

if __name__ == "__main__":

    args = Args()
    args.map_name = '3m'
    args.msg_dim = 32
    args.rnn_hidden_dim = 128
    args.hub_hidden_dim = 128
    args.gamma = 0.99
    args.lr = 1e-3
    args.l2 = 0
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
    args.buffer_size = 256
    args.log_every = 20
    args.log_num = None
    args.n_episodes = 50
    args.run_name = 'test_checkpt'
    args.start_type = 'new'   #new continue
    main(args)