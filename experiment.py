from smac.env import StarCraft2Env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os
from Agent import VDNAgent,MQAgent,JALAgent,MQAgent2,JALAgent2,MQAgent3
from episode_buffer import EpisodeBuffer
from runnner import Runner
from learner import Learner
from controller import Controller
from writter_util import WritterUtil

class Experiment:
    def __init__(self, args):        
        self.args = args


    def save(self):
        run_state = {}
        run_state['args'] = self.args.__dict__
        run_state['model_state'] = self.sys_agent.state_dict()
        run_state['optim_state'] = self.learner.optimizer.state_dict()
        run_state['episode'] = self.e            
        run_state['buffer_state'] = self.buffer.state_dict()
        torch.save(run_state, self.path_checkpt)


    def start(self):
        args = self.args
        path_checkpts = 'checkpoints'
        if not os.path.exists(path_checkpts):
            os.mkdir(path_checkpts)
        path_checkpt = os.path.join(path_checkpts, args.run_name+ '.tar')
        if not args.new_run:
            run_state = torch.load(path_checkpt)
            args.__dict__.update(run_state['args'])
            
        env = StarCraft2Env(map_name=args.map_name)
        env_info = env.get_env_info()

        n_actions = env_info["n_actions"]
        n_agents = env_info["n_agents"]
        
        args.n_agents = n_agents
        args.n_actions = n_actions
        args.input_dim = env_info['obs_shape']
        args.episode_limit = env_info['episode_limit']
        if args.model == 'mq':
            sys_agent = MQAgent(args)
        elif args.model == 'mq2':
            sys_agent = MQAgent2(args)
        elif args.model == 'mq3':
            sys_agent = MQAgent3(args)
        elif args.model == 'jal':
            sys_agent = JALAgent(args)
        else:
            sys_agent = VDNAgent(args)

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
        
        e = 0
        if not args.new_run:
            e = run_state['episode'] + 1
            sys_agent.load_state_dict(run_state['model_state'])
            learner.optimizer.load_state_dict(run_state['optim_state'])
            buffer.load_state_dict(run_state['buffer_state'])

        self.path_checkpt = path_checkpt
        self.e = e
        self.env = env
        self.sys_agent = sys_agent
        self.writter = writter
        self.w_util = w_util
        self.ctrler = ctrler
        self.runner = runner
        self.learner = learner
        self.buffer = buffer
        

        
    def run(self):
        e = self.e
        args = self.args
        buffer = self.buffer
        sys_agent = self.sys_agent
        runner = self.runner
        learner = self.learner
        w_util = self.w_util


        loss = None
        for e in range(e ,args.n_episodes):
            
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
                w_util.WriteScalar('test/reward', reward_avg, e)
                w_util.WriteScalar('test/win_rate', win_rate, e)
                
                print('Test reward = {}, win_rate = {}'.format(reward_avg, win_rate))

            if e % args.log_every == 0:
                w_util.WriteModel('model', sys_agent, e)
                self.e = e
                self.save()
        self.env.close()