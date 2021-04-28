from smac.env import StarCraft2Env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os
from datetime import datetime
from .learner_mix import Learner
from .controller import Controller
from .episode_buffer import EpisodeBuffer
from .runnner import Runner
from writter_util import WritterUtil

class Experiment:
    def __init__(self, args):        
        self.args = args


    def save(self):

        run_state = {}                
        run_state['episode'] = self.e
        run_state['step'] = self.step
        run_state['best_win_rate'] = self.best_win_rate
        run_state['learner'] = self.learner.state_dict()
        run_state['buffer'] = self.buffer.state_dict()
        torch.save(run_state, self.path_checkpt)        

    def load(self):
        run_state = torch.load(self.path_checkpt)
        self.e = run_state['episode']
        self.step = run_state['step']
        self.best_win_rate = run_state['best_win_rate']
        self.learner.load_state_dict(run_state['learner'])                
        self.buffer.load_state_dict(run_state['buffer'])
        self.result = np.load(self.path_result)

    def start(self):
        args = self.args

        path_checkpt = 'checkpoints'
        path_result = 'results'
        path_model = 'models'
        if not os.path.exists(path_checkpt):
            os.mkdir(path_checkpt)
        if not os.path.exists(path_result):
            os.mkdir(path_result)
        if not os.path.exists(path_model):
            os.mkdir(path_model)
        path_checkpt = os.path.join(path_checkpt, args.run_name + '.tar')
        path_result = os.path.join(path_result, args.run_name + '.npy')
        path_model = os.path.join(path_model, args.run_name + '.tar')
            
        env = StarCraft2Env(map_name=args.map_name, window_size_x=640, window_size_y=480)
        env_info = env.get_env_info()

        args.n_agents = env_info["n_agents"]
        args.n_actions = env_info["n_actions"]
        args.obs_dim = env_info['obs_shape']
        args.input_dim = args.obs_dim
        if args.agent_id:
            args.input_dim += args.n_agents
        if args.last_action:        
            args.input_dim += args.n_actions        
        args.state_dim = env_info['state_shape']
        args.episode_limit = env_info['episode_limit']
        
        writter = SummaryWriter('runs/'+ args.run_name + '/' + datetime.now().strftime('%Y-%m-%d,%H%M%S'))
        w_util = WritterUtil(writter,args)
        learner = Learner(w_util,args)               
        ctrler = Controller(learner.sys_actor,args)
        runner = Runner(env,ctrler,args)
                
        scheme = {}
        n_agents = args.n_agents
        scheme['obs'] = {'shape':(n_agents, args.obs_dim), 'dtype': torch.float32}
        scheme['valid'] = {'shape':(), 'dtype': torch.int32}
        scheme['actions'] = {'shape':(n_agents,), 'dtype': torch.int32}
        scheme['avail_actions'] = {'shape':(n_agents, args.n_actions), 'dtype': torch.int32}
        scheme['reward'] = {'shape':(), 'dtype': torch.float32}
        scheme['state'] = {'shape':(args.state_dim,), 'dtype': torch.float32}

        buffer = EpisodeBuffer(scheme, args)
        result = np.zeros((3, args.n_steps // args.test_every_step))

        self.e = 0
        self.step = 0
        self.best_win_rate = 0                   
        self.path_checkpt = path_checkpt
        self.path_result = path_result
        self.path_model = path_model      
        self.env = env        
        self.writter = writter
        self.w_util = w_util
        self.ctrler = ctrler
        self.runner = runner
        self.learner = learner
        self.buffer = buffer
        self.result = result

        if args.continue_run:
            self.load()
                    
        
    def run(self):
        
        args = self.args
        buffer = self.buffer        
        runner = self.runner
        learner = self.learner
        w_util = self.w_util
        
        while self.step < args.n_steps:
        #for self.e in range(self.e, args.n_episodes+1):
            self.e += 1            
            data, episode_reward, win_tag, step =  runner.run()
            old_step = self.step
            self.step += step                   
            w_util.set_step(self.step)            
            w_util.WriteScalar('train/reward', episode_reward)
            print("Episode {}, step {}, win = {}, reward = {}".format(self.e, self.step, win_tag, episode_reward))
            buffer.add_episode(data)
            data = buffer.sample(args.n_batch, args.top_n)
            if data:
                loss = learner.train(data)                        
                        
            if self.step // args.test_every_step != old_step // args.test_every_step:
                self.test_model()            

            if args.save_every and self.e % args.save_every == 0:                
                self.save()

        self.env.close()

    def test_model(self):
        args = self.args
        w_util = self.w_util
        runner = self.runner
        result = self.result

        win_count = 0
        reward_avg = 0
        for i in range(args.test_count):
            _, episode_reward, win_tag, _ =  runner.run(test_mode=True)
            if win_tag:
                win_count += 1
            reward_avg += episode_reward                
        win_rate = win_count/args.test_count        
        reward_avg /= args.test_count
        w_util.WriteScalar('test/reward', reward_avg)
        w_util.WriteScalar('test/win_rate', win_rate)
        result[:, self.step // args.test_every_step - 1] = [self.e, self.step, win_rate]
        np.save(self.path_result, self.result)
        if win_rate >= self.best_win_rate:
            self.best_win_rate = win_rate
            torch.save(self.learner.sys_actor.state_dict(), self.path_model)            
        print('Test reward = {}, win_rate = {}'.format(reward_avg, win_rate))