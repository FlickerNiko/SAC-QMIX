import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from smac.env import StarCraft2Env

class Runner:
    def __init__(self, env, controller, args):        
        self.controller = controller
        self.env = env
        
    def run(self, test_mode=False):
        
        terminated = False
        episode_reward = 0
        data = {'obs':[],'valid':[],'actions':[],'avail_actions':[],'explores':[],'reward':[]}

        self.env.reset()        
        self.controller.new_episode()
        
        while not terminated:
            
            obs = self.env.get_obs()
            #obs = self.env.get_state()
            avail_actions = self.env.get_avail_actions()
            explore = False
            
            if not test_mode:
                explore = True
                
            actions, explores = self.controller.get_actions(obs, avail_actions, explore)
            reward, terminated, _ = self.env.step(actions)
            episode_reward += reward

            data['obs'].append(obs)
            data['valid'].append(1)
            data['actions'].append(actions)
            data['avail_actions'].append(avail_actions)
            data['explores'].append(explores)
            data['reward'].append(reward)

        return data, episode_reward
            
