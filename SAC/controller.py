import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


#n_batch = 1


class Controller:
    def __init__(self, sys_agent, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.msg_dim = args.msg_dim
        self.epsilon_st = args.epsilon_st
        self.epsilon_ed = args.epsilon_ed
        self.eps_half_time = args.eps_half_time
        self.epsilon = args.epsilon_st
        self.device = args.device
        self.explore_type = args.explore_type
        self.sys_agent_src = sys_agent
        self.sys_agent = type(sys_agent)(args)        
        self.sys_agent.eval()
        self.episode = 0

    def new_episode(self):
        state_dict = self.sys_agent_src.state_dict()
        self.sys_agent.load_state_dict(state_dict)        
        self.hiddens = self.sys_agent.init_hiddens(1)        
        self.episode += 1

        T = self.episode/self.eps_half_time
        self.epsilon = self.epsilon_st * (0.5 ** T)
        if self.epsilon < self.epsilon_ed:
            self.epsilon = self.epsilon_ed

    def get_actions(self, states, avail_actions, explore=False):
                
        epsilon = self.epsilon
        states = torch.as_tensor(states).unsqueeze(0)
        avail_actions = torch.as_tensor(avail_actions).unsqueeze(0)
        
        with torch.no_grad():
            ps, acts, hs_next = self.sys_agent.forward(states,avail_actions, self.hiddens)
        self.hiddens = hs_next

        if explore:
            actions = acts[0]
        else:
            actions = torch.argmax(ps[0],-1)
        return actions.numpy()

    def one_hot(self, tensor, n_classes):
        return F.one_hot(tensor.to(dtype=torch.int64), n_classes).to(dtype=torch.float32)
