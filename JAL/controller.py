import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from .jal_agent import JALAgent

class Controller:
    def __init__(self, jal_agent:JALAgent, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.device = args.device
        self.jal_agent = jal_agent
        

    def new_episode(self):
        self.hiddens = self.jal_agent.init_hiddens(1)
        self.last_actions = torch.zeros(1, self.n_agents, self.n_actions, device=self.device)

    def get_actions(self, states, avail_actions, explore=False):
        states = torch.as_tensor(states, device=self.device).unsqueeze(0)
        avail_actions = torch.as_tensor(avail_actions, device=self.device)
        qs, hs_next = self.jal_agent.forward(states, None, self.hiddens)
        avail_ja = qs.new_ones([self.n_actions]*self.n_agents)
        for i in range(self.n_agents):
            view_shape = [1]*self.n_agents
            view_shape[-i] = self.n_actions
            avail_ja *= avail_actions[i].view(view_shape)
        
        qs = qs[0].view([self.n_actions]*self.n_agents)  #debatch
        qs -= (1-avail_ja)*1e38

        ja = torch.argmax(qs).item()
        
        actions = []
        for i in range(self.n_agents):
            a = ja%self.n_actions
            actions.append(a)
            ja //= self.n_actions

        actions.reverse()
        self.hiddens = hs_next

        return actions
        
        