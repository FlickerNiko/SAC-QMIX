import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from .mq_agent import MQAgent

#n_batch = 1


class Controller:
    def __init__(self, mq_agent: MQAgent, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.device = args.device
        self.mq_agent = mq_agent
        self.m_zero = torch.zeros(1, args.msg_dim, device=self.device)
        self.a_zero = torch.zeros(1, args.n_actions, device=self.device)

    def new_episode(self):

        self.hiddens = self.mq_agent.init_hiddens(1)
        self.last_actions = torch.zeros(1, self.n_agents, self.n_actions, device=self.device)

    def get_actions(self, states, avail_actions, explore=False):
        # data: obs, explore
               
        states = torch.as_tensor(states, device=self.device).unsqueeze(0)
        avail_actions = torch.as_tensor(avail_actions,device=self.device)
        qs, hs_next = self.mq_agent.forward(states, self.last_actions, self.hiddens)
        self.hiddens = hs_next

        #actions = torch.zeros(self.n_agents,dtype=torch.int64)
        actions = []
        
        for q, avail_action in zip(qs[0], avail_actions):
            avail_index = torch.nonzero(avail_action)[:,0]
            avail_q = q.gather(0,avail_index) #q:(n_actions,)

            if explore:
                a_index = torch.randint(0, len(avail_index),(),device=self.device)
            else:
                a_index = torch.argmax(avail_q)
            a = avail_index[a_index]
            
            actions.append(a)

        actions = torch.stack(actions)

        self.last_actions = self.action_transform(actions, self.n_actions).unsqueeze(0)
        return actions.to(device='cpu').numpy()

    def action_transform(self, actions, n_actions):

        #actions = actions.to(dtype=torch.int64)
        shape = actions.shape + (n_actions,)
        actions = actions.unsqueeze(-1)
        output = torch.zeros(shape, device=self.device)
        output.scatter_(len(shape)-1, actions, 1)

        return output
