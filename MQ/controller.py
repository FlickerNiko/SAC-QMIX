import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from .mq_agent import MQAgent

#n_batch = 1


class Controller:
    def __init__(self, mq_agent: MQAgent, args):
        self.args = args
        self.mq_agent = mq_agent
        self.m_zero = torch.zeros(1, self.args.msg_dim)
        self.a_zero = torch.zeros(1, self.args.n_actions)

    def new_episode(self):

        self.hiddens = self.mq_agent.init_hiddens(1)
        self.last_actions = [self.a_zero] * self.args.n_agents

    def get_actions(self, states, avail_actions, explores=None):
        # data: obs, explore
        if explores == None:
            explores = [False] * self.args.n_agents

        
        states = [torch.unsqueeze(torch.as_tensor(state), 0)
                  for state in states]

        qs, hs_next = self.mq_agent.forward(states, self.last_actions, self.hiddens)
        self.hiddens = hs_next

        actions = []
        for q, avail_action, explore in zip(qs, avail_actions, explores):
            avail_action = np.nonzero(avail_action)[0]   #one-hot to index
            avail_q = torch.take(q, torch.tensor(avail_action, dtype = torch.int64))
            if explore:
                a_index = random.randint(0, len(avail_action)-1)
            else:
                a_index = torch.argmax(avail_q)
            a = avail_action[a_index]
            actions.append(a)

        self.last_actions = [self.action_transform(
            [a], self.args.n_actions) for a in actions]
        return actions

    def action_transform(self, action, n_action):

        len_a = len(action)
        output = torch.zeros(len_a, n_action)
        # if action[0]:   # not None
        output[torch.arange(0, len_a, dtype=torch.int64), action] = 1
        return output
