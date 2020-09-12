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
        self.mq_agent = mq_agent
        self.m_zero = torch.zeros(1, args.msg_dim)
        self.a_zero = torch.zeros(1, args.n_actions)

    def new_episode(self):

        self.hiddens = self.mq_agent.init_hiddens(1)
        self.last_actions = torch.zeros(1, self.n_agents, self.n_actions)

    def get_actions(self, states, avail_actions, explore=False):
        # data: obs, explore
               
        states = torch.as_tensor(states).unsqueeze(0)
        qs, hs_next = self.mq_agent.forward(states, self.last_actions, self.hiddens)
        self.hiddens = hs_next

        actions = []
        for q, avail_action in zip(qs.transpose(0,1), avail_actions):
            avail_action = np.nonzero(avail_action)[0]   #one-hot to index
            avail_q = torch.take(q, torch.tensor(avail_action, dtype = torch.int64))
            if explore:
                a_index = random.randint(0, len(avail_action)-1)
            else:
                a_index = torch.argmax(avail_q)
            a = avail_action[a_index]
            actions.append(a)

        self.last_actions = self.action_transform(actions, self.n_actions).unsqueeze(0)
        return actions

    def action_transform(self, actions, n_actions):

        actions = torch.as_tensor(actions, dtype = torch.int64)
        shape = actions.shape + (n_actions,)
        actions = actions.unsqueeze(-1)
        output = torch.zeros(shape)

        output.scatter_(len(shape)-1, actions, 1)

        return output
