import torch
import torch.nn as nn
import torch.nn.functional as F
from .rnn_agent import RNNAgent
from .message_hub import MessageHub



class MQAgent(nn.Module):
    def __init__(self, args):
        super(MQAgent, self).__init__()
        self.n_agents = args.n_agents
        self.agents = [RNNAgent(args) for i in range(self.n_agents)]
        self.msg_hub = MessageHub(args)
        self.m_zero = torch.zeros(1, args.msg_dim)

    def init_hiddens(self, n_batch):
        return [agent.init_hidden(n_batch) for agent in self.agents]

    def forward(self, states, actions_last, hiddens):
        ms_send = []
        for agent, state, a_last, hidden in zip(self.agents, states, actions_last, hiddens):
            ms_send.append(agent.forward(state, self.m_zero, a_last, hidden)[1])
        ms_recv = self.msg_hub(ms_send)

        qs = []
        hs_next = []

        for agent, state, message, a_last, hidden in zip(self.agents, states, ms_recv, actions_last, hiddens):
            q, _, h_next = agent.forward(state, message, a_last, hidden)
            qs.append(q)
            hs_next.append(h_next)

        return qs, hs_next