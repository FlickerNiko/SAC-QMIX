import torch
import torch.nn as nn
import torch.nn.functional as F
from .rnn_agent import RNNAgent
from .message_hub import MessageHub



class MQAgent(nn.Module):
    def __init__(self, args):
        super(MQAgent, self).__init__()
        self.n_agents = args.n_agents
        self.msg_dim = args.msg_dim
        self.agents = [RNNAgent(args) for i in range(self.n_agents)]
        self.msg_hub = MessageHub(args)
        #self.m_zero = torch.zeros(1, args.msg_dim)

    def init_hiddens(self, n_batch):
        hiddens = [agent.init_hidden(n_batch) for agent in self.agents]
        hiddens = torch.stack(hiddens,1)
        return hiddens

    def forward(self, states, actions_last, hiddens):
        
        ms_send = []
        n_batch = len(states)
        m_zero = torch.zeros(n_batch, self.msg_dim)
        for i in range(self.n_agents):
            ms_send.append(self.agents[i].forward(states[:,i], m_zero,actions_last[:,i] , hiddens[:,i])[1])

        ms_recv = self.msg_hub(ms_send)

        qs = []
        hs_next = []

        for i in range(self.n_agents):
            q, _, h_next = self.agents[i].forward(states[:,i], ms_recv[i], actions_last[:,i] , hiddens[:,i])
            qs.append(q)
            hs_next.append(h_next)

        qs = torch.stack(qs,1)
        hs_next = torch.stack(hs_next,1)
        return qs, hs_next