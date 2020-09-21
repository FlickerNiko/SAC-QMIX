import torch
import torch.nn as nn
import torch.nn.functional as F
from .rnn_agent import RNNAgent

class VDNAgent(nn.Module):
    def __init__(self, args):
        super(VDNAgent, self).__init__()
        self.n_agents = args.n_agents
        self.device = args.device
        self.agent = RNNAgent(args)

    def init_hiddens(self, n_batch):
        hiddens = [self.agent.init_hidden(n_batch) for i in range(self.n_agents)]
        hiddens = torch.stack(hiddens,1)
        return hiddens

    def forward(self, states, actions_last, hiddens):
        n_batch = states.shape[0]
        qs = []
        hs_next = []
        indices = torch.arange(0,self.n_agents,dtype=torch.int64, device=self.device)    # n_batch, n_agents, n_agents        
        indices = F.one_hot(indices, self.n_agents).to(dtype=torch.float32)
        indices = torch.stack([indices]*n_batch, 0)
        for i in range(self.n_agents):
            q, h_next = self.agent.forward(states[:,i],indices[:,i], hiddens[:,i])
            qs.append(q)
            hs_next.append(h_next)
        
        qs = torch.stack(qs,1)
        hs_next = torch.stack(hs_next,1)
        return qs, hs_next

