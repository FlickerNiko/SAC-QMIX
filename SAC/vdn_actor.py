import torch
import torch.nn as nn
import torch.nn.functional as F
from .actor import Actor

class VDNActor(nn.Module):
    def __init__(self, args):
        super(VDNActor, self).__init__()
        self.n_agents = args.n_agents
        self.agent = Actor(args)

    def init_hiddens(self, n_batch):
        hiddens = [self.agent.init_hidden(n_batch) for i in range(self.n_agents)]
        hiddens = torch.stack(hiddens,1)
        return hiddens

    def forward(self, states, avail_actions, hiddens):
        n_batch = states.shape[0]
        ys = []
        acts = []
        hs_next = []
        indices = torch.arange(0,self.n_agents,dtype=torch.int64, device=states.device)    # n_batch, n_agents, n_agents        
        indices = F.one_hot(indices, self.n_agents).to(dtype=torch.float32)
        indices = torch.stack([indices]*n_batch, 0)
        for i in range(self.n_agents):
            y, a, h_next = self.agent.forward(states[:,i], avail_actions[:,i], indices[:,i], hiddens[:,i])
            ys.append(y)
            acts.append(a)
            hs_next.append(h_next)        
        ys = torch.stack(ys,1)
        acts = torch.stack(acts,1)
        hs_next = torch.stack(hs_next,1)
        
        return ys, acts, hs_next

