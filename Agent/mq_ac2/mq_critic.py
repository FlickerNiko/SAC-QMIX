import torch
import torch.nn as nn
import torch.nn.functional as F


class MQCritic(nn.Module):
    def __init__(self, args):
        super(MQCritic, self).__init__()
        self.n_agents = args.n_agents
        self.msg_dim = args.msg_dim
        self.hidden_dim = args.hub_hidden_dim
        self.fc1 =  nn.Linear(self.n_agents * self.msg_dim *2, self.hidden_dim)
        self.fc2 =  nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 =  nn.Linear(self.hidden_dim, 1)
    
    def forward(self, msg_in, msg_out):
        x1 = msg_in.view(-1,self.n_agents * self.msg_dim)
        x2 = msg_out.view(-1,self.n_agents * self.msg_dim)
        x = torch.cat([x1,x2],1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        q = self.fc3(h)                
        return q

