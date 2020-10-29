import torch
import torch.nn as nn
import torch.nn.functional as F


class MsgHub(nn.Module):
    def __init__(self, args):
        super(MsgHub, self).__init__()
        self.n_agents = args.n_agents
        self.msg_dim = args.msg_dim
        self.hidden_dim = args.hub_hidden_dim
        self.fc1 =  nn.Linear(self.n_agents * self.msg_dim, self.hidden_dim)
        self.fcm =  nn.Linear(self.hidden_dim, self.n_agents * self.msg_dim)
    
    def forward(self, messages):
        x = messages.view(-1,self.n_agents * self.msg_dim)
        #x = torch.cat(messages,1)
        x = F.relu(self.fc1(x))
        #m = F.tanh(self.fcm(x))
        m = self.fcm(x)        
        m_out = m.view(-1,self.n_agents, self.msg_dim)
        return m_out

