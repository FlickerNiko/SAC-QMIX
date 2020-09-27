import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageHub(nn.Module):
    def __init__(self, args):
        super(MessageHub, self).__init__()
        self.n_agents = args.n_agents
        self.msg_dim = args.msg_dim
        self.hidden_dim = args.hub_hidden_dim
        self.fc1 = nn.Linear(self.n_agents * self.msg_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_agents * self.msg_dim)
    
    def init_hidden(self, n_batch):
        # make hidden states on same device as model
        return self.fc1.weight.new_zeros(n_batch, self.hidden_dim)

    def forward(self, messages, hiddens):
        x = messages.view(-1,self.n_agents * self.msg_dim)        
        x = F.relu(self.fc1(x))
        h = self.rnn(x,hiddens)
        m = F.tanh(self.fc2(h))
        m_out = m.view(-1,self.n_agents, self.msg_dim)
        return m_out, h

