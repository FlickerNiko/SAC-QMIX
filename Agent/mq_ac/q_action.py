import torch
import torch.nn as nn
import torch.nn.functional as F


class QAction(nn.Module):
    def __init__(self, args):
        super(QAction, self).__init__()
        self.n_actions = args.n_actions
        self.msg_dim = args.msg_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.fc1 =  nn.Linear(self.msg_dim, self.hidden_dim)
        self.fc2 =  nn.Linear(self.hidden_dim, self.n_actions)
    
    def forward(self, messages):
        x = messages.view(-1,self.n_agents * self.msg_dim)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)        
        return q