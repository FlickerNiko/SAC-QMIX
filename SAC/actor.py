import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args

        self.input_dim = args.input_dim
        self.msg_dim = args.msg_dim
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.fc1 = nn.Linear(self.input_dim + self.n_agents, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def init_hidden(self, n_batch):
        # make hidden states on same device as model
        return self.fc1.weight.new_zeros(n_batch, self.rnn_hidden_dim)

    def forward(self, inputs, avail_actions, index, h_last):
        x = torch.cat([inputs,index],1)
        h = F.relu(self.fc1(x))        
        h = self.rnn(h, h_last)
        y = self.fc2(h)
        blocked = torch.nonzero(1-avail_actions,as_tuple=True)
        y[blocked] = -1e38
        y = torch.softmax(y,-1)

        #_y = y.clone()

        #_y[blocked] = 0
        
        a = torch.multinomial(y,1).squeeze(1)

        return y,a,h
    

