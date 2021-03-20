import torch
import torch.nn as nn
import torch.nn.functional as F
from .actor import Actor

class VDNActor(nn.Module):
    def __init__(self, args):
        super(VDNActor, self).__init__()
        self.n_agents = args.n_agents        

        self.input_dim = args.input_dim        
        self.n_actions = args.n_actions
        
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.hidden_dim = args.hidden_dim
        self.head_num = args.head_num
        self.fc1 = nn.Linear(self.input_dim, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.hidden_dim)
        self.mha = nn.MultiheadAttention(self.hidden_dim,self.head_num)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.n_actions)


    def init_hiddens(self, n_batch):
        hiddens = self.fc1.weight.new_zeros(n_batch*self.n_agents, self.rnn_hidden_dim)
        hiddens = hiddens.reshape(n_batch,self.n_agents,-1)
        return hiddens

    def forward(self, states, avail_actions, hiddens):
        n_batch = states.shape[0]
        
        states = states.reshape(-1,states.shape[-1])
        hiddens = hiddens.reshape(-1,hiddens.shape[-1])
        avail_actions = avail_actions.reshape(-1,self.n_actions)
        h_last = hiddens
        x = states

        h = F.relu(self.fc1(x))        
        h = self.rnn(h, h_last)

        h_next = h.reshape(-1,self.n_agents,self.rnn_hidden_dim)
        h = self.fc2(h)
        
        h = h.reshape(-1,self.n_agents,self.hidden_dim)
        h = h.transpose(0,1)

        h += self.mha(h,h,h)[0]

        h = h.transpose(0,1)

        h = h.reshape(-1,self.hidden_dim)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        h[avail_actions == 0] = -float('inf')
        
        #h[blocked] = -1e38
        h = torch.softmax(h,-1)        
        y = h.reshape(n_batch,self.n_agents,self.n_actions)                
        return y, h_next

