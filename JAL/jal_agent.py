import torch
import torch.nn as nn
import torch.nn.functional as F

class JALAgent(nn.Module):
    def __init__(self, args):
        
        super(JALAgent, self).__init__()

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim

        self.fc1 = nn.Linear((self.input_dim+self.n_actions)*self.n_agents, self.hidden_dim)
        #self.fc1 = nn.Linear(self.input_dim + self.n_actions*self.n_agents, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_actions*self.n_agents)        
        


    def init_hiddens(self, n_batch):
        return self.fc1.weight.new_zeros(n_batch, self.hidden_dim)
        

    def forward(self, states, actions_last, hiddens):
        

        #a = actions_last.view(-1, self.n_agents*self.n_actions)
        #x = torch.cat([states, a], dim=1)
        x = torch.cat([states,actions_last],dim=2)
        x = x.view(-1, (self.input_dim+self.n_actions)*self.n_agents)        
        x = F.relu(self.fc1(x))
        
        h = self.rnn(x,hiddens)
        q = self.fc2(h)
        q = q.view(-1,self.n_agents,self.n_actions)
        return q, h