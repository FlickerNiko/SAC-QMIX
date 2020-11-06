import torch
import torch.nn as nn
import torch.nn.functional as F

class JALAgent2(nn.Module):
    def __init__(self, args):
        
        super(JALAgent2, self).__init__()

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim

        self.fc1 = nn.Linear((self.input_dim+self.n_actions*2)*self.n_agents, self.hidden_dim)        
        self.rnn = nn.GRUCell(self.hidden_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_actions*self.n_agents)        
        


    def init_hiddens(self, n_batch):
        return self.fc1.weight.new_zeros(n_batch, self.hidden_dim)
        

    def forward(self, states, actions_explore, actions_last, hiddens):
        qs = []
        for i in range(self.n_agents):
            actions_explore_i = actions_explore.clone()
            actions_explore_i[:,i].zero_()


            q,_ = self._forward(states,actions_explore_i,actions_last,hiddens)
            qs.append(q[:,i])
        qs = torch.stack(qs,1)
        _, hs = self._forward(states,torch.zeros_like(actions_explore),actions_last,hiddens)
        return qs, hs

    def _forward(self,states, actions_explore, actions_last, hiddens):

        x = torch.cat([states,actions_explore, actions_last],dim=2)
        x = x.view(-1, (self.input_dim+self.n_actions*2)*self.n_agents) 
        x = F.relu(self.fc1(x))    
        h = self.rnn(x,hiddens)
        q = self.fc2(h)
        q = q.view(-1,self.n_agents,self.n_actions)
        return q,h