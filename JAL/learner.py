import torch
import torch.nn as nn
import torch.nn.functional as F
from .jal_agent import JALAgent

class Learner:
    def __init__(self, jal_agent:JALAgent, args):
        self.device = args.device
        
        self.jal_agent = jal_agent
        self.jal_agent_tar = JALAgent(args)
        if self.device == 'cuda':
            self.jal_agent_tar.cuda()

        self.jal_agent_tar.requires_grad_(False)
        self.update_target()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.lr = args.lr
        self.target_update = args.target_update
        self.step = 0

        self.optimizer = torch.optim.Adam(self.jal_agent.parameters(), lr = self.lr)
    
    def train(self, data):

        self.step += 1
        n_batch = data['obs'].shape[0]
        T = data['obs'].shape[1]

        obs = data['obs'].to(device=self.device)
        actions = data['actions'].to(device=self.device)
        reward = data['reward'].to(device=self.device)
        valid = data['valid'].to(device=self.device)
        avail_actions = data['avail_actions'].to(device=self.device)
        hiddens = self.jal_agent.init_hiddens(n_batch)
        hiddens_tar = self.jal_agent_tar.init_hiddens(n_batch)

        Qs = []
        Qs_tar = []
        qs = []
        qs_tar = []

        agent_shape = [n_batch,T] + [self.n_actions]*self.n_agents
        j_actions = actions.new_zeros(n_batch,T)
        avail_ja = obs.new_ones(agent_shape)
        for i in range(self.n_agents):
            view_shape = [1]*self.n_agents
            view_shape[i] = self.n_actions
            view_shape = [n_batch,T] + view_shape
            avail_ja *= avail_actions[:,:,i].view(view_shape)
            j_actions *= self.n_actions
            j_actions += actions[:,:,i]
        
        

        for i in range(T):
            a = j_actions[:,i]
            v = valid[:,i]
            
            hiddens = hiddens * v.view([-1] + [1] * (hiddens.ndim-1))
            hiddens_tar =  hiddens_tar* v.view([-1] + [1] * (hiddens_tar.ndim-1))

            Q, hiddens = self.jal_agent.forward(obs[:,i], None,hiddens)
            Q_tar, hiddens_tar = self.jal_agent_tar.forward(obs[:,i], None, hiddens_tar)

            Q = Q*v.view([-1] + [1] * (Q.ndim-1))            
            Q_tar = Q_tar*v.view([-1] + [1] * (Q_tar.ndim-1))

            Qs.append(Q)
            Qs_tar.append(Q_tar)
            qs.append(self.gather_end(Q, a))
            #a_last = self.action_transform(a,self.n_actions)

        
        Qs = torch.stack(Qs,1)
        Qs_tar = torch.stack(Qs_tar,1)

        Qs = Qs - (1-avail_ja.view(n_batch,T,self.n_actions**self.n_agents))*1e38
        

        for i in range(T-1):
            a = actions[:,i]
            r = reward[:,i]
            
            a_next = torch.argmax(Qs[:,i+1],1)            

            q_next = self.gather_end(Qs_tar[:,i+1],a_next)
            q_tar = r + self.gamma * q_next
            qs_tar.append(q_tar)

        #i = T - 1

        qs_tar.append(reward[:, T-1] + reward.new_zeros(qs_tar[-1].shape))

        qs = torch.stack(qs,1)
        qs_tar = torch.stack(qs_tar,1)

        loss = F.mse_loss(qs,qs_tar)
        self.optimizer.zero_grad()
        loss.backward()       
        self.optimizer.step()

        if self.step % self.target_update == 0:
            self.update_target()

        return loss

    def gather_end(self, input, index):
        index = torch.unsqueeze(index,-1).to(dtype=torch.int64)
        return torch.gather(input, input.ndim -1, index).squeeze(-1) 

    def action_transform(self, actions, n_actions):

        actions = actions.to(dtype = torch.int64)
        shape = actions.shape + (n_actions,)
        actions = actions.unsqueeze(-1)
        output = actions.new_zeros(shape, dtype = torch.float)
        output.scatter_(len(shape)-1, actions, 1)

        return output
    

    def update_target(self):
        self.jal_agent_tar.load_state_dict(self.jal_agent.state_dict())

    



