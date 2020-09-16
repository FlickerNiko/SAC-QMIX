import torch
import torch.nn as nn
import torch.nn.functional as F
from .mq_agent import MQAgent

class Learner:
    def __init__(self, mq_agent: MQAgent, args):
        
        self.device = args.device
        
        self.mq_agent = mq_agent
        self.mq_agent_tar = MQAgent(args)
        if self.device == 'cuda':
            self.mq_agent_tar.cuda()
        self.mq_agent_tar.requires_grad_(False)
        self.update_target()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.lr = args.lr
        self.target_update = args.target_update
        self.step = 0
        self.args = args

        self.optimizer =  torch.optim.Adam(self.mq_agent.parameters(), lr = self.lr)
        
    def train(self, data):
        
        self.step += 1

        n_batch = data['obs'].shape[0]
        T = data['obs'].shape[1]

        obs = data['obs'].to(device=self.device)
        actions = data['actions'].to(device=self.device)
        reward = data['reward'].to(device=self.device)
        valid = data['valid'].to(device=self.device)
        avail_actions = data['avail_actions'].to(device=self.device)
        hiddens = self.mq_agent.init_hiddens(n_batch)
        hiddens_tar = self.mq_agent_tar.init_hiddens(n_batch)
        a_last = torch.zeros(n_batch, self.n_agents, self.n_actions, device=self.device)
        Qs = []
        Qs_tar = []
        qs = []
        qs_tar = []
       
        for i in range(T):
            a = actions[:,i]
            v = valid[:,i]
            
            hiddens *= v.view([-1] + [1] * (hiddens.ndim-1))
            hiddens_tar *= v.view([-1] + [1] * (hiddens_tar.ndim-1))

            Q, hiddens = self.mq_agent.forward(obs[:,i], a_last,hiddens)
            Q_tar, hiddens_tar = self.mq_agent_tar.forward(obs[:,i], a_last, hiddens_tar)

            Q *= v.view([-1] + [1] * (Q.ndim-1))            
            Q_tar *= v.view([-1] + [1] * (Q_tar.ndim-1))

            Qs.append(Q)
            Qs_tar.append(Q_tar)
            qs.append(self.gather_end(Q, a))
            a_last = self.action_transform(a,self.n_actions)

        Qs = torch.stack(Qs,1)
        Qs_tar = torch.stack(Qs_tar,1)

        Qs -= (1-avail_actions)*1e38

        for i in range(T-1):
            a = actions[:,i]
            r = reward[:,i]
            
            a_next = torch.argmax(Qs[:,i+1],2)            
            q_next = self.gather_end(Qs_tar[:,i+1],a_next)
            q_tar = r.unsqueeze(1) + self.gamma * q_next
            qs_tar.append(q_tar)

        #i = T - 1

        qs_tar.append(reward[:, T-1].unsqueeze(1) + reward.new_zeros(qs_tar[-1].shape))

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
        self.mq_agent_tar.load_state_dict(self.mq_agent.state_dict())
            

        

        

        
        