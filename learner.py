import torch
import torch.nn as nn
import torch.nn.functional as F


class Learner:
    def __init__(self, sys_agent, args):
        
        self.device = args.device
        
        self.sys_agent = sys_agent
        self.sys_agent.train()
        self.sys_agent_tar = type(sys_agent)(args)
        if self.device == 'cuda':
            self.sys_agent_tar.cuda()
        self.sys_agent_tar.requires_grad_(False)
        self.update_target()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.lr = args.lr
        self.l2 = args.l2
        self.target_update = args.target_update
        self.step = 0
        self.learn_mask = args.learn_mask
        self.action_explore = args.action_explore
        self.args = args

        self.optimizer = torch.optim.Adam(self.sys_agent.parameters(), lr = self.lr, weight_decay=self.l2)
        
    def train(self, data):
        
        self.step += 1
        
        obs = data['obs'].to(device=self.device)
        actions = data['actions'].to(device=self.device)
        reward = data['reward'].to(device=self.device)
        valid = data['valid'].to(device=self.device)
        avail_actions = data['avail_actions'].to(device=self.device)
        explores = data['explores'].to(device=self.device)
        learns = data['learns'].to(device=self.device)

        n_batch = obs.shape[0]
        T = obs.shape[1]

        hiddens = self.sys_agent.init_hiddens(n_batch)
        hiddens_tar = self.sys_agent_tar.init_hiddens(n_batch)
        a_last = torch.zeros(n_batch, self.n_agents, self.n_actions, device=self.device)
        Qs = []
        Qs_tar = []
        qs = []
        qs_tar = []
        
        actions_onehot = self.one_hot(actions,self.n_actions)
        if self.action_explore:
            actions_explore = actions_onehot*explores.unsqueeze(-1)
        else:
            actions_explore = torch.zeros_like(actions_onehot)
        ae_zero = torch.zeros_like(actions_explore)        

        for i in range(T):
            #a = actions[:,i]
            #v = valid[:,i]
            
            #hiddens = hiddens * v.view([-1] + [1] * (hiddens.ndim-1))
            #hiddens_tar = hiddens_tar * v.view([-1] + [1] * (hiddens_tar.ndim-1))

            Q, hiddens = self.sys_agent.forward(obs[:,i], actions_explore[:,i], a_last,hiddens)
            Q_tar, hiddens_tar = self.sys_agent_tar.forward(obs[:,i], ae_zero[:,i], a_last, hiddens_tar)

            Qs.append(Q)
            Qs_tar.append(Q_tar)
            a_last = actions_onehot[:,i]

        Qs = torch.stack(Qs,1)
        Qs_tar = torch.stack(Qs_tar,1)

        Qs -= (1-avail_actions)*1e38
        Qs *= valid.view(list(valid.shape) + [1] * (Qs.ndim - valid.ndim))
        Qs_tar *= valid.view(list(valid.shape) + [1] * (Qs.ndim - valid.ndim))
        qs = self.gather_end(Qs,actions)
        

        for i in range(T-1):
            #a = actions[:,i]
            r = reward[:,i]
            
            a_next = torch.argmax(Qs[:,i+1],2)            
            q_next = self.gather_end(Qs_tar[:,i+1],a_next)
            q_tar = r.unsqueeze(1) + self.gamma * q_next
            qs_tar.append(q_tar)

        #i = T - 1

        qs_tar.append(reward[:, T-1].unsqueeze(1) + reward.new_zeros(qs_tar[-1].shape))

        qs_tar = torch.stack(qs_tar,1)

        if self.learn_mask:
            qs *= learns
            qs_tar *= learns

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

    def one_hot(self, tensor, n_classes):
        return F.one_hot(tensor.to(dtype=torch.int64), n_classes).to(dtype=torch.float32)

    def update_target(self):
        self.sys_agent_tar.load_state_dict(self.sys_agent.state_dict())
            

        

        

        
        