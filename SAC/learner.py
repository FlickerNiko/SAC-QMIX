import torch
import torch.nn as nn
import torch.nn.functional as F
#import cuprof
#import profile
class Learner:
    def __init__(self, sys_actor, sys_critic, args):
        
        self.device = args.device
        
        self.sys_actor = sys_actor
        self.sys_critic = sys_critic
        self.sys_critic.train()
        self.sys_critic_tar = type(sys_critic)(args)
        if self.device == 'cuda':
            self.sys_critic_tar.cuda()
        self.sys_critic_tar.requires_grad_(False)
        self._sync_target()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.lr = args.lr
        self.lr_actor = args.lr_actor
        self.l2 = args.l2
        self.target_update = args.target_update
        self.step = 0
        self.action_explore = args.action_explore
        self.optim_type = args.optim_type
        self.args = args
        if self.optim_type == 'SGD':
            self.optim_actor = torch.optim.SGD(self.sys_actor.parameters(), lr=self.lr_actor, momentum = 0.9, weight_decay=self.l2)            
            self.optim_critic = torch.optim.SGD(self.sys_critic.parameters(), lr=self.lr, momentum = 0.9, weight_decay=self.l2)            
        else:
            self.optim_actor = torch.optim.Adam(self.sys_actor.parameters(), lr = self.lr_actor, weight_decay=self.l2)
            self.optim_critic = torch.optim.Adam(self.sys_critic.parameters(), lr = self.lr, weight_decay=self.l2)
        
    def train(self, data):

        self.step += 1
        
        obs = data['obs'].to(device=self.device, non_blocking=True)
        actions = data['actions'].to(device=self.device, non_blocking=True)
        reward = data['reward'].to(device=self.device, non_blocking=True)
        valid = data['valid'].to(device=self.device, non_blocking=True)
        avail_actions = data['avail_actions'].to(device=self.device, non_blocking=True)
        
 
        n_batch = obs.shape[0]
        T = obs.shape[1]

        hiddens = self.sys_critic.init_hiddens(n_batch)
        hiddens_tar = self.sys_critic_tar.init_hiddens(n_batch)
        hiddens_actor = self.sys_actor.init_hiddens(n_batch)
        actions_onehot = self.one_hot(actions,self.n_actions)

        qs = []
        qs_tar = []
        qs_star = []
        ps = []
        
        

        for i in range(T):
            
            q, hiddens = self.sys_critic.forward(obs[:,i], actions_onehot[:,i], hiddens)
            p, a, hiddens_actor = self.sys_actor.forward(obs[:,i], avail_actions[:,i], hiddens_actor)

            p = self.gather_end(p,a)
            a_onehot = self.one_hot(a,self.n_actions)
            with torch.no_grad():
                
                q_tar, hiddens_tar = self.sys_critic_tar.forward(obs[:,i], a_onehot, hiddens_tar)
            
            qs.append(q)
            qs_tar.append(q_tar)
            ps.append(p)

        qs = torch.stack(qs,1)
        qs_tar = torch.stack(qs_tar,1)
        ps = torch.stack(ps,1)
        
        qs = self._valid_mask(qs,valid)
        qs_tar = self._valid_mask(qs_tar,valid)
        

        #qs *= valid.view(list(valid.shape) + [1] * (qs.ndim - valid.ndim))
        #Vs_tar *= valid.view(list(valid.shape) + [1] * (Vs_tar.ndim - valid.ndim))
        
        for i in range(T-1):            
            r = reward[:,i]
            q_next = qs_tar[:,i+1]
            q_star = r.view(-1,1,1) + self.gamma * q_next
            qs_star.append(q_star)

        #i = T - 1
        qs_star.append(reward[:, T-1].view(-1,1,1) + reward.new_zeros(q_next.shape))
        qs_star = torch.stack(qs_star,1)
        qs_star = self._valid_mask(qs_star,valid)
        #Vs_star *= valid.view(list(valid.shape) + [1] * (Vs_star.ndim - valid.ndim))

        #As = (Vs_star - Vs.detach()).squeeze(-1)

        loss = F.mse_loss(qs,qs_star)
        valid_rate = len(torch.nonzero(valid, as_tuple=False))/valid.numel()
        loss /= valid_rate
        
        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()
        self.update_target()

        # train actor

        loss_actor = -torch.mean(qs_tar.squeeze(-1)*torch.log(ps+1e-38))
        loss_actor/=valid_rate

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()
        

        return loss.item()



        
    def gather_end(self, input, index):
        index = torch.unsqueeze(index,-1).to(dtype=torch.int64)
        return torch.gather(input, input.ndim -1, index).squeeze(-1) 

    def one_hot(self, tensor, n_classes):
        return F.one_hot(tensor.to(dtype=torch.int64), n_classes).to(dtype=torch.float32)

    def update_target(self):
        if self.target_update >=1:
            if self.step % self.target_update == 0:
                self._sync_target()
        else:
            cur_state = self.sys_critic.state_dict()
            tar_state = self.sys_critic_tar.state_dict()
            for key in tar_state:
                
                v_tar = tar_state[key]
                v_cur = cur_state[key]
                v_tar += self.target_update*(v_cur-v_tar).detach()
                #v_tar.copy_(((1-self.target_update)*v_tar + self.target_update*v_cur).detach())
                #tar_state[key] = (0.9*v_tar + 0.1*v_cur).detach()            
            #self.sys_critic_tar.load_state_dict(tar_state)
    
    def _sync_target(self):
        self.sys_critic_tar.load_state_dict(self.sys_critic.state_dict())
 
    def _valid_mask(self, input, valid):
        input *= valid.view(list(valid.shape) + [1] * (input.ndim - valid.ndim))
        return input

        
        