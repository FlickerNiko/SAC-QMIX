import torch
import torch.nn as nn
import torch.nn.functional as F
from .us_actor import VDNActor
from .us_critic import VDNCritic
#import cuprof
#import profile
class Learner:
    def __init__(self, args):
        
        self.device = args.device
        
        self.sys_actor = VDNActor(args)
        self.sys_critic1 = VDNCritic(args)
        self.sys_critic2 = VDNCritic(args)
        self.sys_critic1_tar = VDNCritic(args)
        self.sys_critic2_tar = VDNCritic(args)
        self.sys_critic1.train()
        self.sys_critic2.train()
        
        if self.device == 'cuda':
            self.sys_actor.cuda()
            self.sys_critic1.cuda()
            self.sys_critic2.cuda()
            self.sys_critic1_tar.cuda()
            self.sys_critic2_tar.cuda()

        
        self.sys_critic1_tar.requires_grad_(False)
        self.sys_critic2_tar.requires_grad_(False)
        



        self._sync_target()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.entropy_tar = args.entropy_tar
        self.lr = args.lr
        self.lr_actor = args.lr_actor
        self.l2 = args.l2
        self.target_update = args.target_update
        self.step = 0
        self.action_explore = args.action_explore
        self.optim_type = args.optim_type
        self.args = args

        
        self.log_alpha = torch.tensor(-1.5, dtype=torch.float32, requires_grad=True)
        


        if self.optim_type == 'SGD':
            self.optim_actor = torch.optim.SGD(self.sys_actor.parameters(), lr=self.lr_actor, momentum = 0.9, weight_decay=self.l2)            
            self.optim_critic1 = torch.optim.SGD(self.sys_critic1.parameters(), lr=self.lr, momentum = 0.9, weight_decay=self.l2)
            self.optim_critic2 = torch.optim.SGD(self.sys_critic2.parameters(), lr=self.lr, momentum = 0.9, weight_decay=self.l2)
            self.optim_alpha = torch.optim.SGD([self.log_alpha], lr=3e-4, momentum = 0.9)
        else:
            self.optim_actor = torch.optim.Adam(self.sys_actor.parameters(), lr = self.lr_actor, weight_decay=self.l2)
            self.optim_critic1 = torch.optim.Adam(self.sys_critic1.parameters(), lr = self.lr, weight_decay=self.l2)
            self.optim_critic2 = torch.optim.Adam(self.sys_critic2.parameters(), lr = self.lr, weight_decay=self.l2)
            self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def train(self, data):

        self.step += 1
        
        obs = data['obs'].to(device=self.device, non_blocking=True)
        actions = data['actions'].to(device=self.device, non_blocking=True)
        reward = data['reward'].to(device=self.device, non_blocking=True)
        valid = data['valid'].to(device=self.device, non_blocking=True)
        avail_actions = data['avail_actions'].to(device=self.device, non_blocking=True)
        
 
        n_batch = obs.shape[0]
        T = obs.shape[1]
        alpha = torch.exp(self.log_alpha.detach())

        hiddens1 = self.sys_critic1.init_hiddens(n_batch)
        hiddens1_tar = self.sys_critic1_tar.init_hiddens(n_batch)
        hiddens2 = self.sys_critic2.init_hiddens(n_batch)
        hiddens2_tar = self.sys_critic2_tar.init_hiddens(n_batch)

        hiddens_actor = self.sys_actor.init_hiddens(n_batch)
        actions_onehot = self.one_hot(actions,self.n_actions)

        q1s = []
        q2s = []
        q1s_tar = []
        q2s_tar = []
        qs_star = []
        ps = []
        
        

        for i in range(T):
            
            q1, hiddens1 = self.sys_critic1.forward(obs[:,i], actions_onehot[:,i], hiddens1)
            q2, hiddens2 = self.sys_critic2.forward(obs[:,i], actions_onehot[:,i], hiddens2)
            p, a, hiddens_actor = self.sys_actor.forward(obs[:,i], avail_actions[:,i], hiddens_actor)
            p = self.gather_end(p,a)
            a_onehot = self.one_hot(a,self.n_actions)
            with torch.no_grad():
                
                q1_tar, hiddens1_tar = self.sys_critic1_tar.forward(obs[:,i], a_onehot, hiddens1_tar)
                q2_tar, hiddens2_tar = self.sys_critic2_tar.forward(obs[:,i], a_onehot, hiddens2_tar)
            
            q1s.append(q1)
            q2s.append(q2)
            q1s_tar.append(q1_tar)
            q2s_tar.append(q2_tar)
            ps.append(p)

        q1s = torch.stack(q1s,1)
        q2s = torch.stack(q2s,1)
        q1s_tar = torch.stack(q1s_tar,1)
        q2s_tar = torch.stack(q2s_tar,1)

        ps = torch.stack(ps,1)
        ps += 1e-38
        q1s = self._valid_mask(q1s,valid)
        q2s = self._valid_mask(q2s,valid)
        q1s_tar = self._valid_mask(q1s_tar,valid)
        q2s_tar = self._valid_mask(q2s_tar,valid)
        
        qs = torch.min(torch.stack([q1s,q2s],-1), -1)[0]
        qs_tar = torch.min(torch.stack([q1s_tar,q2s_tar],-1), -1)[0]
        

        
        for i in range(T-1):            
            r = reward[:,i]
            q_next = qs_tar[:,i+1]
            p_next = ps[:,i+1]
            q_star = r.view(-1,1) + self.gamma * (q_next - alpha * torch.log(p_next).mean(-1,keepdim=True))            
            #q_star = r.view(-1,1) + self.gamma * q_next
            qs_star.append(q_star)

        #i = T - 1
        qs_star.append(reward[:, T-1].view(-1,1) + reward.new_zeros(q_next.shape))
        qs_star = torch.stack(qs_star,1)
        qs_star = qs_star.detach()
        qs_star = self._valid_mask(qs_star,valid)
        
        valid_rate = len(torch.nonzero(valid, as_tuple=False))/valid.numel()

        loss1 = F.mse_loss(q1s,qs_star)
        loss2 = F.mse_loss(q2s,qs_star)

        
        loss1 /= valid_rate
        loss2 /= valid_rate
        
        self.optim_critic1.zero_grad()
        self.optim_critic2.zero_grad()
        loss1.backward()
        loss2.backward()
        self.optim_critic1.step()
        self.optim_critic2.step()
        self.update_target()

        # train actor
        
        # loss_actor = - qs_tar * torch.log(ps)
        loss_actor = (alpha *  (torch.log(ps.detach())+1) - qs_tar) * torch.log(ps)        
        # loss_actor = (alpha * torch.log(ps) - qs.detach()) * ps/ps.detach()
        loss_actor = self._valid_mask(loss_actor,valid)
        loss_actor = torch.mean(loss_actor)
        loss_actor/=valid_rate

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()
        

        h = -torch.log(ps.detach())
        h = self._valid_mask(h,valid)
        h = torch.mean(h)/valid_rate        
        q = torch.mean(q1s)/valid_rate

        loss_alpha = self.log_alpha*(h - self.entropy_tar)
        self.optim_alpha.zero_grad()
        loss_alpha.backward()
        self.optim_alpha.step()
        

        ret = {}
        ret['loss'] = ((loss1+loss2)/2).item()
        ret['q'] = q.item()
        ret['h'] = h.item()
        ret['alpha'] = alpha.item()
        return ret



        
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
            def soft_update(src,tar):
                cur_state = src.state_dict()
                tar_state = tar.state_dict()
                for key in tar_state:                
                    v_tar = tar_state[key]
                    v_cur = cur_state[key]
                    v_tar += self.target_update*(v_cur-v_tar).detach()
            soft_update(self.sys_critic1,self.sys_critic1_tar)
            soft_update(self.sys_critic2,self.sys_critic2_tar)
    
    def _sync_target(self):
        self.sys_critic1_tar.load_state_dict(self.sys_critic1.state_dict())
        self.sys_critic2_tar.load_state_dict(self.sys_critic2.state_dict())
 
    def _valid_mask(self, input, valid):
        input *= valid.view(list(valid.shape) + [1] * (input.ndim - valid.ndim))
        return input
        