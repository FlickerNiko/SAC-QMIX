import torch
import torch.nn as nn
import torch.nn.functional as F
from .vdn_actor import VDNActor
from .us_critic2 import VDNCritic
from .vdn_net import VDNNet
from .qmix_net import QMixNet
#import cuprof
#import profile
vdn = False
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
    
        self.vdn_net1 = QMixNet(args)        
        self.vdn_net2 = QMixNet(args)        
        self.vdn_net1_tar = QMixNet(args)
        self.vdn_net2_tar = QMixNet(args)
        
        if self.device == 'cuda':
            self.sys_actor.cuda()
            self.sys_critic1.cuda()
            self.sys_critic2.cuda()
            self.sys_critic1_tar.cuda()
            self.sys_critic2_tar.cuda()
            self.vdn_net1.cuda()
            self.vdn_net2.cuda()
            self.vdn_net1_tar.cuda()
            self.vdn_net2_tar.cuda()
        
        self.sys_critic1_tar.requires_grad_(False)
        self.sys_critic2_tar.requires_grad_(False)
        self.vdn_net1_tar.requires_grad_(False)
        self.vdn_net2_tar.requires_grad_(False)



        self._sync_target()

        
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        self.entropy_tar = args.entropy_tar
        self.lr = args.lr
        self.lr_actor = args.lr_actor
        self.lr_alpha = args.lr_alpha
        self.l2 = args.l2
        self.target_update = args.target_update
        self.step = 0        
        self.optim_type = args.optim_type
        self.agent_id = args.agent_id
        self.last_action = args.last_action        
        self.args = args

        
        self.log_alpha = torch.tensor([-2]*self.n_agents, dtype=torch.float32, requires_grad=True, device = self.device)        
        #self.log_alpha = torch.zeros(self.n_agents,dtype=torch.float32,requires_grad=True,device=self.device)
        self.mix_alpha_tar = 0
        params_critic = list(self.sys_critic1.parameters()) + list(self.sys_critic2.parameters()) + list(self.vdn_net1.parameters()) + list(self.vdn_net2.parameters())
        
        self.params_critic = params_critic
        self.params_actor = list(self.sys_actor.parameters())

        if self.optim_type == 'SGD':
            self.optim_actor = torch.optim.SGD(self.sys_actor.parameters(), lr=self.lr_actor, momentum = 0.9, weight_decay=self.l2)            
            self.optim_critic = torch.optim.SGD(params_critic, lr=self.lr, momentum = 0.9, weight_decay=self.l2)
            
            self.optim_alpha = torch.optim.SGD([self.log_alpha], lr=self.lr_alpha, momentum = 0.9)
        else:
            self.optim_actor = torch.optim.Adam(self.sys_actor.parameters(), lr = self.lr_actor, weight_decay=self.l2)
            self.optim_critic = torch.optim.Adam(params_critic, lr = self.lr, weight_decay=self.l2)
            
            self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)

    def train(self, data):

        self.step += 1
        
        state = data['state'].to(device=self.device, non_blocking=True)
        obs = data['obs'].to(device=self.device, non_blocking=True)
        actions = data['actions'].to(device=self.device, non_blocking=True)
        reward = data['reward'].to(device=self.device, non_blocking=True)
        valid = data['valid'].to(device=self.device, non_blocking=True)
        avail_actions = data['avail_actions'].to(device=self.device, non_blocking=True)
        actions_onehot = self.one_hot(actions,self.n_actions)
 
        n_batch = obs.shape[0]
        T = obs.shape[1]
        alpha = torch.exp(self.log_alpha.detach())
        valid_rate = torch.mean(valid.float())

        if self.agent_id:
            agent_ids = torch.eye(self.n_agents,device=obs.device)
            agent_ids = agent_ids.reshape((1,)*(obs.ndim-2)+agent_ids.shape)            
            agent_ids = agent_ids.expand(obs.shape[:-2]+(-1,-1))
            obs = torch.cat([obs,agent_ids],-1)
        if self.last_action:
            last_actions = torch.zeros_like(actions_onehot)
            last_actions[:,1:] = actions_onehot[:,:-1]
            obs = torch.cat([obs,last_actions],-1)

        hiddens1 = self.sys_critic1.init_hiddens(n_batch)
        hiddens1_tar = self.sys_critic1_tar.init_hiddens(n_batch)
        hiddens2 = self.sys_critic2.init_hiddens(n_batch)
        hiddens2_tar = self.sys_critic2_tar.init_hiddens(n_batch)

        hiddens_actor = self.sys_actor.init_hiddens(n_batch)
        

        Q1s = []
        Q2s = []
        Q1s_tar = []
        Q2s_tar = []
        qs_star = []
        ps = []
        
        

        for i in range(T):
            
            Q1, hiddens1 = self.sys_critic1.forward(obs[:,i], avail_actions[:,i], hiddens1)
            Q2, hiddens2 = self.sys_critic2.forward(obs[:,i], avail_actions[:,i], hiddens2)
            p, hiddens_actor = self.sys_actor.forward(obs[:,i], avail_actions[:,i], hiddens_actor)
            
            with torch.no_grad():
                
                Q1_tar, hiddens1_tar = self.sys_critic1_tar.forward(obs[:,i], avail_actions[:,i], hiddens1_tar)
                Q2_tar, hiddens2_tar = self.sys_critic2_tar.forward(obs[:,i], avail_actions[:,i], hiddens2_tar)
            
            Q1s.append(Q1)
            Q2s.append(Q2)
            Q1s_tar.append(Q1_tar)
            Q2s_tar.append(Q2_tar)
            ps.append(p)

        Q1s = torch.stack(Q1s,1)
        Q2s = torch.stack(Q2s,1)
        Q1s_tar = torch.stack(Q1s_tar,1)
        Q2s_tar = torch.stack(Q2s_tar,1)

        ps = torch.stack(ps,1)
        ps[valid == 0] = 0
        
        log_ps = torch.log(ps + 1e-38)
        log_ps[valid == 0] = 0
        entropy = -torch.sum(ps*log_ps, -1)
        

        Q1s[valid == 0] = 0
        Q2s[valid == 0] = 0
        Q1s_tar[valid == 0] = 0
        Q2s_tar[valid == 0] = 0

        q1s = self.gather_end(Q1s,actions)
        q2s = self.gather_end(Q2s,actions)

        vdn_q1s = self.vdn_net1(q1s, state)
        vdn_q2s = self.vdn_net2(q2s, state)
        
        vdn_q1s[valid == 0] = 0
        vdn_q2s[valid == 0] = 0

        
        
        
        #Qs = torch.min(torch.stack([Q1s,Q2s],-1), -1)[0]
        #Qs_tar = torch.min(torch.stack([Q1s_tar,Q2s_tar],-1), -1)[0]
        
        
        V1s = torch.sum(ps * Q1s.detach(), -1)
        V2s = torch.sum(ps * Q2s.detach(), -1)
        
        V1s_tar = torch.sum(ps * Q1s_tar, -1)# + alpha*entropy
        V2s_tar = torch.sum(ps * Q2s_tar, -1)# + alpha*entropy
        
        self.vdn_net1.requires_grad_(False)
        self.vdn_net2.requires_grad_(False)
        
        vdn_V1s = self.vdn_net1.forward(V1s, state)
        vdn_V2s = self.vdn_net2.forward(V2s, state)
        vdn_V1s_tar = self.vdn_net1_tar.forward(V1s_tar, state)
        vdn_V2s_tar = self.vdn_net2_tar.forward(V2s_tar, state)
        
        vdn_Vs = torch.min(torch.stack([vdn_V1s,vdn_V2s],-1), -1)[0]
        vdn_Vs_tar = torch.min(torch.stack([vdn_V1s_tar,vdn_V2s_tar],-1), -1)[0]

        vdn_Vs[valid == 0] = 0
        vdn_Vs_tar[valid == 0] = 0

        self.vdn_net1.requires_grad_(True)
        self.vdn_net2.requires_grad_(True)
                
        vdn_Ves = vdn_Vs + torch.sum(alpha * entropy, -1, keepdim=True)
        vdn_Ves_tar = vdn_Vs_tar + torch.sum(alpha * entropy, -1, keepdim=True).detach()
        #vdn_Ves_tar = vdn_Vs_tar + torch.sum(entropy.detach(), -1, keepdim=True)*alpha.mean()

        # train actor
        #loss_actor = - torch.mean(Ves)/valid_rate
        loss_actor = - torch.mean(vdn_Ves)/valid_rate

        self.optim_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.params_actor, self.args.grad_norm_clip)
        self.optim_actor.step()
               
        h = torch.mean(entropy.detach())/valid_rate        
        q = torch.mean(V1s.detach())/valid_rate        
        q_total = torch.mean(vdn_Vs.detach())/valid_rate
                        
        qs_star = torch.zeros_like(vdn_q1s)
        qs_star += reward.unsqueeze(-1)
        qs_star[:,:-1] += self.gamma * (vdn_Ves_tar.detach()[:,1:])

        #qs_star = qs_star.detach()
        #qs_star[valid == 0] = 0
        
        loss1 = F.mse_loss(vdn_q1s,qs_star)/valid_rate
        loss2 = F.mse_loss(vdn_q2s,qs_star)/valid_rate        
        loss = loss1+loss2
        self.optim_critic.zero_grad()        
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(self.params_critic, self.args.grad_norm_clip)      
        self.optim_critic.step()        
        self.update_target()
        
        loss_alpha = self.log_alpha*(entropy.detach()-self.entropy_tar)
        loss_alpha[valid ==0] = 0
        loss_alpha = torch.mean(loss_alpha)/valid_rate

        self.optim_alpha.zero_grad()
        loss_alpha.backward()
        self.optim_alpha.step()
        # max_grads = []
        # for param in self.sys_actor.parameters():
        #     max_grads.append(torch.abs(param.grad).max())
        # for param in self.params_critic:
        #     max_grads.append(torch.abs(param.grad).max())
        
        #max_grad = max(max_grads)
        max_p = torch.mean(torch.max(ps.detach(),-1)[0])/valid_rate
        ret = {}
        ret['loss'] = (loss/2).item()
        ret['q'] = q.item()
        ret['q_total'] = q_total.item()
        ret['h'] = h.item()
        ret['max_p'] = max_p
        ret['alpha'] = alpha.mean().item()
        ret['max_grad'] = 0
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
            soft_update(self.vdn_net1,self.vdn_net1_tar)
            soft_update(self.vdn_net2,self.vdn_net2_tar)
    
    def _sync_target(self):
        self.sys_critic1_tar.load_state_dict(self.sys_critic1.state_dict())
        self.sys_critic2_tar.load_state_dict(self.sys_critic2.state_dict())
        self.vdn_net1_tar.load_state_dict(self.vdn_net1.state_dict())
        self.vdn_net2_tar.load_state_dict(self.vdn_net2.state_dict())
 
    def _valid_mask(self, input, valid):
        input *= valid.view(list(valid.shape) + [1] * (input.ndim - valid.ndim))
        return input
        