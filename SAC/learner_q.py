import torch
import torch.nn as nn
import torch.nn.functional as F
from .vdn_actor import VDNActor
from .us_critic2 import VDNCritic
from .vdn_net import VDNNet
from .qmix_net import QMixNet
#import cuprof
#import profile
class Learner:
    def __init__(self, args):
        
        self.device = args.device
        
        
        self.sys_critic = VDNCritic(args)        
        self.sys_critic_tar = VDNCritic(args)        
        self.sys_critic.train()        
        
        # self.vdn_net = QMixNet(args)
        # self.vdn_net_tar = QMixNet(args)
        self.vdn_net = VDNNet()
        self.vdn_net_tar = VDNNet()
        if self.device == 'cuda':
            
            self.sys_critic.cuda()
            
            self.sys_critic_tar.cuda()
            
            self.vdn_net.cuda()
            self.vdn_net_tar.cuda()
        
        self.sys_critic_tar.requires_grad_(False)
        self.vdn_net_tar.requires_grad_(False)
        



        self._sync_target()        
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.gamma = args.gamma
        
        self.lr = args.lr
        self.lr_actor = args.lr_actor
        self.l2 = args.l2
        self.target_update = args.target_update
        self.step = 0        
        self.optim_type = args.optim_type
        self.agent_id = args.agent_id
        self.args = args

        
        
        
        params_critic = list(self.sys_critic.parameters()) + list(self.vdn_net.parameters())


        if self.optim_type == 'SGD':            
            self.optim_critic = torch.optim.SGD(params_critic, lr=self.lr, momentum = 0.9, weight_decay=self.l2)                        
        else:            
            self.optim_critic = torch.optim.Adam(params_critic, lr = self.lr, weight_decay=self.l2)
            

    def train(self, data):

        self.step += 1
        
        state = data['state'].to(device=self.device, non_blocking=True)
        obs = data['obs'].to(device=self.device, non_blocking=True)
        actions = data['actions'].to(device=self.device, non_blocking=True)
        reward = data['reward'].to(device=self.device, non_blocking=True)
        valid = data['valid'].to(device=self.device, non_blocking=True)
        avail_actions = data['avail_actions'].to(device=self.device, non_blocking=True)
        
 
        n_batch = obs.shape[0]
        T = obs.shape[1]
        if self.agent_id:
            agent_ids = torch.eye(self.n_agents,device=obs.device)
            agent_ids = agent_ids.reshape((1,)*(obs.ndim-2)+agent_ids.shape)            
            agent_ids = agent_ids.expand(obs.shape[:-2]+(-1,-1))
            obs = torch.cat([obs,agent_ids],-1)

        hiddens = self.sys_critic.init_hiddens(n_batch)
        hiddens_tar = self.sys_critic_tar.init_hiddens(n_batch)        
        actions_onehot = self.one_hot(actions,self.n_actions)

        Qs = []        
        Qs_tar = []        
        qs_star = []
        
        for i in range(T):
            
            Q, hiddens = self.sys_critic.forward(obs[:,i], avail_actions[:,i], hiddens)            
            with torch.no_grad():                
                Q_tar, hiddens_tar = self.sys_critic_tar.forward(obs[:,i], avail_actions[:,i], hiddens_tar)
            Qs.append(Q)            
            Qs_tar.append(Q_tar)
            

        Qs = torch.stack(Qs,1)        
        Qs_tar = torch.stack(Qs_tar,1)

        Qs[valid == 0] = 0            
        Qs_tar[valid == 0] = 0
        qs = self.gather_end(Qs,actions)
        #vdn_qs = self.vdn_net(qs, state)
        vdn_qs = self.vdn_net(qs)
        qs_tar = torch.max(Qs_tar,-1)[0]
        #vdn_qs_tar = self.vdn_net_tar(qs_tar, state)
        vdn_qs_tar = self.vdn_net_tar(qs_tar)

        vdn_qs[valid == 0] = 0
        vdn_qs_tar[valid == 0] = 0

        qs_star = torch.zeros_like(vdn_qs)
        qs_star += reward.unsqueeze(-1)
        qs_star[:,:-1] += self.gamma * (vdn_qs_tar.detach()[:,1:])
        
        valid_rate = torch.mean(valid.float())

        loss = F.mse_loss(vdn_qs,qs_star)/valid_rate
        
        
        self.optim_critic.zero_grad()        
        loss.backward()        
        self.optim_critic.step()        
        self.update_target()

        q = torch.mean(vdn_qs)/valid_rate


        ret = {}
        ret['loss'] = loss.item()
        ret['q'] = q.item()
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
            soft_update(self.sys_critic,self.sys_critic_tar)
            soft_update(self.vdn_net,self.vdn_net_tar)
            
    
    def _sync_target(self):
        self.sys_critic_tar.load_state_dict(self.sys_critic.state_dict())
        self.vdn_net_tar.load_state_dict(self.vdn_net.state_dict())
        
        
 
    def _valid_mask(self, input, valid):
        input *= valid.view(list(valid.shape) + [1] * (input.ndim - valid.ndim))
        return input
        