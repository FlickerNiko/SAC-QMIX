import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


#n_batch = 1


class Controller:
    def __init__(self, sys_agent, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions        
        self.agent_id = args.agent_id
        self.last_action = args.last_action
        self.device = args.device    
        self.sys_agent_src = sys_agent
        self.sys_agent = type(sys_agent)(args)        
        self.sys_agent.requires_grad_(False)        
        self.episode = 0

    def new_episode(self):
        state_dict = self.sys_agent_src.state_dict()
        self.sys_agent.load_state_dict(state_dict)        
        self.hiddens = self.sys_agent.init_hiddens(1)
        self.last_actions = torch.zeros(1, self.n_agents, self.n_actions)      
        self.episode += 1

    def get_actions(self, states, avail_actions, explore=False):
                        
        states = torch.as_tensor(states).unsqueeze(0)
        avail_actions = torch.as_tensor(avail_actions).unsqueeze(0)
        if self.agent_id:
            agent_ids = torch.eye(self.n_agents,device=states.device)
            agent_ids = agent_ids.reshape((1,)*(states.ndim-2)+agent_ids.shape)            
            agent_ids = agent_ids.expand(states.shape[:-2]+(-1,-1))
            states = torch.cat([states,agent_ids],-1)
        if self.last_action:            
            states = torch.cat([states, self.last_actions],-1)
        with torch.no_grad():
            ps, hs_next = self.sys_agent.forward(states, avail_actions, self.hiddens)
        self.hiddens = hs_next
        acts = torch.multinomial(ps[0],1).squeeze(-1)
        if explore:
            actions = acts
        else:
            actions = torch.argmax(ps[0],-1)
        self.last_actions = self.one_hot(actions, self.n_actions).unsqueeze(0)
        return actions.numpy()

    def one_hot(self, tensor, n_classes):
        return F.one_hot(tensor.to(dtype=torch.int64), n_classes).to(dtype=torch.float32)
