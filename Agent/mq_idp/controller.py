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
        self.epsilon_st = args.epsilon_st
        self.epsilon_ed = args.epsilon_ed
        self.eps_half_time = args.eps_half_time
        self.epsilon = args.epsilon_st
        self.device = args.device
        self.explore_type = args.explore_type
        self.msg_dim = args.msg_dim
        self.sys_agent_src = sys_agent
        self.sys_agent = type(sys_agent)(args)        
        self.sys_agent.eval()
        self.episode = 0

    def new_episode(self):
        state_dict = self.sys_agent_src.state_dict()
        self.sys_agent.load_state_dict(state_dict)        
        self.hiddens = self.sys_agent.init_hiddens(1)
        self.last_actions = torch.zeros(1, self.n_agents, self.n_actions)
        self.episode += 1

        T =  self.episode/self.eps_half_time
        self.epsilon = self.epsilon_st * (0.5 ** T)
        if self.epsilon < self.epsilon_ed:
            self.epsilon = self.epsilon_ed

    def get_actions(self, states, avail_actions, explore=False):
        
        epsilon = self.epsilon

        states = torch.as_tensor(states).unsqueeze(0)
        avail_actions = torch.as_tensor(avail_actions).unsqueeze(0)
        actions_explore = torch.zeros(1,self.n_agents,self.n_actions)
        with torch.no_grad():
            ms_recv = self.sys_agent.message_forward(states, actions_explore, self.last_actions, self.hiddens)
            if explore:
                for i in range(self.n_agents):
                    rand = random.random()
                    if rand < epsilon:
                        msg = ms_recv[0,i]
                        msg = torch.randn_like(msg)
            
            ms_index = torch.argmax(ms_recv,2)
            ms_onehot = self.one_hot(ms_index, self.msg_dim)
            qs, hs_next = self.sys_agent.q_forward(states, actions_explore, self.last_actions, self.hiddens, ms_onehot)
        
        self.hiddens = hs_next
        qs -= (1-avail_actions)*1e38
        
        explores = torch.zeros(self.n_agents, dtype= torch.int32)       

        if explore:
            if self.explore_type == 'independent':
                for i in range(self.n_agents):
                    rand = random.random()
                    if rand < epsilon:
                        explores[i] = 1

            elif self.explore_type == 'solo':
                rand = random.random()
                if rand < epsilon:
                    agent_id = random.randint(0,self.n_agents-1)
                    explores[agent_id] = 1

            else: #sync
                rand = random.random()
                if rand < epsilon:                    
                    explores = torch.ones(self.n_agents, dtype= torch.int32)

        actions = []

        for q, avail_a, explore in zip(qs[0], avail_actions[0], explores) :
            
            if explore:
                q = torch.rand_like(q) - (1-avail_a)*1e38

            a = torch.argmax(q)                        
            actions.append(a)

        actions = torch.stack(actions)

        self.last_actions = self.one_hot(actions, self.n_actions).unsqueeze(0)
        return actions.numpy(),ms_onehot[0].numpy(), explores.numpy()

    def one_hot(self, tensor, n_classes):
        return F.one_hot(tensor.to(dtype=torch.int64), n_classes).to(dtype=torch.float32)
