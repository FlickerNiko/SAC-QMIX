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
        self.epsilon_len = args.epsilon_len
        self.device = args.device
        self.explore_type = args.explore_type
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

    def get_actions(self, states, avail_actions, explore=False):
           
        states = torch.as_tensor(states).unsqueeze(0)
        avail_actions = torch.as_tensor(avail_actions).unsqueeze(0)
        actions_explore = torch.zeros(1,self.n_agents,self.n_actions)
        with torch.no_grad():
            qs, hs_next = self.sys_agent.forward(states, actions_explore, self.last_actions, self.hiddens)
        self.hiddens = hs_next
        qs -= (1-avail_actions)*1e38
        
        explores = torch.zeros(self.n_agents, dtype= torch.int32)
        learns = torch.ones(self.n_agents, dtype= torch.int32)

        if self.episode> self.epsilon_len:
            epsilon = self.epsilon_ed
        else:
            rate = self.episode/self.epsilon_len
            epsilon =  (1-rate)*self.epsilon_st + rate*self.epsilon_ed

        if explore:
            if self.explore_type == 'independent':
                #learns = torch.zeros(self.n_agents, dtype= torch.int32)
                for i in range(self.n_agents):
                    rand = random.random()
                    if rand < epsilon:
                        explores[i] = 1
                        learns[i] = 1
                #if torch.nonzero(explores).shape[0] == 0:                
                    #learns = torch.ones(self.n_agents, dtype = torch.int32)

            elif self.explore_type == 'solo':
                rand = random.random()
                if rand < epsilon:
                    agent_id = random.randint(0,self.n_agents-1)
                    explores[agent_id] = 1
                    learns = torch.zeros(self.n_agents, dtype= torch.int32)
                    learns[agent_id] = 1
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
        return actions.numpy(), explores.numpy(), learns.numpy()

    def one_hot(self, tensor, n_classes):
        return F.one_hot(tensor.to(dtype=torch.int64), n_classes).to(dtype=torch.float32)
