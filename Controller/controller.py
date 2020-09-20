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
        self.epsilon = args.epsilon
        self.device = args.device
        self.solo_explore = args.solo_explore
        self.sys_agent = sys_agent


    def new_episode(self):

        self.hiddens = self.sys_agent.init_hiddens(1)
        self.last_actions = torch.zeros(1, self.n_agents, self.n_actions, device=self.device)

    def get_actions(self, states, avail_actions, explore=False):
        # data: obs, explore
               
        states = torch.as_tensor(states, device=self.device).unsqueeze(0)
        avail_actions = torch.as_tensor(avail_actions,device=self.device).unsqueeze(0)
        #with torch.no_grad():
        qs, hs_next = self.sys_agent.forward(states, self.last_actions, self.hiddens)
        self.hiddens = hs_next
        qs -= (1-avail_actions)*1e38
        actions = []
        explores = [0]*self.n_agents
        to_learns = [1]*self.n_agents
        if explore:
            rand = random.random()
            if(rand > self.epsilon):
                explore = False
        if explore:
            if self.solo_explore:
                index = random.randint(0,self.n_agents-1)
                explores[index] = 1
                to_learns = [0]*self.n_agents
                to_learns[index] = 1
                
            else:
                explores = [1]*self.n_agents

        for q, avail_a, explore in zip(qs[0], avail_actions[0], explores) :
            
            if explore:
                q = torch.rand_like(q) - (1-avail_a)*1e38

            a = torch.argmax(q)                        
            actions.append(a)

        actions = torch.stack(actions)

        self.last_actions = self.one_hot(actions, self.n_actions).unsqueeze(0)
        return actions.to(device='cpu').numpy(), to_learns

    def one_hot(self, tensor, n_classes):
        return F.one_hot(tensor.to(dtype=torch.int64), n_classes).to(dtype=torch.float32)
