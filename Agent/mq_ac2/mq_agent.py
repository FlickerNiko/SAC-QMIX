import torch
import torch.nn as nn
import torch.nn.functional as F
from .mrnn_agent import RNNAgent
from .msg_hub import MsgHub
from .q_action import QAction
from .mq_critic import MQCritic
from itertools import chain
#from .rnn_msg_hub import RNNMsgHub


class MQAgent(nn.Module):
    def __init__(self, args):
        super(MQAgent, self).__init__()
        
        self.n_agents = args.n_agents
        self.msg_dim = args.msg_dim
        self.rnn_hub = args.rnn_hub
        self.agent = RNNAgent(args)
        self.msg_hub = MsgHub(args)
        self.mq_critic = MQCritic(args)
        self.counter_factual = args.counter_factual

    def init_hiddens(self, n_batch):
        hiddens = [self.agent.init_hidden(n_batch) for i in range(self.n_agents)]
        hiddens = torch.stack(hiddens,1)
        
        return hiddens

    def forward(self, states, actions_explore, actions_last, hiddens):  
        with torch.no_grad():    
            ms_send, _ = self.msg_forward(states, actions_explore, actions_last, hiddens)
            ms_recv = self.msg_hub.forward(ms_send)
            ms_index = torch.argmax(ms_recv,2)
            ms_onehot = F.one_hot(ms_index, self.msg_dim).to(dtype=torch.float32)
        qs, hs_next = self.q_forward(states, actions_explore, actions_last, hiddens, ms_onehot)
        return qs, hs_next

    def critic_forward(self, states, actions_explore, actions_last, hiddens, ms_recv):
        ms_send, _ = self.msg_forward(states, actions_explore, actions_last, hiddens)
        q = self.mq_critic.forward(ms_send,ms_recv)
        return q

    def q_forward(self, states, actions_explore, actions_last, hiddens, ms_recv):
        n_batch = len(states)
        a_zero = torch.zeros_like(actions_explore[:,0])
        agent_ids = torch.arange(0,self.n_agents,dtype=torch.int64, device=states.device)        
        agent_ids = F.one_hot(agent_ids, self.n_agents).to(dtype=torch.float32)
        agent_ids = torch.stack([agent_ids]*n_batch, 0)

        qs = []
        hs_next = []

        for i in range(self.n_agents):
            q, _, h_next = self.agent.forward(states[:,i], agent_ids[:,i],a_zero, ms_recv[:,i], actions_last[:,i] , hiddens[:,i])
            qs.append(q)
            hs_next.append(h_next)

        qs = torch.stack(qs,1)
        hs_next = torch.stack(hs_next,1)

        return qs, hs_next
        
    def msg_forward(self, states, actions_explore, actions_last, hiddens):
        hs_agent = hiddens
        ms_send = []
        n_batch = len(states)
        m_zero = states.new_zeros(n_batch, self.msg_dim)        
        agent_ids = torch.arange(0,self.n_agents,dtype=torch.int64, device=states.device)        
        agent_ids = F.one_hot(agent_ids, self.n_agents).to(dtype=torch.float32)
        agent_ids = torch.stack([agent_ids]*n_batch, 0)
        hs_next = []
        for i in range(self.n_agents):
            _,m,h = self.agent.forward(states[:,i], agent_ids[:,i],actions_explore[:,i],m_zero,actions_last[:,i] , hs_agent[:,i])
            ms_send.append(m)
            hs_next.append(h)
            
        ms_send = torch.stack(ms_send,1)
        hs_next = torch.stack(hs_next,1)

        return ms_send, hs_next

    def actor_forward(self, states, actions_explore, actions_last, hiddens):
        ms_send, _ = self.msg_forward(states, actions_explore, actions_last, hiddens)
        ms_recv = self.msg_hub.forward(ms_send)
        return ms_recv
    
    def train_actor_forward(self, states, actions_explore, actions_last, hiddens):
        ms_send, hs_next = self.msg_forward(states, actions_explore, actions_last, torch.zeros_like(hiddens))
        ms_recv = self.msg_hub.forward(ms_send)        
        
        q = self.mq_critic.forward(ms_send.detach(),ms_recv)
        return q, hs_next
    
    def actor_parameters(self):
        para_agent = self.agent.parameters()
        para_hub = self.msg_hub.parameters()
        return chain(para_agent,para_hub)

    def critic_parameters(self):
        para_agent = self.agent.parameters()
        para_critic = self.mq_critic.parameters()
        return chain(para_agent,para_critic)
