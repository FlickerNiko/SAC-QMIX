import torch
import torch.nn as nn
import torch.nn.functional as F
from .mrnn_agent import RNNAgent
from .msg_hub import MsgHub

class MQAgent(nn.Module):
    def __init__(self, args):
        super(MQAgent, self).__init__()

        self.n_agents = args.n_agents
        self.msg_dim = args.msg_dim
        self.rnn_hub = args.rnn_hub
        self.agent = RNNAgent(args)
        self.msg_hub = MsgHub(args)
        self.counter_factual = args.counter_factual

    def init_hiddens(self, n_batch):
        hiddens = [self.agent.init_hidden(n_batch) for i in range(self.n_agents)]
        hiddens = torch.stack(hiddens,1)
        
        return hiddens

    def forward(self, states, actions_explore, actions_last, hiddens):
        
    
        ms_recv = self.message_forward(states, actions_explore, actions_last, hiddens)

        ms_index = torch.argmax(ms_recv.detach(),2)
        ms_onehot = F.one_hot(ms_index,self.msg_dim).to(dtype=torch.float32)
    
        qs, hs_next = self.q_forward(states, actions_explore, actions_last, hiddens, ms_onehot)


        return qs, ms_recv, hs_next

    def message_forward(self, states, actions_explore, actions_last, hiddens):
         
        ms_send = []
        n_batch = len(states)
        m_zero = states.new_zeros(n_batch, self.msg_dim)        
        agent_ids = torch.arange(0,self.n_agents,dtype=torch.int64, device=states.device)        
        agent_ids = F.one_hot(agent_ids, self.n_agents).to(dtype=torch.float32)
        agent_ids = torch.stack([agent_ids]*n_batch, 0)

        for i in range(self.n_agents):
            ms_send.append(self.agent.forward(states[:,i], agent_ids[:,i], actions_explore[:,i], m_zero, actions_last[:,i] , hiddens[:,i])[1])

        ms_send = torch.stack(ms_send,1)        
        ms_recv = self.msg_hub(ms_send)
        return ms_recv

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
    