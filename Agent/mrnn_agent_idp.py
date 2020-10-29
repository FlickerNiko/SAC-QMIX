import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, args):
        super(RNNAgent, self).__init__()
        self.input_dim = args.input_dim
        self.msg_dim = args.msg_dim
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.rnn_hidden_dim = args.rnn_hidden_dim
        
        self.fc1 = nn.Linear(self.input_dim + self.n_agents + self.msg_dim + self.n_actions*2, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.n_actions)
        self.fcm = nn.Linear(self.rnn_hidden_dim, self.msg_dim)

    def init_hidden(self, n_batch):
        # make hidden states on same device as model
        return self.fc1.weight.new_zeros(n_batch, self.rnn_hidden_dim)

    def forward(self, inputs, agent_id, explore_actions, message, last_action, hidden_state):
        x = torch.cat([inputs,agent_id, explore_actions, message, last_action],1)
        x = F.relu(self.fc1(x))
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        # m = F.tanh(self.fcm(h))
        m = self.fcm(h)
        return q, m, h
