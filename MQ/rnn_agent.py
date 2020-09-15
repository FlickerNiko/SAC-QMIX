import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.input_dim + args.msg_dim + args.n_actions, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.fcm = nn.Linear(args.rnn_hidden_dim, args.msg_dim)
    def init_hidden(self, n_batch):
        # make hidden states on same device as model
        return self.fc1.weight.new_zeros(n_batch, self.args.rnn_hidden_dim)

    def forward(self, inputs, message, last_action, hidden_state):
        x = torch.cat([inputs,message, last_action],1)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        m = F.tanh(self.fcm(h))
        return q, m, h
