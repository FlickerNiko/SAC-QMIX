import torch
import numpy as np


class EpisodeBuffer:
    def __init__(self,
                 scheme,
                 buffer_size,
                 max_seq_length,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.buffer_size = buffer_size
        self.max_seq_length = max_seq_length
        self.device = device
        self._setup_data()

    def _setup_data(self):
        self.index_st = 0
        self.n_sample = 0

        self.data = {}
        for k, v in self.scheme.items():
            shape = (self.buffer_size, self.max_seq_length) + v['shape']
            self.data[k] = torch.zeros(shape, dtype=v['dtype'])

    def sample(self, batch_size):
        if self.n_sample < batch_size:
            return None

        def trans_ids(ep_id):
            ep_id += self.index_st
            ep_id %= self.buffer_size
            return ep_id

        ep_ids = np.random.choice(self.n_sample, batch_size, replace=False)
        ep_ids = list(map(trans_ids, ep_ids))

        ret = {}
        for k, v in self.data.items():
            ret[k] = v[ep_ids]

        return ret

    def add_episode(self, data):

        #len_ep = len
        if self.n_sample < self.buffer_size:
            self.n_sample += 1
        else:
            self.index_st += 1
            self.index_st %= self.buffer_size

        index_ep = (self.index_st + self.n_sample - 1) % self.buffer_size
        
        for k, v in data.items():
            ep_len = len(v)
            dtype = self.scheme[k]['dtype']
            self.data[k][index_ep].zero_()
            self.data[k][index_ep, 0:ep_len] = torch.as_tensor(v, dtype=dtype)
