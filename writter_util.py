import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

class WritterUtil:
    def __init__(self, writter: SummaryWriter, args):
        self.writter = writter
        self.log_every = args.log_every
        self.scalars = {}
        self.step = 0
        self.log_step = 0

    def set_step(self, step):
        self.step += 1
        self.log_step = step

    def WriteScalar(self, tag, value):
        
        step = self.step
        log_step = self.log_step
        self.writter.add_scalar('raw/'+tag, value, log_step)

        if tag not in self.scalars:
            self.scalars[tag] = [0, step-1, torch.zeros(self.log_every)]
        
        n, last_step, buffer = self.scalars[tag]
        
        buffer[n] = value
        n += 1
        if step - last_step >= self.log_every:
            mean = torch.mean(buffer[0:n])
            self.writter.add_scalar('mean/'+tag, mean, log_step)
            self.writter.add_histogram(tag, buffer[0:n], log_step)
            
            last_step = step
            n=0


        self.scalars[tag] = [n,last_step,buffer]
        
        