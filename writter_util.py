import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

class WritterUtil:
    def __init__(self, writter: SummaryWriter, args):
        self.writter = writter
        self.log_every = args.log_every
        self.scalars = {}
    def WriteModel(self, tag, model, step):
        # mean_dict = {}
        # std_dict = {}
        # mean_grad_dict = {}
        # std_grad_dict = {}

        for name, paramater in model.named_parameters():
            data = paramater
            # std, mean = torch.std_mean(data)
            
            self.writter.add_histogram('param/'+ tag + '/' + name, data, step)
            
            # mean_dict[name] = mean
            # std_dict[name] = std
            
            grad = data.grad
            if grad != None:
                # std_grad, mean_grad = torch.std_mean(grad)
                self.writter.add_histogram('grad/'+ tag + '/' + name, grad, step)
                
                # mean_grad_dict[name] = mean_grad
                # std_grad_dict[name] = std_grad


        # self.writter.add_scalars('param/'+tag+'/mean', mean_dict, step)
        # self.writter.add_scalars('param/'+tag+'/std', std_dict, step)
        # if len(mean_grad_dict):
        #     self.writter.add_scalars('grad/'+tag+'/mean', mean_grad_dict, step)
        # if len(std_grad_dict):
        #     self.writter.add_scalars('grad/'+tag+'/std', std_grad_dict, step)


    def WriteScalar(self, tag, value, step):

        self.writter.add_scalar('raw/'+tag, value, step)

        if tag not in self.scalars:
            self.scalars[tag] = [0, step-1, torch.zeros(self.log_every)]
        
        n, last_step, buffer = self.scalars[tag]
        
        buffer[n] = value
        n += 1
        if step - last_step >= self.log_every:
            mean = torch.mean(buffer[0:n])
            self.writter.add_scalar('mean/'+tag, mean, step)
            self.writter.add_histogram(tag, buffer[0:n], step)
            
            last_step = step
            n=0


        self.scalars[tag] = [n,last_step,buffer]
        
        