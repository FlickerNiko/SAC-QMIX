from smac.env import StarCraft2Env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import torch
import sys
import os
from SAC.experiment import Experiment

class Args:
    pass


if __name__ == "__main__":

    
    config_path='config.json'
    new_run = True
    argv = sys.argv
    if len(argv)>1:
        if argv[1] == '-c':  ##continue
            new_run = False

    with open(config_path, 'r') as f:
        config = json.load(f)
    args = Args()
    args.__dict__.update(config)    
    args.new_run = new_run
    experiment = Experiment(args)
    experiment.start()
    experiment.run()
    

    
