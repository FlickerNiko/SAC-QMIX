from smac.env import StarCraft2Env
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import torch
import sys
import os
from Agent import VDNAgent,MQAgent,JALAgent
from episode_buffer import EpisodeBuffer
from runnner import Runner
from learner import Learner
from controller import Controller
from writter_util import WritterUtil
from experiment import Experiment
class Args:
    pass


if __name__ == "__main__":

    
    config_path='config.json'
    new_run = True
    argv = sys.argv
    if len(argv)>1:
        config_path = argv[1]
    if len(argv)>2:
        if argv[2] == '-c':  ##continue
            new_run = False

    with open(config_path, 'r') as f:
        config = json.load(f)
    args = Args()
    args.__dict__.update(config)
    args.run_name = os.path.split(config_path)[-1][0:-5]
    args.new_run = new_run
    experiment = Experiment(args)
    experiment.start()
    experiment.run()
    

    
