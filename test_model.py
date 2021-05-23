import torch
import numpy as np
import json
import sys
import os
from smac.env import StarCraft2Env
from SAC.runnner import Runner
from SAC.controller import Controller
from SAC.actors import Actors





class Args:
    pass


if __name__ == "__main__":

    config_path='test_config.json'
        
    with open(config_path, 'r') as f:
        config = json.load(f)

    args = Args()
    args.__dict__.update(config)
        
    env = StarCraft2Env(map_name=args.map_name, window_size_x=640, window_size_y=480)
    env_info = env.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.obs_dim = env_info['obs_shape']
    args.input_dim = args.obs_dim
    if args.agent_id:
        args.input_dim += args.n_agents
    if args.last_action:        
        args.input_dim += args.n_actions        
    args.state_dim = env_info['state_shape']
    args.episode_limit = env_info['episode_limit']

    actor = Actors(args)
    actor.to(device=args.device)
    path_model = os.path.join('models', args.run_name + '.tar')
    state_dict = torch.load(path_model, args.device)
    actor.load_state_dict(state_dict)
    controller = Controller(actor,args)
    runner = Runner(env, controller, args)
        
    win_count = 0
    reward_avg = 0
    test_count = args.test_count
    for i in range(test_count):
        _, episode_reward, win_tag, _ =  runner.run(test_mode=True)
        print("Episode {}, reward = {}".format(i, episode_reward))
        if win_tag:
            win_count += 1
        reward_avg += episode_reward                
        
    win_rate = win_count/test_count       
    reward_avg /= test_count    
    print('Test reward = {}, win_rate = {}'.format(reward_avg, win_rate))
    if args.save_replay:
        env.save_replay()
