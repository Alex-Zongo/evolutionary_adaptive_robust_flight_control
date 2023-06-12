import os
import time
import random
import torch
import numpy as np
from environments.aircraftenv import AircraftEnv
from core_algorithms.agent import Agent
from parameters import Parameters
import wandb

if __name__ == '__main__':

    # parameters:
    should_log = True
    run_name = "test_train"
    parameters = Parameters()
    # num_frames = 800_000  # Number of frames to learn from:
    # batch_size = 86  # Number of experiences to use for each training step:
    # buffer_size = 100_000  # Size of the replay buffer:

    # create the env:
    env = AircraftEnv(configuration="full_control",
                      render_mode=False, realtime=False, incremental=False)
    env_name = "Citation Aircraft"
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    params_dict = parameters.__dict__

    # stats tracker with WandB (Weights And Biases):
    if should_log:
        print("WandB logging Started")
        run = wandb.init(
            project="Intelligent_Fault_tolerant_adaptive_flight_control",
            entity="alexanicetzongo",
            dir="./logs",
            name=run_name,
            config=params_dict)
        parameters.save_foldername = str(run.dir)
        wandb.config.update({
            "save_foldername": parameters.save_foldername,
            "run_name": run.name,
        }, allow_val_change=True)

    # seed
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # create the agent:
    agent = Agent(args=parameters, env=env)
    print(
        f" Running the environment: {env_name}\n State_dim: {parameters.state_dim}\n Action_dim: {parameters.action_dim}\n")

    # main training loop:
    start_time = time.time()
    while agent.num_frames <= parameters.num_frames:
        # evaluate over all episodes and return stats:
        stats = agent.train()

        # print some of the stats:
        print('Episodes: ', agent.num_episodes, '\nFrames: ', agent.num_frames,
              '\nTrain Max: ', '%.2f' % stats['best_train_fitness'] if stats['best_train_fitness'] is not None else None,
              '\nTest Max: ', '%.2f' % stats['test_score'] if stats['test_score'] is not None else None,
              '\nTest SD: ', '%.2f' % stats['test_sd'] if stats['test_sd'] is not None else None,
              '\nPopulation Avg: ', '%.2f' % stats['pop_avg'],
              '\nWeakest :', '%.2f' % stats['pop_min'],
              '\nNovelty :', '%.2f' % stats['pop_novelty'],
              '\n',
              '\nAvg. ep. len:', '%.2fs' % stats['avg_ep_len'],
              '\nRL Reward:', '%.2f' % stats['rl_reward'],
              '\nPG Objective:', '%.4f' % stats['PG_obj'],
              '\nTD Loss:', '%.4f' % stats['TD_loss'],
              '\n')

        # update the loggers and stats:
        stats['frames'] = agent.num_frames
        stats['episodes'] = agent.num_episodes
        stats['time'] = time.time() - start_time
        if len(agent.pop):
            stats['rl_elite_fraction'] = agent.evolver.selection_stats['elite'] / \
                agent.evolver.selection_stats['total']
            stats['rl_selected_fraction'] = agent.evolver.selection_stats['selected'] / \
                agent.evolver.selection_stats['total']
            stats['rl_discarded_fraction'] = agent.evolver.selection_stats['discarded'] / \
                agent.evolver.selection_stats['total']

        if should_log:
            wandb.log(stats)  # call to wand logger

    # save the final model:
    elite_index = stats["elite_index"]
    # save the best model. #TODO add the parameters:
    agent.save_agent(parameters, elite_index)

    if should_log:
        run.finish()
