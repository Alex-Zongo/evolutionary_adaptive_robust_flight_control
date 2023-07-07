import os
import argparse
import time
import random
import torch
import numpy as np
from core_algorithms.utils import load_config
from environments.aircraftenv import AircraftEnv
from environments.config import select_env
from core_algorithms.agent import Agent
from parameters import Parameters
import wandb

# -store_true means that it becomes true if I mention the argument
parser = argparse.ArgumentParser()

parser.add_argument('-env', type=str, help='Environment Choices: (LunarLanderContinuous-v2) (PHLab)',
                    default='PHlab_attitude_nominal')

parser.add_argument(
    '-should_log', help='Wether the WandB loggers are used', action='store_true')
parser.add_argument('-run_name', default='test_train', type=str)
# parser.add_argument('-env', help='Environment Choices: (LunarLanderContinuous-v2) (PHLab)',
#                     type=str, default='PHlab_attitude_nominal')
parser.add_argument(
    '-frames', help='Number of frames to learn from', type=int, required=True)

parser.add_argument(
    '-pop_size', help='Population size (if 0 only RL learns)', default=10, type=int)
parser.add_argument('-champion_target',
                    help='Use champion actor as target policy for critic update.', action='store_true')
parser.add_argument('-seed', help='Random seed to be used',
                    type=int, default=7)
parser.add_argument('-disable_cuda', help='Disables CUDA',
                    action='store_true', default=False)
parser.add_argument('-use_caps', help='Use CAPS loss regularization for smooth actions.',
                    action='store_true', default=False)
parser.add_argument(
    '-use_ounoise', help='Replace zero-mean Gaussian noise with time-correlated OU noise', action='store_true')


parser.add_argument(
    '-novelty', help='Use novelty exploration', action='store_true')
parser.add_argument(
    '-mut_type', help='Type of mutation operator', type=str, default='proximal')
parser.add_argument('-use_distil', help='Use distilation crossover',
                    action='store_true', default=False)
parser.add_argument('-distil_type', help='Use distilation crossover. Choices: (novelty)(fitness) (distance)',
                    type=str, default='fitness')

parser.add_argument('-test_ea', help='Test the EA loop and deactivate RL.',
                    default=False, action='store_true')
parser.add_argument(
    '-verbose_mut', help='Make mutations verbose', action='store_true')
parser.add_argument('-verbose_crossover',
                    help='Make crossovers verbose', action='store_true')
parser.add_argument(
    '-use_ddpg', help='Wether to use DDPG in place of TD3 for the RL part.', action='store_true')
parser.add_argument(
    '-opstat', help='Store statistics for the variation operators', action='store_true')
parser.add_argument('-test_operators',
                    help='Test the variational operators', action='store_true')

parser.add_argument(
    '-per', help='Use Prioritised Experience Replay', action='store_true')
parser.add_argument(
    '-sync_period', help="How often to sync to population", type=int, default=1)
parser.add_argument(
    '-save_periodic', help='Save actor, critic and memory periodically', action='store_true')
parser.add_argument(
    '-next_save', help='Generation save frequency for save_periodic', type=int, default=1000)

parser.add_argument('-config_path', help='Generation save frequency for save_periodic',
                    type=str, default=None)
parser.add_argument(
    '-smooth_fitness', help='Added negative smoothness penalty to the fitness.', action='store_true')

if __name__ == '__main__':

    # parameters:
    conf = parser.parse_args()

    should_log = True
    run_name = "cpu_train_attitude_incremental_1000000_frames_SERL50_caps_distil_fitness_mut_safe_smooth_fitness_paperHyperparams"
    # run_name = "test_ref_values_no_log"
    parameters = Parameters(conf=conf)
    # num_frames = 800_000  # Number of frames to learn from:
    # batch_size = 86  # Number of experiences to use for each training step:
    # buffer_size = 100_000  # Size of the replay buffer:

    # create the env:
    env = select_env(conf.env)
    # env = AircraftEnv(configuration="full_control",
    #                   render_mode=False, realtime=False, incremental=False)
    env_name = "PHlab Citation Aircraft"
    # env_name = "LunarLanderContinuous-v2"
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    if conf.config_path is not None:
        path = os.getcwd()
        pwd = os.path.abspath(os.path.join(path, os.pardir))
        config_path = pwd + conf.config_path
        config_dict = load_config(config_path)
        parameters.update_from_dict(config_dict)

    params_dict = parameters.__dict__
    # stats tracker with WandB (Weights And Biases):
    if conf.should_log:
        print("WandB logging Started")
        run = wandb.init(
            project="Intelligent_Fault_tolerant_adaptive_flight_control",
            entity="alexanicetzongo",
            dir="./logs",
            name=run_name,
            config=params_dict)
        parameters.save_foldername = str(run.dir)
        print('Saved to:', parameters.save_foldername)
        wandb.config.update({
            "save_foldername": parameters.save_foldername,
            "run_name": run.name,
        }, allow_val_change=True)

    # seed
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # create the agent:
    agent = Agent(args=parameters, environment=env)
    print(
        f" Running the environment: {parameters.env_name}\n State_dim: {parameters.state_dim}\n Action_dim: {parameters.action_dim}\n")

    # main training loop:
    start_time = time.time()
    print("num frames", parameters.num_frames)
    print("agent frames", agent.num_frames)
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

        if conf.should_log:
            wandb.log(stats)  # call to wand logger

    # save the final model:
    elite_index = stats["elite_index"]
    # save the best model. #TODO add the parameters:
    agent.save_agent(parameters, elite_index)

    if conf.should_log:
        run.finish()
