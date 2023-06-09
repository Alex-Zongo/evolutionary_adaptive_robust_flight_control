import os
import time
import random
import torch
import numpy as np
from environments.aircraftenv import AircraftEnv
from core_algorithms.agent import Agent
from parameters import Parameters

if __name__ == '__main__':

    # parameters:
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

    # seed
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # create the agent:
    agent = Agent()
    print(f" Running the environment: {env_name}")

    # main training loop:
    start_time = time.time()
    while agent.num_frames <= parameters.num_frames:
        # evaluate over all episodes and return stats:
        stats = agent.train()

        # print some of the stats:
        print("Printing the stats...")

        # update the loggers and stats:

    # save the final model:
    elite_index = stats["elite_index"]
    # save the best model. #TODO add the parameters:
    agent.save_agent(elite_index)
