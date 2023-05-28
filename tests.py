from environments.lunarlander import LunarLanderWrapper
from environments.aircraftenv import AircraftEnv
import torch.nn as nn
import torch
import random
import numpy as np


class Actor:
    def __init__(self):
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        print(self.device)

    def select_action(self, state: torch.tensor):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = np.array([float(random.randint(-10, 10)),
                          float(random.randint(-10, 10)), float(random.randint(-10, 10)), float(random.randint(-10, 10))])/10
        return action


if __name__ == '__main__':
    # env = LunarLanderWrapper()
    # actor = Actor()

    # print(env.simulate(actor, render=True))
    env = AircraftEnv(configuration="full_control",
                      render_mode=True, realtime=True, incremental=False)
    env.reset()

    print(env.step(np.array([0.1, 0.1, 0.05])))
