from core_algorithms.genetic_agent import GeneticAgent
from pathlib import Path
import os
import numpy as np
import torch
from core_algorithms.genetic_agent import GeneticAgent, Actor
from dataclasses import dataclass
from signals.sequences import SmoothedStepSequence


@dataclass
class Stats:
    nmae: np.float16
    nmae_std: np.float16
    sm: np.float16
    sm_std: np.float16


def gen_refs(t_max: int, ampl_times: np.array, ampl_max: float, num_trials: int = 10):
    """Generate a list of reference smoothened step signals. from 0 to t_max.

    Args:
        t_max (int): Episode time.
        ampl_times (np.array): Starting times of each new step block.
        ampl_max (float): Maximum amplitude of the reference signal, symmetric wrt zero.
        num_trials (int, optional): number of random references. Defaults to 10.
    """
    refs_lst = []

    for _ in range(num_trials):
        # Possible choices:
        ampl_choices = np.linspace(-ampl_max, ampl_max, 6)

        # Generate random amplitudes:
        amplitudes = np.random.choice(ampl_choices, size=6, replace=True)
        amplitudes[0] = 0.0

        # disturb starting times for each step:
        ampl_times = [ampl_times[0]] + [
            t + np.random.uniform(-0.05, 0.05) for t in ampl_times[1:]
        ]

        # step object:
        _step = SmoothedStepSequence(
            times=ampl_times,
            amplitudes=amplitudes,
            smooth_width=t_max//10
        )
        refs_lst.append(_step)

    return refs_lst


def find_logs_path(logs_name: str, root_dir: str = './logs/wandb/'):
    cwd = os.getcwd()

    if not cwd.endswith('control'):
        pwd = Path(os.path.abspath(os.path.join(cwd, os.pardir)))
        cwd = pwd
    wandb = cwd / Path(root_dir)

    logs_name = logs_name.lower()
    for _path in wandb.iterdir():
        if _path.is_dir():
            if _path.stem.lower().endswith(logs_name):
                print(_path.stem.lower())
                return wandb / _path
    return None


def load_pop(model_path: str, args):
    """ Load evolutionary population"""
    model_path = model_path / Path('files/')
    actor_path = os.path.join(model_path, 'evolution_agents.pkl')

    agents_pop = []
    checkpoint = torch.load(actor_path)

    for _, model in checkpoint.items():
        _agent = GeneticAgent(args)
        _agent.actor.load_state_dict(model)
        agents_pop.append(_agent)

    print("Genetic actors loaded from: " + str(actor_path))

    return agents_pop


def load_rl_agent(model_path: str, args):
    """ Load RL actor from model path according to configuration."""
    model_path = model_path / Path('files/')
    actor_path = os.path.join(model_path, 'rl_agent.pkl')

    checkpoint = torch.load(actor_path)
    rl_agent = GeneticAgent(args)
    rl_agent.actor.load_state_dict(checkpoint)

    print("RL actor loaded from: " + str(actor_path))

    return rl_agent
