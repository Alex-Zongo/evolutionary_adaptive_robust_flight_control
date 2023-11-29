
from dataclasses import dataclass
import toml
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, CompoundKernel, ConstantKernel, RationalQuadratic
import multiprocessing as mp
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import copy
import torch.multiprocessing as mp
import argparse

import numpy as np
from tqdm import tqdm
from core_algorithms.model_identification import RLS, GaussianProcess
from core_algorithms.replay_memory import IdentificationBuffer, ReplayMemory
from core_algorithms.td3 import Actor
from core_algorithms.utils import calc_nMAE, calc_smoothness, load_config

from environments.config import select_env
from evaluation_es_utils import find_logs_path, gen_eval_refs, load_agent, load_cov
from parameters_es import ESParameters
from plotters.plot_utils import plot

parser = argparse.ArgumentParser()

# ***** Arguments *****#
parser.add_argument('--env_name', help='Environment to be used: (PHLab)',
                    type=str, default='PHlab_attitude_nominal')
parser.add_argument(
    '--agent_name', help='Path to the agent to be evaluated', type=str)
parser.add_argument('--seed', help='Random seed to be used',
                    type=int, default=7)
parser.add_argument(
    '--use_mu', help='Use mu instead of best agent', action='store_true')
parser.add_argument('--use_best_mu', action='store_true')
parser.add_argument('--use_best_elite', action='store_true')
parser.add_argument('--disable_cuda', help='Disables CUDA',
                    action='store_true', default=False)
parser.add_argument('--num_trials', type=int, default=2)
parser.add_argument('--generate_sol', help='Generate solution',
                    action='store_true', default=False)
parser.add_argument('--save_plots', help='Save plots', action='store_true')
parser.add_argument('--save_stats', help='Save stats', action='store_true')
parser.add_argument('--save_trajectory',
                    help='Save trajectory', action='store_true')
parser.add_argument('--mem_size', type=int, default=100000)
# ********************* #

parsed_args = parser.parse_args()
env = select_env(parsed_args.env_name)
t_max = 100
env.set_eval_mode(t_max=t_max)
params = ESParameters(parsed_args, init=True)

# **** ENV setup:
params.action_dim = env.action_space.shape[0]
params.state_dim = env.observation_space.shape[0]
_, _, fault_name = parsed_args.env_name.split('_')

# load the agent or agents:
# if One agent: then load the cov matrix:
# Sample a list of agent actor for the task:
agents = []
n_sol = 10
# if parsed_args.generate_sol and parsed_args.agent_name:
#     # load mu agent and covariance matrix:
#     agents = generate_agents(n_sol, params, parsed_args)
# else:
# load agents: mu, best_mu, elite and best elite
# agents_dir = [
#     "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_seed0",
#     # "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_parents10_seed42",
#     "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_covSaving_seed7",

#     "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_covSaving_seed0",
#     "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_covSaving_seed0"
# ]
n_agents_name = [
    "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_parents10_seed42",
    "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_seed0",
    "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_seed0",
    # "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_seed42",
    # "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_covSaving_seed0",
]

# "mu", "elite", "best_mu", "best_elite"
agents_type = ["use_best_mu", "use_best_elite",
               "use_best_elite", "use_elite"]


def agents_(agents_name, type):
    l = []
    for name in agents_name:
        for t in type:
            l.append((name, t))

    return l


p_comb = agents_(agents_name=n_agents_name, type=[
    "use_best_mu", "use_best_elite", "use_elite", "use_mu"])
# p_comb = list(zip(agents_dir, agents_type))

for dir, type in p_comb:
    logs_dir = find_logs_path(dir)
    model_config = load_config(logs_dir)
    actor_params = copy.deepcopy(params)
    actor_params.update_from_dict(model_config)
    setattr(parsed_args, type, True)
    setattr(actor_params, 'device', torch.device("cpu"))
    actor = Actor(actor_params, init=True)
    agent = load_agent(logs_dir, actor_params, parsed_args)

    agents.append(agent)

print(">> Number of agents: ", len(agents))
##################
# 1.0*RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e1))
kernel_dict = {
    "RBF":  RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
    "RationalQ": RationalQuadratic(length_scale=1.0, alpha=1.0,  length_scale_bounds=(1e-5, 1e5)),
}


def choose_kernel(name: str):
    if not name:
        name = "RBF"
    print("choose_kernel: ", name)
    return kernel_dict[name]
#################


rls_config = dict(
    gamma=0.99,
    cov0=1e-1,
    state_size=6,
    eps_thresh=1e-3,
    seed=42,
    kernel=choose_kernel("RationalQ"),

)

# identified_model = RLS(config_dict=rls_config, env=env)
identified_model = GaussianProcess(config_dict=rls_config, env=env)
# load saved model if exists:
identified_model.reset()
t_horizon = 10.0  # 10
# establish the ref trajectory:
amp1 = [0, 12, 3, -4, -8, 2]
amp1_max = 12.0
amp_theta = [0, -2, 4, 6, -3, 2]
max_theta = 6.0
amp2 = [2, -2, 2, 10, 2, -6]
amp2_max = 10.0
amp_phi = [2, -3, 3, -4, 5, -6]
max_phi = 5.0
user_eval_refs = gen_eval_refs(
    amp_theta=amp1,
    amp_phi=amp2,
    max_theta=amp1_max,
    max_phi=amp2_max,
    t_max=t_max,
    num_trails=parsed_args.num_trials,
)
# initialize the identification model:
user_refs = {
    'theta_ref': user_eval_refs[-1][0],
    'phi_ref': user_eval_refs[-1][1],
}

obs, _ = env.reset(user_refs=user_refs)

identified_model.sync_env(env)


def child_process(shared_array, agents):
    for i, agent in enumerate(agents):
        pass


def parallel_predictive_control(controller):
    global agents
    global identified_model
    global t_horizon
    rewards, action_lst, times = identified_model.predictive_control(
        agents[controller], t_horizon=t_horizon)
    return sum(rewards)


def generate_agents(n_sol, params, parsed_args):
    np.random.seed(7)
    logs_dir = find_logs_path(parsed_args.agent_name)
    print(logs_dir)
    model_config = load_config(logs_dir)
    params.update_from_dict(model_config)
    setattr(parsed_args, 'use_best_mu', True)
    setattr(params, 'env_name', params.env_name)
    params.stdout()
    actor = Actor(params, init=True)
    mu_agent = load_agent(logs_dir, params, parsed_args)
    agents = [None] * n_sol
    agents[0] = mu_agent
    mu = mu_agent.extract_parameters().cpu().numpy()
    cov = load_cov(logs_dir)
    epsilon_half = np.random.randn(n_sol//2, len(mu))/np.sqrt(2)
    epsilon = np.concatenate([epsilon_half, -epsilon_half])
    candidates = mu + epsilon * np.sqrt(cov)
    for i in range(1, n_sol):
        actor.inject_parameters(candidates[i])
        agents[i] = actor
    return agents


def find_online_logs_path(logs_name: str = './online_eval_logs'):
    cwd = os.getcwd()
    if not cwd.endswith('control'):
        pwd = Path(os.path.abspath(os.path.join(cwd, os.pardir)))
        cwd = pwd

    online_logs = cwd / Path(logs_name)
    logs_name = logs_name.lower()
    if online_logs.is_dir():
        return online_logs
    return None


def online_eval(env, agents, identified_model, t_horizon, user_refs):
    done = False
    obs, _ = env.reset(user_refs=user_refs)
    ref_lst, errors, rewards, t_lst = [], [], [], []
    x_lst, x_ctrl_lst, u_lst = [], [], []
    rho = 0.05
    # curr_actor = generate_agents(n_sol, params, parsed_args)[0]
    curr_actor = copy.deepcopy(agents[0])
    steps, num_agent_change = 0, 0
    curr_actor_idx = 0
    while not done:
        u_lst.append(env.last_u)
        x_lst.append(env.x)
        x_ctrl_lst.append(env.get_controlled_state())
        current_agent = curr_actor.extract_parameters().cpu().numpy()
        action = curr_actor.select_action(obs)
        action = np.clip(action, -1, 1)
        # prev_model_ctrl_x = copy.copy(env.x[identified_model.ctrl_state_idx])
        ref_value = np.deg2rad(
            np.array([ref(env.t) for ref in env.ref]).flatten()
        )
        next_obs, reward, done, _ = env.step(action.flatten())

        ##########
        identified_model.update(
            obs[:identified_model.state_size],
            action,
            next_obs[:identified_model.state_size],
        )
        ##########
        # TODO: run subprocess to evaluate the agents on time horizon control:
        # agents_perf = np.zeros((n_sol, ))
        # agents_perf = mp.Array('d', n_sol)
        # processes = []
        # results = pool.starmap(parallel_eval, [(
        #     i, t_horizon, n_sol, params, parsed_args, env.t, user_refs) for i in range(n_sol)])

        # agents_perf = [sum(result[0] for result in results)]
        # print(agents_perf)
        if steps % 2500 == 0 and steps > 0:
            identified_model.sync_env(env)
            results = map(lambda controller: parallel_predictive_control(
                online_model=identified_model, controller=controller, t_horizon=t_horizon), agents)
            # print(list(results)[0])
            results = list(results)
            # print(results[0][0])
            agents_perf = [sum(result[0]) for result in results]
            #########
            idx = np.argmax(np.asarray(agents_perf))
            if curr_actor_idx != idx:
                # count the number of change of actor:
                num_agent_change += 1

                curr_actor_idx = idx
                print(idx)
        soft_switch(agents[curr_actor_idx], curr_actor, tau=0.05)

        # save the stats:
        ref_lst.append(ref_value)
        errors.append(ref_value - x_ctrl_lst[-1])
        rewards.append(reward)
        t_lst.append(env.t)

        obs = next_obs
        steps += 1
        if done:
            env.reset()

    # pool.close()
    # pool.join()
    env.close()
    # env.finish()

    # TODO: save identified model:
    identified_model.save_weights(save_dir)

    errors = np.asarray(errors)
    nmae = calc_nMAE(errors)

    actions = np.asarray(u_lst)
    smoothness = calc_smoothness(actions)

    # stats:
    rewards = np.asarray(rewards).reshape((-1, 1))
    ref_values = np.array(ref_lst)
    t_lst = np.asarray(t_lst).reshape((-1, 1))
    data = np.concatenate(
        (ref_values, actions, x_lst, rewards, t_lst), axis=1)
    # stats to print:
    fitness = np.sum(rewards) + smoothness
    print(f'Episode finished after {steps} steps.')

    print(f'Episode length: {env.t} seconds.')
    print(f'Episode Agent fitness: {fitness}')
    print(f'Episode smoothness: {smoothness}')
    print(f'Episode nMAE: {nmae}')
    print(f'Number of agent change: {num_agent_change}')
    return data, nmae, smoothness


def load_actor(dir, type):
    logs_dir = find_logs_path(dir)
    model_config = load_config(logs_dir)
    params.update_from_dict(model_config)
    setattr(parsed_args, type, True)
    agent = load_agent(logs_dir, params, parsed_args)
    print(">> Agent loaded successfully!")
    return agent


def save_trajectory(fault_path, data, agent_type):
    """save the actor trajectory data

    Args:
        fault_path (_type_): _description_
        data (_type_): _description_
    """
    save_path = fault_path / Path(agent_type + '_trajectory.csv')
    with open(save_path, 'w+', encoding='utf-8') as fp:
        np.savetxt(fp, data)

    print(f'Trajectory saved as: {save_path}')
    fp.close()


@dataclass
class OnlineStats:
    nmae: np.float16
    fitness: np.float16
    sm: np.float16
    n_change: np.int16


if __name__ == "__main__":
    # ctx = torch.multiprocessing.get_context('spawn')
    def soft_switch(target_agent, source_agent, tau):
        """
        Soft update of the target network parameters.
        θ_local = (1-τ)*θ_local + τ*θ_target
        """

        for target_param, param in zip(target_agent.parameters(), source_agent.parameters()):
            param.data.copy_(tau*target_param.data + (1.0-tau)*param.data)
            # target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

    def comp_reward(new_r, c_r):
        """_summary_

        Args:
            new_r (_type_): new policy reward
            c_r (_type_): current policy reward

        Returns:
            Wether to switch or not
        """
        return (new_r/c_r) <= 0.75

    done = False
    ref_lst, errors, rewards, t_lst = [], [], [], []
    x_lst, x_ctrl_lst, u_lst = [], [], []
    curr_agent = copy.deepcopy(agents[0])
    step = 0
    num_of_agent_change = 0
    curr_actor_idx = 0
    while not done:
        # print(obs)
        u_lst.append(env.last_u)
        x_lst.append(env.x)
        x_ctrl_lst.append(env.get_controlled_state())
        ref_value = np.deg2rad(
            np.array([ref(env.t) for ref in env.ref]).flatten()
        )

        action = curr_agent.select_action(obs)
        action = np.clip(action, -1, 1)
        next_obs, reward, done, _ = env.step(action.flatten())

        identified_model.update(
            state=obs[:identified_model.state_size],
            action=action,
            next_state=next_obs[:identified_model.state_size],
        )
        # every 5 s and after 10 s
        if step % 200 == 0 and step >= 0:
            p = mp.Pool(len(agents))
            identified_model.sync_env(env)
            # print(identified_model.env.t)

            # p = ctx.Pool(4)
            r = p.map(parallel_predictive_control, list(range(len(agents))))
            idx = np.argmax(r)
            if curr_actor_idx != idx and comp_reward(r[idx], r[curr_actor_idx]):
                num_of_agent_change += 1
                print(idx, r[idx]/r[curr_actor_idx])
                curr_actor_idx = idx

        soft_switch(agents[curr_actor_idx], curr_agent, tau=0.00005)  # 0.00005

        # save the stats:
        ref_lst.append(ref_value)
        errors.append(ref_value - x_ctrl_lst[-1])
        rewards.append(reward)
        t_lst.append(env.t)

        obs = next_obs
        step += 1
        if done:
            env.reset()
            print(">> Env reset!")

    env.close()

    errors = np.asarray(errors)
    nmae = calc_nMAE(errors)

    actions = np.asarray(u_lst)
    smoothness = calc_smoothness(actions)

    # stats:
    rewards = np.asarray(rewards).reshape((-1, 1))
    ref_values = np.array(ref_lst)
    t_lst = np.asarray(t_lst).reshape((-1, 1))
    data = np.concatenate(
        (ref_values, actions, x_lst, rewards, t_lst), axis=1)
    # stats to print:
    fitness = np.sum(rewards) + smoothness
    print(f'Episode finished after {step} steps.')

    print(f'Episode length: {t_lst[-1]} seconds.')
    print(f'Episode Agent fitness: {fitness}')
    print(f'Episode smoothness: {smoothness}')
    print(f'Episode nMAE: {nmae}')
    print(f'Number of agent change: {num_of_agent_change}')

    faults = {
        'nominal': 'Normal Flight',
        'ice': 'Iced Wing',
        'cg-shift': 'Shifted CG',
        'sa': 'Saturated Aileron',
        'se': 'Saturated Elevator',
        'be': 'Broken Elevator',
        'jr': 'Jammed Rudder',
        'high-q': 'High Q',
        'low-q': 'Low Q',
        'noise': 'Sensor Noise',
        'gust': 'Gust of Wind',
        'cg-for': 'Forward Shifted CG',
        'cg': 'Backward Shifted CG',
    }

    fig, _ = plot(
        data, name=f'Fault: {faults[fault_name]} |  nMAE: {nmae:0.2f}% - Smoothness: {smoothness:0.2f} rad.Hz | Num-Switch: {num_of_agent_change}  ', fault=fault_name
    )

    fig_path = find_online_logs_path() / Path('figures')
    fault_path = fig_path / Path(fault_name)
    if parsed_args.save_plots:
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        if not os.path.exists(fault_path):
            os.mkdir(fault_path)
        fig_name = fault_path / \
            Path(fault_name + '.png')

        fig.savefig(fname=fig_name, dpi=300, format='png')
        plt.close()
    else:
        plt.show()

    if parsed_args.save_trajectory:
        save_trajectory(fault_path, data, agent_type="online_adaptation")

    # if parsed_args.save_stats:
    #     stats = OnlineStats(
    #         nmae=nmae,
    #         fitness=fitness,
    #         sm=smoothness,
    #         n_change=num_of_agent_change,
    #     )

    #     toml_path = find_online_logs_path() / Path(fault_name)
    #     stats_dict = {fault_name: stats.__dict__}

    #     with open(os.path.join(toml_path, 'stats.toml'), 'w', encoding='utf-8') as f:
    #         f.write('\n\n')
    #         toml.dump(stats_dict, f, encoder=toml.TomlNumpyEncoder())

    #     f.close()
