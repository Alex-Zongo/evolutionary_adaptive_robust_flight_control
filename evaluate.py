import warnings
import toml
from copy import deepcopy, copy
import os
from pathlib import Path
from evaluation_utils import find_logs_path, gen_refs, load_pop, load_rl_agent
from signals.sequences import SmoothedStepSequence
import random
import argparse
import numpy as np
from core_algorithms.utils import calc_nMAE, calc_smoothness, load_config
from parameters import Parameters
import matplotlib.pyplot as plt
from environments.aircraftenv import AircraftEnv
from environments.config import select_env
from tqdm import tqdm
from evaluation_utils import Stats
from parameters import Parameters
import torch
from plotters.plot_utils import plot


parser = argparse.ArgumentParser()

############ Arguments #############
parser.add_argument('-env', help='Environment to be used: (PHLab)',
                    type=str, default='PHlab_attitude_nominal')  # TODO: change full state to attitude control
parser.add_argument('-disable_cuda', help='Disables CUDA',
                    action='store_true', default=False)
parser.add_argument('-num_trials', type=int, default=1)
parser.add_argument(
    '-agent_name', help='Path to the agent to be evaluated', type=str, required=True)
parser.add_argument('-seed', help='Random seed to be used',
                    type=int, default=7)
parser.add_argument('-agent_index', type=int)
parser.add_argument('-eval_rl', default=False, action='store_true')
parser.add_argument('-eval_pop', default=False, action='store_true')
parser.add_argument('-eval_actor', default=False, action='store_true')
parser.add_argument('-plot_spectra', default=False, action='store_true')
parser.add_argument('-save_plots', default=False, action='store_true')
parser.add_argument('-save_trajectory', default=False, action='store_true')
parser.add_argument('-save_stats', default=False, action='store_true')
parser.add_argument('-verbose', default=False, action='store_true')
############################################

parsed_param, unknown = parser.parse_known_args()

params = Parameters(conf=parsed_param)

t_max = 80

env = select_env(parsed_param.env)
# env = AircraftEnv(configuration="full_control")
env.set_eval_mode(t_max=t_max)


def evaluate(actor, **kwargs: dict):
    """Simulate one episode in the environment with the given actor"""

    # reset the environment:
    done = False
    obs = env.reset(**kwargs)

    # total number of steps is at most t_max//dt:
    x_lst, rewards, u_lst = [], [], []
    x_ctrl_lst = []
    errors = []
    ref_lst = []
    t_lst = []

    while not done:
        x_lst.append(env.x)
        u_lst.append(env.last_u)
        x_ctrl_lst.append(env.get_controlled_state())

        # select the action:
        action = actor.select_action(obs)

        # simulate one step into the env:
        # clip the action to -1, 1
        action = np.clip(action, -1, 1)
        ref_value = np.deg2rad(
            np.array([ref(env.t) for ref in env.ref]).flatten())
        next_obs, reward, done, _ = env.step(action.flatten())

        if kwargs.get('verbose'):
            print(
                f'Action: {np.rad2deg(action)} -> deflection: {np.rad2deg(env.last_u)}')
            print(f't:{env.t:0.2f} theta:{env.theta:.03f} q:{env.q:.03f} phi:{env.phi:.03f} alpha:{env.alpha:.03f} V:{env.V:.03f} H:{env.h:.03f}')
            print(
                f'Error: {np.rad2deg(obs[:env.n_actions])} Reward: {reward:.03f} \n\n')

        # Update:
        obs = next_obs

        # save:
        ref_lst.append(ref_value)
        errors.append(ref_value - x_ctrl_lst[-1])
        rewards.append(reward)
        t_lst.append(env.t)

    env.finish()

    # control inputs
    errors = np.asarray(errors)

    # compute the scaled smoothness fitness:
    actions = np.asarray(u_lst)
    smoothness = calc_smoothness(actions, **kwargs)

    # format data:
    rewards = np.asarray(rewards).reshape((-1, 1))
    ref_values = np.array(ref_lst)
    t_lst = np.asarray(t_lst).reshape((-1, 1))
    data = np.concatenate((ref_values, actions, x_lst, rewards, t_lst), axis=1)

    # calculate nMAE:
    nmae = calc_nMAE(errors)

    return data, nmae, smoothness


def validate_agent(gen_agent, user_ref_lst: list, num_trials: int = 1, **kwargs):
    """ Evaluate teh agent over the given reference signals and number of trials
    Note that the time traces from data come from the LAST EPISODE
    """
    agent_nmae_lst, agent_sm_lst = [], []

    for i in tqdm(range(num_trials+1), total=num_trials):
        ref_t = user_ref_lst[i]
        user_refs = {
            'theta_ref': ref_t[0],
            'phi_ref': ref_t[1],
            # 'psi_ref': ref_t[2],
        }
        data, nmae, smoothness = evaluate(
            gen_agent.actor, user_refs=user_refs, **kwargs)
        agent_nmae_lst.append(nmae)
        agent_sm_lst.append(smoothness)

    nmae = np.mean(agent_nmae_lst)
    nmae_std = np.std(agent_nmae_lst)
    smoothness = np.mean(agent_sm_lst)
    sm_std = np.std(agent_sm_lst)

    if kwargs.get('stdout'):
        print(f'nMAE: {nmae:0.1f}% with STD: {nmae_std:0.1f}')
        print(f'Smoothness: {smoothness:0.0f} with STD: {sm_std:0.1f}')

    stats = Stats(nmae, nmae_std, smoothness, sm_std)
    return data, stats


def main():
    # seed:
    # env.seed(params.seed)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    params.action_dim = env.action_space.shape[0]
    params.state_dim = env.observation_space.shape[0]

    # TODO: Identify the fault:
    _, _, fault_name = parsed_param.env.split('_')

    ''' Prepare References '''
    # Time set points:
    time_array = np.linspace(0., t_max, 6)

    # pitch set-points:
    amp1 = [0, 12, 3, -4, -8, 2]
    base_ref_theta = SmoothedStepSequence(
        times=time_array, amplitudes=amp1, smooth_width=t_max//10)

    # Roll set-points:
    amp2 = [2, -2, 2, 10, 2, -6]
    base_ref_phi = SmoothedStepSequence(
        times=time_array, amplitudes=amp2, smooth_width=t_max//10)

    # build list of reference tuples from seed:
    theta_refs = gen_refs(t_max=t_max, ampl_times=time_array,
                          ampl_max=12.0, num_trials=parsed_param.num_trials)

    phi_refs = gen_refs(t_max, time_array, 10.0,
                        num_trials=parsed_param.num_trials)
    theta_refs.append(base_ref_theta)
    phi_refs.append(base_ref_phi)

    user_eval_refs = list(zip(theta_refs, phi_refs))

    # Load agent info
    logs_dir = find_logs_path(parsed_param.agent_name)
    model_config = load_config(logs_dir)

    # build path to figures:
    fig_path = logs_dir / Path('figures')
    fault_path = fig_path / Path(fault_name)

    # update config:
    params.update_from_dict(model_config)
    setattr(params, 'env_name', params.env_name)
    params.stdout()

    '''         Load Population         '''
    if parsed_param.eval_pop or parsed_param.eval_actor:
        pop = load_pop(logs_dir, args=params)

    if parsed_param.eval_actor:
        # evaluate only one actor:
        if parsed_param.agent_index is not None:
            idx = parsed_param.agent_index
        else:
            idx = int(input('Enter the actor index to evaluate: '))

        data, stats = validate_agent(
            pop[idx], user_eval_refs, num_trials=parsed_param.num_trials, stdout=True, plot_spectra=parsed_param.plot_spectra)

        _fig, _ = plot(
            data, name=f'nMAE: {stats.nmae:0.1f}% Smoothness: {stats.sm:0.0f} rad.Hz', fault=fault_name)

        if parsed_param.save_plots:
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            if not os.path.exists(fault_path):
                os.mkdir(fault_path)
            fig_name = fault_path / \
                Path(f'actor{idx}'+'_' + fault_name + '.png')
            _fig.savefig(fname=fig_name, dpi=300, format='png')
            plt.close()
        else:
            plt.show()

        # save trajectories on base reference:
        if parsed_param.save_trajectory:
            save_trajectory(fault_path, data)

    elif parsed_param.eval_pop:
        """Evaluate the entire population"""
        nmae_lst, sm_lst = [], []
        nmae_min = 500

        for i, agent in enumerate(pop):
            print('Actor: ', i)
            data, stats = validate_agent(
                agent, user_eval_refs, parsed_param.num_trials, stdout=parsed_param.verbose)

            # save for logging:
            sm_lst.append(stats.sm)
            nmae_lst.append(stats.nmae)

            # check for champion:
            if stats.nmae < nmae_min:
                nmae_min = stats.nmae
                champion_data = copy(data)
                champion_stats = copy(stats)
                idx = i

        # plot the champion actor:
        print('Champion', idx)
        champ_fig, _ = plot(
            champion_data, name=f'nMAE: {champion_stats.nmae:0.1f}% Smoothness: {champion_stats.sm:0.0f} rad.Hz', fault=fault_name)

        if parsed_param.save_plots:
            # TODO:
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            if not os.path.exists(fault_path):
                os.mkdir(fault_path)
            fig_name = fault_path / \
                Path('champion_' + 'actor' +
                     f'{idx}' + '_' + fault_name + '.png')
            champ_fig.savefig(fname=fig_name, dpi=300, format='png')
            plt.close()
        else:
            plt.show()

        print('Population stats:')
        print(
            f'Average nMAE: {np.mean(nmae_lst):0.1f} with SD: {np.std(nmae_lst):0.1f}')
        print(
            f'Average Smoothness: {np.mean(sm_lst):0.0f} with SD: {np.std(sm_lst):0.0f}')

        ''' Save data to files '''
        if parsed_param.save_stats:
            save_path = fault_path / Path('final_performance.csv')
            with open(save_path, 'w+', encoding='utf-8') as fp:
                for _sm, _nmae in zip(sm_lst, nmae_lst):
                    fp.write(f'{_sm}, {_nmae}\n')  # save to csv:
            fp.close()

            # full statistics:
            toml_path = logs_dir / Path('stats.toml')
            stats_dict = {fault_name: {
                'champion_idx': idx,
                'champion': champion_stats.__dict__,
                'average': {
                    'nmae': np.mean(nmae_lst),
                    'nmae_sd': np.std(nmae_lst),
                    'sm': np.mean(sm_lst),
                    'sm_sd': np.std(sm_lst),
                }
            }}

            with open(toml_path, 'a+', encoding='utf-8') as f:
                toml.dump(stats_dict, f, encoder=toml.TomlNumpyEncoder())
                f.write('\n\n')
            f.close()

    elif parsed_param.eval_rl:
        '''    RL Evaluation   '''
        rl_parameters = deepcopy(params)
        rl_parameters.update_from_dict(model_config)

        rl_agent = load_rl_agent(logs_dir, args=rl_parameters)
        data_rl, stats_rl = validate_agent(
            rl_agent, user_eval_refs, parsed_param.num_trials, stdout=True, plot_spectra=parsed_param.plot_spectra)

        # plot the RL actor:
        rl_fig, _ = plot(
            data_rl, name=f'nMAE: {stats_rl.nmae:0.1f}% Smoothness: {stats_rl.sm:0.0f} rad.Hz', fault=fault_name)

        if parsed_param.save_plots:
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            if not os.path.exists(fault_path):
                os.mkdir(fault_path)
            fig_name = fault_path / \
                Path('rl_' + fault_name + '.png')
            rl_fig.savefig(fname=fig_name, dpi=300, format='png')
            plt.close()
        else:
            plt.show()

        '''   Save data files  '''
        if parsed_param.save_stats:
            toml_path = logs_dir / Path('stats.toml')
            stats_dict = {fault_name: stats_rl.__dict__}

            with open(toml_path, 'a+', encoding='utf-8') as f:
                f.write('\n\n')
                toml.dump(stats_dict, f, encoder=toml.TomlNumpyEncoder())

            f.close()

        # Save trajectories on base reference:
        if parsed_param.save_trajectory:
            save_trajectory(fault_path, data_rl)

    else:
        warnings.warn(
            'The command does not specify which part of the agent to evaluate. Please add one of the following args: [-eval_pop] [-eval_rl] [-eval_actor]')


def save_trajectory(fault_path, data):
    """save the actor trajectory data

    Args:
        fault_path (_type_): _description_
        data (_type_): _description_
    """
    save_path = fault_path / Path('nominal_trajectory.csv')
    with open(save_path, 'w+', encoding='utf-8') as fp:
        np.savetxt(fp, data)

    print(f'Trajectory saved as: {save_path}')
    fp.close()


if __name__ == "__main__":
    main()
