import os
import torch
import numpy as np
from core_algorithms.utils import Episode, calc_smoothness
from parameters import Parameters
from typing import List, Dict, Tuple
from core_algorithms.replay_memory import ReplayMemory, PrioritizedReplayMemory
from environments.aircraftenv import AircraftEnv
from tqdm import tqdm
from core_algorithms.genetic_agent import GeneticAgent, Actor
from core_algorithms.model_utils import OUNoise, GaussianNoise
from core_algorithms.ddpg import DDPG
from core_algorithms.td3 import TD3
from core_algorithms.neuro_evo_model import SSNE


class Agent:
    """ Intelligent Agent controller managing both the population and the RL agent"""

    def __init__(self, args: Parameters, environment: AircraftEnv):
        self.params = args
        self.env = environment

        # define Rl agent:
        # initialize the RL agent: #TODO: TD3, DDPG, PPO or SAC
        self.rl_agent = TD3(args)

        # initialize the population:
        self.pop: List = []
        # TODO: replace None with the genetic agent Actor algorithm
        self.pop = [GeneticAgent(args) for _ in range(self.params.pop_size)]

        # define the Memory or replay buffer:
        if self.params.per:
            self.replay_buffer = PrioritizedReplayMemory(
                self.params.buffer_size, self.params.device, beta_frames=self.params.num_frames)
        else:
            self.replay_buffer = ReplayMemory(
                capacity=self.params.buffer_size, device=self.params.device)

        # TODO: define noise process:
        if self.params.use_ounoise:
            self.noise_process = OUNoise(args.action_dim)
        else:
            self.noise_process = GaussianNoise(
                args.action_dim, std=args.noise_sd)

        # TODO: initialise the evolutionary loop:
        if self.pop:
            # do something:
            self.evolver = SSNE(
                self.params, self.rl_agent.critic, self.evaluate)

        # TODO: testing:
        self.validation_tests = 5

        # trackers: what to track basically:
        self.num_episodes = 0
        self.num_frames = 0
        self.iterations = 0
        self.gen_frames = None
        self.rl_history = None
        self.rl_iteration = 0
        self.champion: GeneticAgent = None  # TODO: add a type genetic agent:
        # TODO: add a type genetic agent Actor:
        self.champion_actor: Actor = None
        self.champion_history: np.ndarray = None

    # TODO: add type to the "agent" param -> [GeneticAgent, TD3, SAC or DDPG]
    def evaluate(self, agent: TD3 or DDPG or GeneticAgent, is_action_noise: bool, store_transition: bool):
        """
        Play one game to evaluate the agent:
        Args:
            agent:
            is_action_noise: whether to add noise to action or not
            store_transition: add frames to memory buffer for training:
        Returns:
            Episode: data class with stats:
        """

        # init states, env
        state_lst, rewards, action_lst = [], [], []
        obs, _ = self.env.reset()
        done = False

        # set action into evaluation mode:
        agent.actor.eval()  # TODO: set the actor model into evaluation mode:

        while not done:
            # pick an action:
            action = agent.actor.select_action(obs)

            # add exploratory noise:
            if is_action_noise:
                clipped_noise = np.clip(self.params.noise_sd*np.random.randn(
                    action.shape[0]), -self.params.noise_clip, self.params.noise_clip)
                action = np.clip(action + clipped_noise, -1.0, 1.0)

            # execute action: simulate one step into the environment:
            if 'lunar' in self.params.env_name.lower():
                next_obs, reward, done, truncated, info = self.env.step(
                    action.flatten())
                action_lst.append(action.flatten())
            else:
                next_obs, reward, done, info = self.env.step(action.flatten())
                action_lst.append(self.env.last_u)  # actuator deflection:
            rewards.append(reward)

            # add experiences to buffer:
            if store_transition:
                transition = (obs, action, next_obs, reward, float(done))
                self.replay_buffer.push(*transition)
                agent.buffer.push(*transition)

                # check the cost: for pre/stall  and high-bank dynamics:
                if info["cost"]:
                    agent.critical_buffer.push(*transition)

                self.num_frames += 1
                self.gen_frames += 1
            else:
                # save for future validation:
                if 'ph' in self.params.env_name.lower():
                    state_lst.append(self.env.x)
                else:
                    state_lst.append(obs)

            # update the agent obs:
            obs = next_obs

        self.env.finish()  # end the env:

        # TODO: if is_done then update the num of episodes:
        if store_transition:
            self.num_episodes += 1

        # compute the smoothness:
        actions = np.asarray(action_lst)
        smoothness = calc_smoothness(actions, plot_spectra=False)
        fitness = np.sum(rewards)

        # TODO: Important: Use a smoothness based fitness: study the effect of this: (the operation of ADDING that value to the reward) Could it better to multiply it by value and add it later
        if self.params.smooth_fitness:
            fitness += smoothness
        # print(info['ref'])  # TODO: check the ref and remove this
        episode = Episode(fitness=fitness, smoothness=smoothness,
                          length=info['t'], state_history=state_lst, ref_signals=info['ref'], actions=actions, reward_lst=rewards)
        return episode

    def rl_to_evo_loop(self, rl_agent: DDPG or TD3, evo_net: GeneticAgent):
        for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
            target_param.data.copy_(param.data)
        evo_net.buffer.reset()
        evo_net.buffer.push_content_of(rl_agent.buffer)
        evo_net.critical_buffer.reset()
        evo_net.critical_buffer.push_content_of(rl_agent.critical_buffer)

    def evo_to_rl_loop(self, rl_agent, evo_net):
        for target_param, param in zip(rl_agent.actor.parameters(), evo_net.actor.parameters(),):
            target_param.data.copy_(param.data)

    def get_pop_novelty(self, bcs: np.ndarray):
        return np.sum(np.std(bcs, axis=0))/bcs.shape[1]

    def train_rl(self, rl_transitions: int) -> Dict[float, float]:
        """ Train the RL agent on the same number of frames seen by the entire population. The frames are sampled from the common replay buffer. """

        policy_grad_loss, TD_loss = [], []  # TODO: the policy gradient loss and TD error

        if len(self.replay_buffer) > self.params.learn_start:
            # start training only when enough frames collected:
            self.rl_agent.actor.train()  # enable training mode:

            # select target policy:
            if self.params.use_champion_target:
                if self.champion_actor is not None:  # TODO: safeguard for RL-only runs
                    self.evo_to_rl_loop(
                        self.rl_agent.actor_target, self.champion_actor)

            # train over generation experiences:
        for _ in tqdm(range(int(rl_transitions*self.params.frac_frames_train)), desc='Train RL', colour='blue'):
            self.rl_iteration += 1
            batch = self.replay_buffer.sample(self.params.batch_size)
            pgl, td_loss = self.rl_agent.update_parameters(
                batch, self.rl_iteration, self.params.use_champion_target)

            if pgl is not None:
                policy_grad_loss.append(-pgl)
            if td_loss is not None:
                TD_loss.append(td_loss)

        return {'PG_obj': np.mean(policy_grad_loss), 'TD_loss': np.median(TD_loss)}

    def validate_agent(self, agent: Actor):
        """ Evaluate the given agent actor and do not store transitions.

        Args:
            agent (Actor): the agent to evaluate.
        Returns:
            Tuple[float, float, float, float, float, float, Episode]: the test score, test score std, episode length, episode length std, smoothness, smoothness std and the last episode.
        """
        test_scores, episode_lengths, smoothness_lst = [], [], []

        for _ in range(self.validation_tests):
            last_episode = self.evaluate(
                agent, is_action_noise=False, store_transition=False)
            test_scores.append(np.sum(last_episode.reward_lst))
            episode_lengths.append(last_episode.length)
            smoothness_lst.append(last_episode.smoothness)

        test_score = np.mean(test_scores)
        test_sd = np.std(test_scores)
        ep_len = np.mean(episode_lengths)
        ep_len_sd = np.std(episode_lengths)
        sm = np.median(smoothness_lst)
        sm_sd = np.std(smoothness_lst)

        return (test_score, test_sd, ep_len, ep_len_sd, sm, sm_sd, last_episode)

    def train(self):
        self.iterations += 1

        # Initialize local trackers:
        self.gen_frames = 0
        best_train_fitness = 1
        worst_train_fitness = 1.0
        population_avg = 1.
        test_score = 1.0
        test_sd = -1.0
        sm = 1.0
        sm_sd = -1.0
        elite_index = -1
        pop_novelty = -1.0
        lengths = []

        '''++++++++++++++ Evolution +++++++++++++'''
        if self.pop:  # TODO: what does this means??
            fitness_lst = np.zeros(
                (self.params.num_evals, self.params.pop_size))
            smoothness_lst = []

            # Evaluate the individual agents/ genomes:
            for j, net in tqdm(enumerate(self.pop), total=len(self.pop), desc='Population Evaluation', colour='green'):
                for i in range(self.params.num_evals):
                    episode = self.evaluate(
                        net, is_action_noise=False, store_transition=(i == self.params.num_evals-1))  # only store the transitions of last episode

                    smoothness_lst.append(episode.smoothness)
                    lengths.append(episode.length)
                    fitness_lst[i, j] = episode.fitness

            # average of the stats:
            pop_fitness = np.mean(fitness_lst, axis=0)
            sm = np.mean(smoothness_lst)
            sm_sd = np.std(smoothness_lst)
            ep_len_avg = np.mean(lengths)
            ep_len_sd = np.std(lengths)

            # get population stats:
            # champion - highest fitness:
            best_train_fitness = np.max(pop_fitness)
            # worst - lowest fitness:
            worst_train_fitness = np.min(pop_fitness)
            # average fitness of the whole population:
            population_avg = np.mean(pop_fitness)
            # get the champion:
            self.champion = self.pop[np.argmax(pop_fitness)]
            # get the champion's actor:
            self.champion_actor = self.champion.actor

            # validation test for NeuroEvolution:
            test_score, test_sd, _, _, _, _, last_episode = self.validate_agent(
                self.champion)
            if self.params.should_log:
                self.champion_history = last_episode.get_history()

            # NeuroEvolution probabilistic selection and recombination step:
            # TODO: evolve and make mutations and combinations:
            elite_index = self.evolver.epoch(self.pop, pop_fitness)

        '''++++++++++++++ RL +++++++++++++'''
        # collect extra experience for RL training:
        self.evaluate(self.rl_agent, is_action_noise=True,
                      store_transition=True)
        # update the RL actor and critic
        rl_train_scores = self.train_rl(self.gen_frames)

        # validate the RL actor separately:
        rl_reward, rl_std, rl_ep_len, rl_ep_std, rl_sm, rl_sm_sd, rl_episode = self.validate_agent(
            self.rl_agent)

        if self.params.pop_size == 0:
            ep_len_avg = rl_ep_len
            ep_len_sd = rl_ep_std

        if self.params.should_log:
            self.rl_history = rl_episode.get_history()

        '''++++++++++++++ Inject RL actor into the POPULATION +++++++++++++'''
        if self.params.pop_size and self.iterations % self.params.rl_to_ea_sync_period == 0:
            # replace any index different from the new elite:
            replace_index = np.argmin(pop_fitness)

            if replace_index == elite_index:
                replace_index = (replace_index + 1) % len(self.pop)

            self.rl_to_evo_loop(self.rl_agent, self.pop[replace_index])
            self.evolver.rl_policy = replace_index
            print('Sync from RL --> Evolution')

        '''++++++++++++++ Collect Statistics +++++++++++++'''
        return {
            'best_train_fitness': best_train_fitness,
            'test_score':         test_score,
            'test_sd':            test_sd,
            'pop_avg':            population_avg,
            'pop_min':            worst_train_fitness,
            'elite_index':        elite_index,
            'avg_smoothness':     sm,
            'smoothness_sd':      sm_sd,
            'rl_reward':          rl_reward,
            'rl_smoothness':      rl_sm,
            'rl_smoothness_std':  rl_sm_sd,
            'rl_std':             rl_std,
            'avg_ep_len':         ep_len_avg,
            'ep_len_sd':          ep_len_sd,
            'PG_obj':             rl_train_scores['PG_obj'],
            'TD_loss':            rl_train_scores['TD_loss'],
            'pop_novelty':        pop_novelty,
        }

    def save_agent(self, parameters: object, elite_index: int) -> None:
        """
        save the trained agent (s)
        Args:
            parameters (object): container class of teh training hyper-parameters
            elite_index (int): Index of the best performing agent. Defaults to None.
        """
        if len(self.pop) > 0:
            pop_dict = {}
            for i, idx in enumerate(self.pop):
                pop_dict[f"actor_{i}"] = idx.actor.state_dict()

            torch.save(pop_dict, os.path.join(
                parameters.save_foldername, 'evolution_agents.pkl'))

            torch.save(self.pop[elite_index].actor.state_dict(), os.path.join(
                parameters.save_foldername, 'elite_agent.pkl'))

            # state history of the champion:
            filename = 'statejistory_episode' + str(self.num_episodes) + '.txt'
            np.savetxt(os.path.join(parameters.save_foldername, filename),
                       self.champion_history, header=str(self.num_episodes))

        # Save RL actor separately:
        torch.save(self.rl_agent.actor.state_dict(), os.path.join(
            parameters.save_foldername, 'rl_agent.pkl'))

        filename = 'rl_statehistory_episode' + str(self.num_episodes) + '.txt'
        np.savetxt(os.path.join(parameters.save_foldername, filename),
                   self.rl_history, header=str(self.num_episodes))

        print('> Saved state history in ' + str(filename) + '\n')
