from copy import deepcopy
import torch
import numpy as np


class CEM:
    '''Cross Entropy Methods'''

    def __init__(self, num_params, params, sigma_init=0.1, mu_init=None, pop_size=10, sigma_decay=0.999, sigma_limit=0.01, damp=1e-3, damp_limit=1e-5, parents=None, elitism=False, antithetic=False, adaptation=False):
        self.params = params
        self.num_params = num_params
        np.random.seed(params.seed)
        self.mu = np.zeros(
            self.num_params) if mu_init is None else np.array(mu_init)
        self.sigma = params.sigma_init if params.sigma_init else sigma_init
        self.sigma_decay = params.sigma_decay if params.sigma_decay else sigma_decay
        self.sigma_limit = params.sigma_limit if params.sigma_limit else sigma_limit
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)
        self.best_cov = deepcopy(self.cov)

        # elite stuff:
        self.elitism = elitism
        # self.elite = np.sqrt(self.sigma) * np.random.randn(self.num_params)
        self.elite = deepcopy(self.mu) if mu_init is not None else np.sqrt(
            self.sigma) * np.random.randn(self.num_params)
        self.elite_score = -np.inf
        self.best_elite_so_far = deepcopy(self.elite)
        self.best_elite_so_far_score = -np.inf
        self.best_mu_so_far = deepcopy(self.mu)
        self.best_mu_so_far_score = -np.inf
        self.mu_score = -np.inf

        self.rl_agent = deepcopy(self.mu)
        self.rl_agent_score = -np.inf
        # sampling stuff:
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"

        # sigma and cov adaptation stuff:
        self.adaptation = adaptation
        if parents is None or parents <= 0:
            self.parents = pop_size//2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents+1)/i)
                                for i in range(1, self.parents+1)])
        self.weights /= self.weights.sum()
        self.first_interaction = True

    def ask(self, pop_size):
        """ Ask for candidate solutions"""
        # np.random.seed(self.params.seed)
        if self.antithetic and not pop_size % 2:
            print('Antithetic sampling')
            epsilon_half = np.random.randn(pop_size//2, self.num_params)
            epsilon = np.concatenate([epsilon_half, -epsilon_half])
        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.first_interaction:
            inds[-1] = self.mu
            self.first_interaction = False
        if self.elitism and not self.first_interaction:

            inds[-1] = self.best_elite_so_far
            if self.rl_agent_score > 1.005 * self.mu_score:
                inds[-2] = self.rl_agent
            # inds[-3] = self.best_elite_so_far
        return inds

    def tell(self, solutions, fitness_table):
        """ Updates the distribution parameters of the candidate solutions and scores table"""
        scores = np.array(fitness_table)
        # times -1 so that maximization becomes minimization problem:

        idx_sorted = np.argsort(scores)[::-1]

        old_mu = deepcopy(self.mu)
        self.damp = self.damp * self.tau + (1-self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        z = (solutions[idx_sorted[:self.parents]] - old_mu)

        # *** sigma adaptation and cov:
        if self.adaptation:
            # ====== version 1
            if self.sigma > self.sigma_limit:
                self.sigma *= self.sigma_decay
            # ====== version 2
            # if scores[idx_sorted[0]] > 1.05 * self.elite_score:
            #     self.sigma *= 0.95
            # else:
            #     self.sigma *= 1.05

            # ===== Covariance
            self.cov = self.weights @ (z * z)
            self.cov = self.sigma * self.cov / \
                np.linalg.norm(self.cov)

        else:
            # ===== Covariance
            self.cov = 1 / self.parents * \
                self.weights @ (z * z) + self.damp * np.ones(self.num_params)

        # ====== Elite
        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        if self.elite_score > self.best_elite_so_far_score:
            self.best_elite_so_far = deepcopy(self.elite)
            self.best_elite_so_far_score = self.elite_score

    def get_distribution_params(self):
        """ Returns the distribution parameters: mu and sigma"""
        return self.mu, self.sigma

    def save_model(self, parameters):
        pass


class CEMA:
    """ Cross-Entropy Methods with sigma adaptation"""

    def __init__(self, num_params, params, sigma_init=1e-3, mu_init=None, pop_size=10, damp=1e-3, damp_limit=1e-5, parents=None, elitism=False, antithetic=False):
        self.params = params
        self.num_params = num_params

        self.mu = np.zeros(
            self.num_params) if mu_init is None else np.array(mu_init)
        self.sigma = params.sigma_init if params.sigma_init else sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff:
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = -np.inf

        # sampling stuff:
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size//2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents+1)/i)
                                for i in range(1, self.parents+1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """ Ask for candidate solutions"""
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size//2, self.num_params)
            epsilon = np.concatenate([epsilon_half, -epsilon_half])
        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite
        return inds

    def tell(self, solutions, fitness_table):
        """ Updates the distribution parameters of the candidate solutions and scores table"""
        scores = np.array(fitness_table)
        # times -1 so that maximization becomes minimization problem:
        scores *= -1
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1-self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        # sigma adaptation:
        if scores[idx_sorted[0]] > 0.95 * self.elite_score:
            self.sigma *= 0.95
        else:
            self.sigma *= 1.05

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = self.weights @ (z * z)
        self.cov = self.sigma * self.cov / np.linalg.norm(self.cov)

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]

    def get_distribution_params(self):
        """ Returns the distribution parameters: mu and sigma"""
        return self.mu, self.sigma
