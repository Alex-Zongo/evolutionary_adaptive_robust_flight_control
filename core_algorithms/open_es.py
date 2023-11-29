from copy import deepcopy
import numpy as np

from parameters_es import ESParameters


class Optimizer(object):

    def __init__(self, pi, epsilon=1e-08):
        self.epsilon = epsilon
        self.pi = pi
        self.dim = pi.num_params
        self.t = 0

    def update(self, grad):
        self.t += 1
        step = self._step(grad)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio

    def _step(self, grad):
        raise NotImplementedError


class BasicSGD(Optimizer):
    """
    Standard gradient descent
    """

    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _step(self, grad):
        step = -self.stepsize * grad
        return step


class SGD(Optimizer):
    """
    Gradient descent with momentum
    """

    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.stepsize, self.momentum = stepsize, momentum
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _step(self, grad):

        self.v = self.momentum * self.v + (1. - self.momentum) * grad
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    """
    Adam optimizer
    """

    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _step(self, grad):

        a = self.stepsize * np.sqrt(1 - self.beta2 **
                                    self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)

        return step


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))]
    which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class OpenES:

    """
    Basic Version of OpenAI Evolution Strategies
    """

    def __init__(self, num_params,
                 params: ESParameters,
                 mu_init=None,
                 sigma_init=0.1,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 lr=1e-2,
                 lr_decay=0.999,
                 lr_limit=0.001,
                 pop_size=256,
                 antithetic=False,
                 weight_decay=0.01,
                 rank_fitness=False,
                 elitism=False
                 ):

        # misc
        self.params = params
        self.num_params = num_params
        np.random.seed(params.seed)

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.best_mu = deepcopy(self.mu)
        self.sigma = params.sigma_init if params.sigma_init else sigma_init
        self.sigma_decay = params.sigma_decay if params.sigma_decay else sigma_decay
        self.sigma_limit = params.sigma_limit if params.sigma_limit else sigma_limit

        # optimization stuff
        self.learning_rate = lr
        self.learning_rate_decay = lr_decay
        self.learning_rate_limit = lr_limit
        self.optimizer = Adam(self, self.learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        self.first_interaction = True

        # elitism
        self.elitism = elitism
        if self.rank_fitness:
            self.elitism = False  # forget the elite or best one if we rank

        self.elite = np.sqrt(self.sigma)*np.random.randn(self.num_params)
        self.elite_score = -np.inf
        self.best_elite_so_far = deepcopy(self.elite)
        self.best_elite_so_far_score = -np.inf
        self.best_mu_so_far = deepcopy(self.mu)
        self.best_mu_so_far_score = -np.inf

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma*sigma))

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        np.random.seed(self.params.seed)
        if self.antithetic:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            self.epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            self.epsilon = np.random.randn(pop_size, self.num_params)
        inds = self.mu + self.epsilon * self.sigma

        if self.elitism:
            inds[-1] = self.best_elite_so_far
        return inds

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        assert (len(scores) ==
                self.pop_size), "Inconsistent reward_table size reported."

        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        idx_sorted = np.argsort(reward)[::-1]
        # elitism
        self.current_elite = solutions[idx_sorted[0]]
        self.current_elite_score = scores[idx_sorted[0]]

        if self.first_interaction:
            self.first_interaction = False
            self.elite_score = self.current_elite_score
            self.elite = self.current_elite
            self.best_elite_so_far = deepcopy(self.elite)
            self.best_elite_so_far_score = self.elite_score
        else:
            if not self.elitism or self.current_elite_score > self.elite_score:
                self.elite_score = self.current_elite_score
                self.elite = self.current_elite

        if self.current_elite_score > self.best_elite_so_far_score:
            self.best_elite_so_far_score = self.current_elite_score
            self.best_elite_so_far = self.current_elite
        # standardized rewards to have gaussian distribution:
        normalized_reward = (reward - np.mean(reward))/np.std(reward)
        epsilon = (solutions - self.mu) / self.sigma
        grad = -1./(self.sigma * self.pop_size) * \
            np.dot(epsilon.T, normalized_reward)

        # optimization step
        self.optimizer.stepsize = self.learning_rate
        update_ratio = self.optimizer.update(grad)

        # adjusting sigma:
        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay
        if self.learning_rate > self.learning_rate_limit:
            self.learning_rate *= self.learning_rate_decay

    def get_distribution_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)


class GES:

    """
    Guided Evolution Strategies
    """

    def __init__(self, num_params,
                 params: ESParameters,
                 mu_init=None,
                 sigma_init=0.1,
                 lr=1e-2,
                 alpha=0.5,
                 beta=2,
                 k=1,
                 pop_size=256,
                 antithetic=True,
                 weight_decay=0.01,
                 rank_fitness=False):

        # misc
        self.num_params = num_params
        self.first_interaction = True

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.U = np.ones((self.num_params, k))

        # optimization stuff
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.learning_rate = lr
        self.optimizer = Adam(self.learning_rate)

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    def ask(self):
        """
        Returns a list of candidates parameters
        """
        np.random.seed(self.params.seed)
        if self.antithetic:
            epsilon_half = np.sqrt(self.alpha / self.num_params) * \
                np.random.randn(self.pop_size // 2, self.num_params)
            epsilon_half += np.sqrt((1 - self.alpha) / self.k) * \
                np.random.randn(self.pop_size // 2, self.k) @ self.U.T
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.sqrt(self.alpha / self.num_params) * \
                np.random.randn(self.pop_size, self.num_params)
            epsilon += np.sqrt(1 - self.alpha) * \
                np.random.randn(self.pop_size, self.num_params) @ self.U.T

        return self.mu + epsilon * self.sigma

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        assert (len(scores) ==
                self.pop_size), "Inconsistent reward_table size reported."

        reward = np.array(scores)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            reward += l2_decay

        epsilon = (solutions - self.mu) / self.sigma
        grad = -self.beta/(self.sigma * self.pop_size) * \
            np.dot(reward, epsilon)

        # optimization step
        step = self.optimizer.step(grad)
        self.mu += step

    def add(self, params, grads, fitness):
        """
        Adds new "gradient" to U
        """
        if params is not None:
            self.mu = params
        grads = grads / np.linalg.norm(grads)
        self.U[:, -1] = grads

    def get_distribution_params(self):
        """
        Returns the parameters of the distribution:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.sigma ** 2)


class PEPG:
    ''' Extension of PEPG '''
    pass


class SERL:
    ''' Extension of Safety Informed Evolution Reinforcement Learning '''

    def __init__(self, args: ESParameters, num_params,
                 sigma_init=0.1,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 pop_size=20,
                 elite_fraction=0.3,
                 weight_decay=0.01,
                 elitism=True,
                 mutation_params=dict(),
                 cross_over_params=dict(),):

        self.params = args
        self.num_params = num_params
        # sigma
        self.sigma = args.sigma_init if args.sigma_init else sigma_init
        self.sigma_decay = args.sigma_decay if args.sigma_decay else sigma_decay
        self.sigma_limit = args.sigma_limit if args.sigma_limit else sigma_limit
        # population size
        self.pop_size = args.pop_size if args.pop_size else pop_size

        self.elite_ratio = elite_fraction
        self.elite_pop_size = int(self.pop_size * self.elite_ratio)

        self.weight_decay = weight_decay

        def safety_informed_mutation(self, a):
            pass

        def cross_over(self, a, b):
            pass

        def ask(self, pop):
            # mate the elite: (cross_over)
            # mute the remaining population or the mated elite: (mutation: Safety-Informed)
            # return the new population
            pass
