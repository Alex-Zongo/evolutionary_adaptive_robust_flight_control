from pprint import pprint
import os
import torch

import numpy as np


class ESParameters:
    def __init__(self, conf={}, init=True):
        if not init:
            return

        # setting the device:
        self.device = torch.device("cpu") if hasattr(
            conf, 'disable_cuda') and conf.disable_cuda else torch.device("cuda")
        print('Current device: %s' % self.device)

        self.env_name = conf.env_name if hasattr(
            conf, 'env') else 'PHlab_attitude_nominal'
        self.use_state_history = conf.use_state_history if hasattr(
            conf, 'use_state_history') else False
        self.state_length = conf.state_length if hasattr(
            conf, 'state_length') else 1
        self.should_log = conf.should_log if hasattr(
            conf, 'should_log') else False
        self.run_name = conf.run_name if hasattr(
            conf, 'run_name') else 'default'

        # =========== TD3 Params ===========
        self.use_td3 = conf.use_td3 if hasattr(conf, 'use_td3') else True
        self.use_multiAgent = conf.use_multiAgent if hasattr(
            conf, 'use_multiAgent') else False
        self.use_rnn = conf.use_rnn if hasattr(conf, "use_rnn") else False
        self.use_lstm = conf.use_lstm if hasattr(conf, "use_lstm") else False
        self.policy_noise = conf.policy_noise if hasattr(
            conf, 'policy_noise') else 0.2
        self.noise_clip = conf.noise_clip if hasattr(
            conf, 'noise_clip') else 0.5
        self.policy_update_freq = conf.policy_update_freq if hasattr(
            conf, 'policy_update_freq') else 2

        # =========== SAC Params ===========
        self.log_std_min = -.25
        self.log_std_max = 0.25
        self.epsilon = 1e-6
        self.policy_type = "Gaussian"  # or deterministic
        self.automatic_entropy_tuning = True
        # ============ Gaussian Noise ==========
        self.gauss_sigma = conf.gauss_sigma if hasattr(
            conf, 'gauss_sigma') else 0.1

        # ============= OU Noise ==============
        self.ou_noise = conf.ou_noise if hasattr(conf, 'ou_noise') else False
        self.ou_theta = conf.ou_theta if hasattr(conf, 'ou_theta') else 0.15
        self.ou_sigma = conf.ou_sigma if hasattr(conf, 'ou_sigma') else 0.2
        self.ou_mu = conf.ou_mu if hasattr(conf, 'ou_mu') else 0.0

        # ============ ES Params ===========
        self.pop_size = conf.pop_size if hasattr(conf, 'pop_size') else 10
        self.parents = conf.parents if hasattr(
            conf, 'parents') else self.pop_size//2
        self.elitism = conf.elitism if hasattr(conf, 'elitism') else False
        self.n_grad = conf.n_grad if hasattr(conf, 'n_grad') else 0
        self.n_noisy = conf.n_noisy if hasattr(conf, 'n_noisy') else 0
        self.sigma_init = conf.sigma_init if hasattr(
            conf, 'sigma_init') else 1e-3
        self.sigma_decay = conf.sigma_decay if hasattr(
            conf, 'sigma_decay') else 0.999  # default=0.999
        self.sigma_limit = conf.sigma_limit if hasattr(
            conf, 'sigma_limit') else 0.001
        self.damp = conf.damp if hasattr(conf, 'damp') else 1e-3
        self.damp_limit = conf.damp_limit if hasattr(
            conf, 'damp_limit') else 1e-5
        self.mult_noise = conf.mult_noise if hasattr(
            conf, 'mult_noise') else False

        # CMAES:
        self.weight_decay = conf.weight_decay if hasattr(
            conf, 'weight_decay') else 0.01  # for CMAES
        # self.sigma_init = conf.sigma_init if hasattr(
        #     conf, "sigma_init") else 0.3

        # ============= Training Params =================
        # Number of experiences to use for each training step:
        self.batch_size = conf.batch_size if hasattr(
            conf, 'batch_size') else 100  # 64 for TD3 alone
        self.n_evals = conf.n_evals if hasattr(conf, 'n_evals') else 2
        self.n_generations = conf.n_generations if hasattr(
            conf, 'n_generations') else 100
        # self.max_steps = conf.max_steps if hasattr(
        #     conf, 'max_steps') else 100000  # num of steps to run:
        # frames accumulated before grad updates:
        self.start_steps = conf.start_steps if hasattr(
            conf, 'start_steps') else 10_000
        self.max_iter = conf.max_iter if hasattr(conf, 'max_iter') else 10
        self.sample_ratio = conf.sample_ratio if hasattr(
            conf, 'sample_ratio') else [0.8, 0.1, 0.1]
        self.g_batch_size = int(np.ceil(self.sample_ratio[0]*self.batch_size))
        self.b_batch_size = int(np.ceil(self.sample_ratio[1]*self.batch_size))
        self.n_batch_size = int(np.ceil(self.sample_ratio[2]*self.batch_size))
        # buffer size:
        self.mem_size = conf.mem_size if hasattr(
            conf, 'mem_size') else 1_000_000
        # number of noisy evaluations:

        # ============= misc ====================

        self.seed = conf.seed if hasattr(conf, "seed") else 7
        # model save frequency:
        self.period = conf.period if hasattr(conf, 'period') else 1000

        self.gamma = 0.999
        self.noise_sd = 0.33

        # soft update:
        self.tau = 0.005
        # hidden layer:
        self.actor_num_layers = conf.actor_num_layers if hasattr(
            conf, 'actor_num_layers') else 2
        # 72 for SERL10 or SERL50, 96 for TD3
        self.actor_hidden_size = conf.actor_hidden_size if hasattr(
            conf, 'actor_hidden_size') else 32
        self.actor_lr = conf.actor_lr if hasattr(
            conf, 'actor_lr') else 0.001  # for TD3 alone 0.0000482

        self.critic_num_layers = conf.critic_num_layers if hasattr(
            conf, 'critic_num_layers') else 2
        self.critic_hidden_size = conf.critic_hidden_size if hasattr(
            conf, 'critic_hidden_size') else [32, 64]  # [200, 300]
        self.critic_lr = conf.critic_lr if hasattr(
            conf, 'critic_lr') else 0.001

        self.activation_actor = 'tanh'
        self.activation_critic = 'tanh'
        self.nonlin_activation = 'relu'  # for SERL10 or SERL50, 'relu' for TD3

        # prioritized experience replay:
        self.per = conf.per if hasattr(conf, "per") else False
        if self.per:
            self.replace_old = True
            self.alpha = 0.7
            self.beta_zero = 0.5

        # CAPS: Condition for Action Policy Smoothness:
        self.use_caps = conf.use_caps if hasattr(conf, "use_caps") else False
        # print("Using CAPS: ", self.use_caps)

        # save the results:
        self.state_dim = None
        self.action_dim = None
        self.save_foldername = './logs'
        self.should_log = conf.should_log if hasattr(
            conf, "should_log") else False

        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

    def write_params(self, stdout=False):
        """ Transfer the parameters to a state dictionary
            Args:
                stdout: whether to print the parameters Defaults to True
        """
        if stdout:
            params = pprint.pformat(vars(self), indent=4)
            print(params)
        return self.__dict__

    def update_from_dict(self, new_config_dict: dict):
        """ Update the parameters from a dictionary
            Args:
                new_config_dict: the new configuration dictionary
        """
        self.__dict__.update(new_config_dict)

    def stdout(self) -> None:
        keys = ['save_foldername', 'seed', 'batch_size',
                'actor_lr', 'critic_lr', 'use_state_history', 'state_length',
                'actor_num_layers', 'actor_hidden_size', 'critic_hidden_size', 'activation_actor',
                'activation_critic', 'pop_size',
                'n_evals', 'n_generations',
                'sigma_init', 'sigma_decay', 'start_steps', 'max_iter', 'sample_ratio'
                ]
        _dict = {}
        for k in keys:
            _dict[k] = self.__dict__[k]

        pprint(_dict)
