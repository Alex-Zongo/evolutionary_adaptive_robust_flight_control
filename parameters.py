from pprint import pprint
import os
import torch


class Parameters:
    def __init__(self, conf={}, init=True):
        if not init:
            return

        # setting the device:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Current device: %s' % self.device)

        self.env_name = conf.env if hasattr(
            conf, 'env') else 'Citation Aircraft'
        self.save_periodic = True if hasattr(conf, 'save_periodic') else False

        # num of frames to run:
        if hasattr(conf, 'num_frames'):
            self.num_frames = conf.num_frames
        else:
            self.num_frames = 800_000

        # synchronization
        if hasattr(conf, 'sync_period'):
            self.rl_to_ea_sync_period = conf.sync_period
        else:
            self.rl_to_ea_sync_period = 1

        # model save frequency:
        self.next_save = conf.next_save if hasattr(conf, 'next_save') else 1000

        # ------- RL Params TD3 or DDPG or SAC and EA others -------------
        self.test_ea = conf.test_ea if hasattr(conf, "test_ea") else False
        if self.test_ea:
            self.frac_frames_train = 0
        else:
            # fraction of frames to train on or default training:
            self.frac_frames_train = 1

        # Number of experiences to use for each training step:
        self.batch_size = 86
        self.buffer_size = 100_000

        self.lr = 0.0004335
        self.gamma = 0.98
        self.noise_sd = 0.2962183114680794

        self.use_done_mask = True
        self.use_ounoise = conf.use_ounoise if hasattr(
            conf, "use_ounoise") else False

        self.tau = 0.005
        self.seed = conf.seed if hasattr(conf, "seed") else 7

        # hidden layer:
        self.num_layers = 3
        self.hidden_size = 72
        self.activation_actor = 'tanh'
        self.activation_critic = 'elu'

        self.learn_start = 10_000  # frames accumulated before grad updates:
        # prioritized experience replay:
        self.per = conf.per if hasattr(conf, "per") else False
        if self.per:
            self.replace_old = True
            self.alpha = 0.7
            self.beta_zero = 0.5

        # CAPS: Condition for Action Policy Smoothness:
        self.use_caps = conf.use_caps if hasattr(conf, "use_caps") else True

        # ------------- TD3 Params: -----------
        self.policy_update_freq = 3      # minimum for TD3 is 2
        self.noise_clip = 0.5                # default for TD3

        # ------------- Neuro Evolution (EA) Params ---------
        # number of actors in the population:
        self.pop_size = conf.pop_size if hasattr(conf, "pop_size") else 10

        # champion is target actor:
        self.use_champion_target = conf.champion_target if hasattr(
            conf, "champion_target") else False

        # genetic memory size or individual buffer size:
        self.individual_bs = 10_000

        if self.pop_size:
            self.smooth_fitness = conf.smooth_fitness if hasattr(
                conf, "smooth_fitness") else False  # or False,#TODO: study the effects

            # increase the buffer size:
            self.buffer_size = 800_000

            # decrease the learning rate:
            self.lr = 0.00018643512599969097

            # num of trials during evaluation step:
            self.num_evals = 3

            # elitism rate - % of elite within the population
            self.elite_fraction = 0.2

            # Mutation and crossover:
            self.mutation_prob = 0.9
            self.mutation_mag = 0.0247682869654

            self.mutation_batch_size = self.batch_size
            # might be provided by the user:
            self.mut_type = conf.mut_type if hasattr(
                conf, "mut_type") else 'proximal'
            # whether to use a distillation crossover:
            self.distil_crossover = True
            self.distil_type = conf.distil_type if hasattr(
                conf, "distil_type") else 'distance'
            self.crossover_prob = 0.0
            self.verbose_crossover = conf.verbose_crossover if hasattr(
                conf, "verbose_crossover") else False
            self.verbose_mut = conf.verbose_mut if hasattr(
                conf, "verbose_mut") else False

        # save the results:
        self.state_dim = None
        self.action_dim = None
        self.save_foldername = './tmp/'
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
                'buffer_size', 'lr', 'gamma', 'noise_sd',
                'num_layers', 'hidden_size', 'activation_actor',
                'activation_critic', 'use_caps', 'pop_size',
                'use_champion_target', 'smooth_fitness', 'num_evals',
                'elite_fraction']
        _dict = {}
        for k in keys:
            _dict[k] = self.__dict__[k]

        pprint(_dict)
