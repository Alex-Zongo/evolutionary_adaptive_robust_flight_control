import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from parameters import Parameters
from core_algorithms.model_utils import activations, is_lnorm_key, LayerNorm
from core_algorithms.replay_memory import ReplayMemory, PrioritizedReplayMemory


class Actor(nn.Module):
    def __init__(self, args: Parameters, init=False):
        super(Actor, self).__init__()
        self.args = args
        h = self.args.actor_hidden_size
        self.L = self.args.actor_num_layers
        self.activation = activations[args.nonlin_activation.lower()]
        # activ_layer = F.tanh() if activation == nn.Tanh else F.elu()

        # input Layer:
        self.input_layer = nn.Linear(args.state_dim, h)
        # hidden layers:
        self.hid_layer = nn.Linear(h, h)
        self.lnorm = LayerNorm(h)
        # output layer:
        self.output_layer = nn.Linear(h, args.action_dim)

        # layers.extend([
        #     nn.Linear(args.state_dim, h),
        #     activation
        # ])

        # # hidden layers:
        # for _ in range(L):
        #     layers.extend([
        #         nn.Linear(h, h),
        #         LayerNorm(h),
        #         activation
        #     ])

        # # output layer:
        # layers.extend([
        #     nn.Linear(h, args.action_dim),
        #     nn.Tanh()
        # ])
        # # print(*layers)
        # self.net = nn.Sequential(*layers)
        self.to(args.device)

    def forward(self, state: torch.tensor):
        # return self.net(state)
        x = self.activation(self.input_layer(state))  # input setup:
        for _ in range(self.L):
            # hidden layers:
            x = self.activation(self.lnorm(self.hid_layer(x)))
        return F.tanh(self.output_layer(x))  # output layer:

    def select_action(self, state: torch.tensor):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state).cpu().data.numpy().flatten()

    def get_novelty(self, batch):
        """ How different is the new action compared to the last one """
        state_batch, action_batch, _, _, _ = batch
        novelty = torch.mean(
            torch.sum((action_batch - self.forward(state_batch))**2, dim=1))
        self.novelty = novelty.item()
        return self.novelty

    def extract_grad(self):
        """ Current pytorch gradient in same order as genome's flattened parameter vector """
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count+sz] = param.grad.view(-1)
            count += sz
        return pvec.detach().clone()

    def extract_parameters(self):
        """ Extract the current flattened neural network weights """
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count+sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    def inject_parameters(self, pvec):
        """ Inject a flat vector of ANN parameters into the model's current neural network weights """
        count = 0
        for name, param in self.named_parameters():
            # only alter W -- skip norms and biases:
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count+sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz

    def count_parameters(self):
        """ Number of parameters in the model """
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            count += param.numel()
        return count


class GeneticAgent:
    def __init__(self, args: Parameters):
        """ Genetic Agent initialization:
        Args:
            args (Parameters): essential parameters for the agent initialization
        """
        self.args = args
        self.actor = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        self.buffer = ReplayMemory(
            self.args.individual_bs, device=args.device)
        self.critical_buffer = ReplayMemory(
            self.args.individual_bs, device=args.device)
        # self.loss = nn.MSELoss()

    def update_parameters(self, batch, parent1, parent2, critic) -> float:
        """ Crossover parameters update:

        Args:
            batch: batch of transitions
            parent1: first parent actor
            parent2: second parent actor
            critic: critic network for filtering

        Returns:
            loss (float): policy cloning loss
        """
        state_batch, _, _, _, _ = batch  # batch: state, action, next state, reward, done

        # parents actions:
        p1_action = parent1(state_batch)
        p2_action = parent2(state_batch)

        # parents Q's (depending on the DRL algo -> double Q's [] for training stability)
        p1_q1, p1_q2 = critic(state_batch, p1_action)
        p1_q = torch.min(p1_q1, p1_q2).flatten()
        p2_q1, p2_q2 = critic(state_batch, p2_action)
        p2_q = torch.min(p2_q1, p2_q2).flatten()

        # selecting best behaving parent based on Q-filtering
        # threshold on how much better an action is wrt to the other:
        eps = 1e-6
        action_batch = torch.cat(
            (p1_action[p1_q - p2_q > eps], p2_action[p2_q - p1_q >= eps])).detach()
        state_batch = torch.cat(
            (state_batch[p1_q - p2_q > eps], state_batch[p2_q - p1_q >= eps])).detach()
        actor_action = self.actor(state_batch)

        # actor update:
        self.actor_optim.zero_grad()
        sq_loss = (actor_action - action_batch)**2
        policy_loss = torch.sum(sq_loss) + torch.mean(actor_action**2)
        policy_mse = torch.mean(sq_loss)
        policy_loss.backward()
        self.actor_optim.step()

        return policy_mse.item()

    def load_from_dict(self, actor_dict: dict):
        self.actor = actor_dict
