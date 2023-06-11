import torch
import torch.nn as nn
from torch.nn import functional as F
from replay_memory import ReplayMemory, PrioritizedReplayMemory
from parameters import Parameters
from model_utils import LayerNorm, hard_update, soft_update
from genetic_agent import Actor
from torch.optim import Adam


class Critic(nn.Module):
    def __init__(self, args: Parameters):
        super(Critic, self).__init__()
        self.args = args
        l1 = 32
        l2 = 64
        l3 = int(l2/2)

        # Input layer:
        # batch norm:
        self.bnorm = nn.BatchNorm1d(args.state_dim + args.action_dim)
        self.input_layer = nn.Linear(args.state_dim + args.action_dim, l1)

        # hidden layers:
        self.hidden1 = nn.Linear(l1, l2)
        self.lnorm1 = LayerNorm(l2)

        self.hidden2 = nn.Linear(l2, l3)
        self.lnorm2 = LayerNorm(l3)

        # output layer:
        self.output_layer = nn.Linear(l3, 1)
        self.output_layer.weight.data.mul_(0.1)
        self.output_layer.bias.data.mul_(0.1)

        self.to(args.device)

    def forward(self, state, action):

        # input interface:
        x = torch.cat((state, action), 1)
        x = self.bnorm(x)
        x = F.elu(self.input_layer(x))

        # hidden layers:
        hid = F.elu(self.lnorm1(self.hidden1(x)))  # hidden layer 1
        hid = F.elu(self.lnorm2(self.hidden(hid)))  # hidden layer 2

        # output layer:
        out = self.output_layer(hid)

        return out


class DDPG(object):
    def __init__(self, args: Parameters):
        self.args = args
        self.buffer = ReplayMemory(args.individual_bs, device=args.device)
        self.critical_buffer = ReplayMemory(
            args.individual_bs, device=args.device)

        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.gamma = args.gamma
        self.tau = args.tau
        self.loss = nn.MSELoss()

        # initialy make sure both network targets are of the same weights:
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def td_error(self, state, action, next_state, reward, done):
        """ Calculates the TD {Temporal Difference} error"""
        next_action = self.actor_target(next_state)
        next_q = self.critic_target(next_state, next_action)

        done = 1 if done else 0
        target_q = reward + (self.gamma * next_q * (1 - done))
        current_q = self.critic(state, action)
        TD = (target_q - current_q).abs()
        return TD.item()

    def update_parameters(self, batch, iteration: int):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        # load everything to device GPU if not already done:;
        self.actor_target.to(self.args.device)
        self.critic_target.to(self.args.device)
        self.critic.to(self.args.device)
        self.actor.to(self.args.device)
        state_batch = state_batch.to(self.args.device)
        next_state_batch = next_state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_batch = reward_batch.to(self.args.device)
        if self.args.use_done_mask: done_batch = done_batch.to(self.args.device)
