import torch
import torch.nn as nn
from torch.nn import functional as F
from core_algorithms.replay_memory import ReplayMemory, PrioritizedReplayMemory
from parameters import Parameters
from core_algorithms.model_utils import LayerNorm, hard_update, soft_update
from core_algorithms.genetic_agent import Actor
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
        hid = F.elu(self.lnorm2(self.hidden2(hid)))  # hidden layer 2

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

    def update_parameters(self, batch, iteration: int, use_champion_target: bool = False):
        """ Update the parameters of the actor and critic networks
        Args:
            batch: a batch of transitions from the replay memory
        Returns:
            actor_policy_loss: the loss of the actor network
            TD error: the TD error of the critic network
        """
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
        if self.args.use_done_mask:
            done_batch = done_batch.to(self.args.device)

        # critic update:
        next_action_batch = self.actor_target.forward(next_state_batch)
        next_q = self.critic_target.forward(
            next_state_batch, next_action_batch)
        if self.args.use_done_mask:
            next_q = next_q * (1 - done_batch)
        target_q = reward_batch + (self.gamma * next_q * (1 - done_batch))

        self.critic_optim.zero_grad()
        current_q = self.critic.forward(state_batch, action_batch)
        delta = (current_q - target_q).abs()
        loss = torch.mean(delta**2)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optim.step()

        # actor update:
        self.actor_optim.zero_grad()
        policy_grad_loss = - \
            self.critic.forward(
                state_batch, self.actor.forward(state_batch)).mean()
        policy_loss = policy_grad_loss

        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        return policy_grad_loss.data.cpu().numpy(), delta.data.cpu().numpy()
