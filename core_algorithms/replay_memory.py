import random
import torch
import numpy as np
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done')
    )


class ReplayMemory:
    """
    Replay memory
    """

    def __init__(self, capacity: int, batch_size: int, seed: int = 0, device: str = "cpu"):
        """
        Args:
            capacity: size of the replay memory
            batch_size: size of the batch to sample from the memory
            seed: random seed
        """
        self.device = device
        self.capacity = capacity
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Push a transition into the memory.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        reshaped_args = []
        for arg in args:
            reshaped_args.append(np.reshape(arg, (1, -1)))
            
        self.memory[self.position] = Transition(*reshaped_args)
        # update the position
        self.position = (self.position + 1) % self.capacity
    
    def push_content_of(self, other: ReplayMemory):
        """
        Push the content of another ReplayMemory into this one
        :param other: the other ReplayMemory
        """
        for transition in other.get_latest(self.capacity):
            self.push(*transition)
        
    def get_latest(self, latest_num: int):
        """
        The latest elements from the buffer with the most recent at the end of the returned list
        :param latest_num: the number of latest elements to return
        :return: a list of the latest elements
        """
        if self.capacity < latest_num:
            latest_transitions = self.memory[self.position:].copy() + self.memory[:self.position].copy()
        elif len(self.memory) < self.capacity:
            latest_transitions = self.memory[-latest_num:].copy()
        elif self.position >= latest_num:
            latest_transitions = self.memory[:self.position][-latest_num:].copy()
        else:
            latest_transitions = self.memory[-latest_num + self.position:].copy() + self.memory[:self.position].copy()
        return latest_transitions
    
    def add_latest_from(self, other: ReplayMemory, latest_num:int):
        """
        Add the latest elements from another ReplayMemory to this one
        :param other: the other ReplayMemory
        :param latest_num: the number of latest elements to add
        """
        latest_transitions = other.get_latest(latest_num)
        for transition in latest_transitions:
            self.push(*transition)
        

    def sample(self):
        """
        Sample a batch of transitions from the memory.
        """
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state = torch.FloatTensor(np.concatenate(batch.state)).to(self.device)
        action = torch.FloatTensor(np.concatenate(batch.action)).to(self.device)
        next_state = torch.FloatTensor(np.concatenate(batch.next_state)).to(self.device)
        reward = torch.FloatTensor(np.concatenate(batch.reward)).to(self.device)
        done = torch.FloatTensor(np.concatenate(batch.done)).to(self.device)
        
        return state, action, next_state, reward, done
    
    def sample_from_latest(self, latest_num: int):
        """
        Sample a batch of transitions from the latest elements in the memory.
        """
        latest_trans = self.get_latest(latest_num)
        transitions = random.sample(latest_trans, self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state = torch.FloatTensor(np.concatenate(batch.state)).to(self.device)
        action = torch.FloatTensor(np.concatenate(batch.action)).to(self.device)
        next_state = torch.FloatTensor(np.concatenate(batch.next_state)).to(self.device)
        reward = torch.FloatTensor(np.concatenate(batch.reward)).to(self.device)
        done = torch.FloatTensor(np.concatenate(batch.done)).to(self.device)
        
        return state, action, next_state, reward, done

    def __len__(self):
        """
        Return the current size of the memory.
        """
        return len(self.memory)
    
    def shuffle(self):
        """
        Shuffle the memory.
        """
        random.shuffle(self.memory)
    
    def reset(self):
        """
        Reset the memory.
        """
        self.memory = []
        self.position = 0
        
        
class PrioritizedReplayMemory:
    def __init__(self, capacity, device, alpha=0.6, beta_start=0.6, beta_frames=100000):
        self.device = device
        self.prob_alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)