import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
from torch.distributions import Categorical
import main 

class PPO(nn.Module):
    def __init__(self, n_resources):
        super().__init__()
        self.n = n_resources
        self.base = nn.Sequential(
            nn.Linear(4*self.n+2,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
        )
        self.produce_head = nn.Linear(64,3*self.n)
        self.orders_head = nn.Linear(64,6*self.n)
        self.critic = nn.Linear(64,1)
    
    def forward(self,x):
        x = self.base(x)
        produces = self.produce_head(x)
        orders = self.orders_head(x)
        value_func = self.critic(x)
        return produces,orders,value_func



