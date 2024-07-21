import pandas as pd
import numpy as np
import gym
import eplus_env
import tqdm as tqdm

import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
# from network import MLP

from stable_baselines3 import DQN, PPO

# import faulthandler
# faulthandler.enable()


# ppo.py
class EPPPO:
    def __init__(self, env):
        self.env = env
        self.obs_dim = in_dim
        self.act_dim = out_dim
        
        # Initializing Actor and Critic networks
        self.actor = MLP(self.obs_dim, self.act_dim)
        self.critic = MLP(self.obs_dim, 1)

        self.initiate_hyperparameters()

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def get_action(self, state):
        """
        Get the mean action from the actor network. 
        Create a multivariate normal distribution 
        Sample an action from the distribution along with it's logprob
        Return sampled action and log prob
        """
        mean = self.actor(state)
         
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)    

        return action.detach().numpy(), log_prob.detach()

    def learn(self, max_timesteps):
        """
        Imitiating Stable Baselines approach, with max_timesteps instead of epochs
        """
        timestep_count = 0
        while timestep_count < max_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

    def init_hyperparams(self):
        self.max_timesteps_per_batch = 1056*10
        self.max_timesteps_per_episode = 1056
        self.gamma = 0.95
    
    def get_random_action(self):
        """
        Environment specific action sampler for energyplus. 
        """
        possible_actions = [5, 6, 7, 8, 9, 10]
        possible_actions2 = [5, 6, 7, 8, 9, 10]

        all_possible_actions = [[i, j] for i in possible_actions for j in possible_actions2]
        return random.choice(all_possible_actions)
    
    def get_reward(self, state):
        """
        Simple reward calculator for the energyplus environment. 
        """
        reward = (state[-16]* 2.77778e-7)+ (state[-15]* 2.77778e-7)+ (state[-17]* 2.77778e-7)+ (state[-14]* 2.77778e-7)+ (state[-13]* 2.77778e-7)+ (state[-12]* 2.77778e-7)+ (state[-11]* 2.77778e-7)+ (state[-10]* 2.77778e-7)+ (state[-9]* 2.77778e-7)
        # reward = (reward * -1)
        return reward
    
    def compute_rtgs(self, batch_rews):
        """
        Returns the rewards to go per episode per batch
        """
        batch_rtgs = []

        for ep_rewards in reversed(batch_rews):
            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
        return batch_rtgs


    def rollout(self):
        """
        One iteration of policy improvement. 
        """
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        
        batch_timesteps = 0
        while batch_timesteps < self.max_timesteps_per_batch:
            ep_rewards = []
            env = gym.make('doe_large_office_chennai_data_gen-v0')
            info, state, done = env.reset()
            for timestep in range(self.max_timesteps_per_episode):
                batch_timesteps += 1
                batch_obs.append(state)
                action, log_prob = self.get_action(state)
                info, state, done = self.env.step(action)

                ep_rewards.append(get_reward(state))
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break
            batch_lens.append(timestep+1)
            batch_rews.append(ep_rewards)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float32)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_actions, batch_log_probs, batch_rtgs, batch_lens