from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random
import os
from evaluate import evaluate_HIV_population, evaluate_HIV
env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

# My code is mainly based on the DQN algorithm that have been implemented in the course notebook
# I have made some modifications to adapt it to the HIVPatient environment with hyperparameters tuning

# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.95,
          'buffer_size': 100000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 20000,
          'epsilon_delay_decay': 100,
          'batch_size': 200,
          'gradient_steps': 5,
          'update_target_freq': 900,
          'criterion': torch.nn.SmoothL1Loss() }

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

    
class ProjectAgent:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n 
        nb_neurons=256

        self.model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons//2),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons//2, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)).to(self.device)
        
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        
        self.best_eval_model = deepcopy(self.model).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20


    def act(self, observation, use_random=False):
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    
    def save(self, path):
        torch.save(self.best_eval_model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(__file__)), path))

    def load(self,path="agent.pkl"):
        # get the folder of the folder 
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), path), map_location=self.device))
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        prev_score_pop = 0
        prev_score = 0
        while episode < max_episode:

            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()

            # update target network if needed
            if step % self.update_target_freq == 0: 
                self.target_model.load_state_dict(self.model.state_dict())
            
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                episode_return.append(episode_cum_reward)
                score_agent_pop =  evaluate_HIV_population(agent=self, nb_episode=1)
                score_agent = evaluate_HIV(agent=self, nb_episode=1)

                print("Episode ", '{:2d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:4d}'.format(len(self.memory)), 
                        ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                        ", score agent pop ", '{:.1e}'.format(score_agent_pop),
                        ", score agent ", '{:.1e}'.format(score_agent),
                        sep='')
                state, _ = env.reset()
                episode_cum_reward = 0
                if score_agent_pop > prev_score_pop: 
                    if (prev_score_pop > 2e10 and score_agent > prev_score):
                        prev_score = score_agent
                        prev_score_pop = score_agent_pop
                        self.best_eval_model.load_state_dict(self.model.state_dict())
                    elif prev_score_pop<2e10:
                        prev_score = score_agent
                        prev_score_pop = score_agent_pop
                        self.best_eval_model.load_state_dict(self.model.state_dict())  
                if episode % 20 ==0:
                    self.save("agent.pkl")
            else:
                state = next_state
        return episode_return


if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train(env, 200)
    agent.save("agent.pkl")