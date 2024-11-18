import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Actor Network: Implements the deterministic policy
# Maps states to specific actions rather than probability distributions
class Actor(nn.Module):
    def __init__(self, input_state, output_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(input_state, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, output_action)
        
    def forward(self, state):
        # Continuous action selection using tanh activation to bound actions to [-1, 1]
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x

# Critic Network: Implements the action-value function Q
# Estimates Q-value for state-action pairs to evaluate the actor's policy
class Critic(nn.Module):
    def __init__(self, input_state, output_action):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(input_state + output_action, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        # Concatenates state and action to estimate Q-value
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Experience Replay Buffer: Stores transitions for off-policy learning
# Breaks correlations in sequential data and enables mini-batch learning
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        
    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))
        self.buffer_counter += 1
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(next_states), 
                np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32))

# DDPG Agent: Implements the core DDPG algorithm
class DDPGAgent:
    def __init__(self, state_dim, action_dim, 
                 buffer_size=100000,
                 batch_size=64,
                 gamma=0.99,        # Discount factor for future rewards
                 tau=0.005,         # Soft update coefficient for target networks
                 actor_lr=1e-4,     # Learning rate for actor
                 critic_lr=1e-3,    # Learning rate for critic
                 noise_std=0.1):    # Standard deviation for exploration noise
        
        self.device = torch.device("cpu")

        # Initialize networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers with custom epsilon
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-7)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-7)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Set hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_noise_std = noise_std
        self.noise_std = noise_std
        self.min_buffer_size = 10000  # Increased from 1000
        self.start_steps = 10000  # Random actions for first 10000 steps
        self.max_steps = 1000000  # Maximum training steps
        
        # Initialize learning rates
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.total_steps = 0
        
    def select_action(self, state, noise=None):
        # Action Selection Process:
        # 1. Initially uses random actions for better exploration (start_steps)
        # 2. Then switches to policy-based actions with exploration noise
        # 3. During testing, uses pure policy actions without noise
        # If noise is not None, we're in test mode
        test_mode = noise is not None
        
        # Take random actions for better exploration in the beginning (only during training)
        if not test_mode and self.total_steps < self.start_steps:
            return np.random.uniform(-1, 1, size=self.actor.layer3.out_features)
            
        if isinstance(state, tuple):
            state = state[0]
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
            # Add Gaussian noise for exploration during training only
            if not test_mode:
                noise = np.random.normal(0, self.noise_std, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            
        return action
    
    def train(self):
        # DDPG Training Algorithm Implementation:
        
        # Step 1: Sample random minibatch of N transitions from replay buffer
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)
        
        # Step 2: Compute target Q-value:
        # y = r + gamma * Q(next_state, next_action)
        # Where Q is target critic and next_action is from target actor
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
        
        # Step 3: Update Critic
        # Minimize loss between target and current Q values
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        # Step 4: Update Actor using policy gradient
        # Maximize Q values for actor's actions
        # Equivalent to minimizing negative Q values
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Step 5: Soft update target networks
        # Blend target networks towards current networks
        self._update_target_networks()
        
        # Additional: Decay learning rates and exploration noise over time
        self._decay_parameters()

    def _update_target_networks(self):
        # Performs soft update of target network parameters
        # Slowly blends target networks towards current networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _decay_parameters(self):
        # Implements decay of learning rates and exploration noise
        # Helps transition from exploration to exploitation
        # Decay learning rates and noise based on total steps
        if self.total_steps < self.max_steps:
            frac = 1.0 - (self.total_steps / self.max_steps)
            
            # Update learning rates
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.actor_lr * frac
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.critic_lr * frac
                
            # Update noise standard deviation
            self.noise_std = self.initial_noise_std * frac
    
    def save(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
    def load_weights(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())