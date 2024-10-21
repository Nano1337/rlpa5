import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Output action probabilities
        return F.softmax(x, dim=1)

class REINFORCEAgent:
    def __init__(self, env, hidden_size=128, learning_rate=1e-2, gamma=0.99):
        self.env = env
        self.gamma = gamma  # Discount factor
        # Initialize policy π_θ(a|s) with random weights θ
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, hidden_size)
        # Initialize optimizer for policy network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.log_probs = []  # Store log probabilities of actions taken
        self.rewards = []  # Store rewards received

    def select_action(self, state):
        # Handle different state representations
        if isinstance(state, tuple):
            state = state[0]  # Extract the numpy array from the tuple
        
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.float().unsqueeze(0)
        else:
            raise ValueError(f"Unexpected state type: {type(state)}")
        
        # Generate an action using the current policy
        probs = self.policy(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        # Store log probability of the action taken
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        # Implement REINFORCE algorithm
        
        # Step 1: Compute returns (discounted future rewards)
        R = 0
        returns = []
        for r in self.rewards[::-1]:  # Iterate rewards in reverse order
            R = r + self.gamma * R    # Compute discounted return
            returns.insert(0, R)      # Insert at beginning to maintain correct order
        
        # Step 2: Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize for stability
        
        # Step 3: Compute policy loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)  # Negative sign for gradient ascent
        
        # Step 4: Perform gradient descent
        self.optimizer.zero_grad()  # Clear previous gradients
        policy_loss = torch.cat(policy_loss).sum()  # Sum up all the losses
        policy_loss.backward()  # Compute gradients
        self.optimizer.step()  # Update policy parameters
        
        # Step 5: Clear episode data
        self.log_probs = []
        self.rewards = []

        return policy_loss

    def save_weights(self, filepath):
        torch.save(self.policy.state_dict(), filepath)

    def load_weights(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))
        self.policy.eval()