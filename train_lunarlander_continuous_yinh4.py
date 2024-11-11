import gym
import numpy as np
from ddpg_yinh4 import DDPGAgent
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Initialize environment
env = gym.make('LunarLanderContinuous-v2')
env = gym.wrappers.RecordEpisodeStatistics(env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize agent
agent = DDPGAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    buffer_size=50000,
    batch_size=64,
    gamma=0.99,
    tau=0.005,
    actor_lr=2e-4,
    critic_lr=3e-4,
    noise_std=0.1,
)

# Training parameters
max_episodes = 2000
max_steps = 1000000
min_buffer_size = 1000
start_steps = 10000
episode_rewards = []
avg_rewards = []
critic_losses = []
actor_losses = []

# Start training timer
start_time = time.time()

# Training loop
for episode in range(max_episodes):
    state = env.reset()
    if isinstance(state, tuple):  # Handle new gym return type
        state = state[0]
    episode_reward = 0
    episode_critic_loss = []
    episode_actor_loss = []
    
    for step in range(max_steps):
        # Select action
        action = agent.select_action(state)
        
        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store experience in replay buffer
        agent.replay_buffer.add(state, action, next_state, reward, done)
        
        # Train agent
        if agent.replay_buffer.buffer_counter >= agent.min_buffer_size:
            critic_loss, actor_loss = agent.train()
            episode_critic_loss.append(critic_loss)
            episode_actor_loss.append(actor_loss)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    # Store episode results
    episode_rewards.append(episode_reward)
    if episode_critic_loss:
        critic_losses.append(np.mean(episode_critic_loss))
        actor_losses.append(np.mean(episode_actor_loss))
    
    # Calculate running average
    if len(episode_rewards) >= 100:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        # Check if we've solved the environment
        if avg_reward >= 200: # increase to 200 for better test performance
            print(f"Environment solved in {episode} episodes!")
            break
    
    # Print progress
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, "
              f"Average Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")

# Calculate training time
training_time = time.time() - start_time

# Save the trained model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
agent.save(
    f"lunarlander_ddpg_actor_weights_yinh4.pth",
    f"lunarlander_ddpg_critic_weights_yinh4.pth"
)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot rewards
ax1.plot(episode_rewards, label='Episode Reward', alpha=0.6)
ax1.plot(range(99, len(avg_rewards) + 99), avg_rewards, label='100-episode Average')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Progress - Rewards')
ax1.legend()

# Plot losses
ax2.plot(critic_losses, label='Critic Loss', alpha=0.6)
ax2.plot(actor_losses, label='Actor Loss', alpha=0.6)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss')
ax2.set_title('Training Progress - Losses')
ax2.legend()

plt.tight_layout()
plt.savefig(f'training_results.png')
plt.close()

# Print final statistics
print(f"\nTraining completed in {training_time:.2f} seconds")
print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")