import gym
import torch
import argparse
import time
from reinforce_yinh4 import REINFORCEAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(logs):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Plot episode rewards
    ax1.plot(logs['episode_rewards'], label='Episode Reward', alpha=0.7)
    ax1.plot(np.convolve(logs['episode_rewards'], np.ones(100)/100, mode='valid'), 
             label='100-episode Running Mean')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Episode Rewards')
    ax1.legend()

    # Plot training losses
    ax2.plot(logs['training_losses'], label='Training Loss', alpha=0.7)
    ax2.plot(np.convolve(logs['training_losses'], np.ones(100)/100, mode='valid'), 
             label='100-episode Running Mean')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Losses')
    ax2.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('training_results_yinh4.png')
    plt.close()

def main():
    # Create the CartPole environment
    env = gym.make('CartPole-v0')
    
    # Initialize the REINFORCE agent
    agent = REINFORCEAgent(env)
    
    # Set training parameters
    max_episodes = 1000 # max episodes, likely ends before
    solved_reward = 195  # The reward threshold for considering the environment solved
    
    # Initialize logs to track training progress
    logs = {
        'episode_rewards': [],
        'training_losses': [],
        'episode_times': []
    }
    
    # Record the start time of the entire training process
    start_time = time.time()
    
    # Main training loop
    for episode in tqdm(range(1, max_episodes + 1), desc="Training"):
        # Reset the environment and get the initial state
        state, _ = env.reset()  # Unpack the state from the reset return value
        episode_reward = 0
        episode_start_time = time.time()
        
        # Episode loop
        while True:
            # Select an action based on the current state
            action = agent.select_action(state)
            
            # Take the action and observe the next state, reward, and done flag
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store the reward for this step
            agent.rewards.append(reward)
            episode_reward += reward
            
            # Move to the next state
            state = next_state
            
            # Check if the episode has ended
            if done or truncated:
                break
        
        # Perform the REINFORCE update after the episode ends
        loss = agent.finish_episode()
        
        # Calculate the duration of this episode
        episode_time = time.time() - episode_start_time

        # Log the episode results
        logs['episode_rewards'].append(episode_reward)
        logs['episode_times'].append(episode_time)
        logs['training_losses'].append(loss.item())  # Convert tensor to scalar

        # Check if the environment is solved (after at least 100 episodes)
        if episode >= 100:
            avg_reward = sum(logs['episode_rewards'][-100:]) / 100
            if avg_reward >= solved_reward:
                print(f"Solved! Episode: {episode}, Average Reward: {avg_reward:.2f}")
                agent.save_weights('cartpole_reinforce_weights_yinh4.pth')
                break

        # Print progress every 100 episodes
        if episode % 100 == 0:
            avg_reward = sum(logs['episode_rewards'][-100:]) / 100
            tqdm.write(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
    
    # Calculate and print the total training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    # Plot the training results
    plot_training_results(logs)
    
    # Close the environment
    env.close()

    # Report training time
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per episode: {np.mean(logs['episode_times']):.4f} seconds")

if __name__ == "__main__":
    main()
