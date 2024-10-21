import gym
import torch
import time
from reinforce_yinh4 import REINFORCEAgent

def main():
    # Create the CartPole environment with RGB array rendering
    env = gym.make('CartPole-v0', render_mode='rgb_array')
    
    # Wrap the environment to record video of all episodes
    env = gym.wrappers.RecordVideo(env, './cartpole_demo_yinh4', episode_trigger=lambda x: True)
    
    # Initialize the REINFORCE agent with the environment
    agent = REINFORCEAgent(env)
    
    # Load pre-trained weights for the agent
    agent.load_weights('cartpole_reinforce_weights_yinh4.pth')
    
    # Set the number of test episodes
    num_tests = 10
    
    # Run the test episodes
    for test in range(1, num_tests + 1):
        # Reset the environment and get initial state
        state, _ = env.reset()
        
        # Initialize episode termination flags
        done = False
        truncated = False
        episode_reward = 0
        
        # Run the episode until it's done or truncated
        while not (done or truncated):
            # Select an action based on the current state
            action = agent.select_action(state)
            
            # Take the action and observe the result
            state, reward, done, truncated, _ = env.step(action)
            
            # Accumulate the reward
            episode_reward += reward
        
        # Print the results of this test episode
        print(f"Test Episode {test}: Reward: {episode_reward}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()