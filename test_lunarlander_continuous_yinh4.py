import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import time
from ddpg_yinh4 import DDPGAgent

# set to True to render the environment in a GUI window
GUI_RENDER = False

# Set the number of test episodes
num_tests = 1

def main():
    """Main function to test the trained DDPG agent on LunarLanderContinuous-v2."""
    # Create the LunarLanderContinuous-v2 environment with rendering
    if GUI_RENDER:
        env = gym.make('LunarLanderContinuous-v2', render_mode='human')
    else:
        # Create the base environment
        env = gym.make('LunarLanderContinuous-v2', render_mode='rgb_array')
        
        # Wrap the environment with RecordVideo
        env = RecordVideo(
            env,
            video_folder=".",
            name_prefix="lunarlander_demo_yinh4",
            episode_trigger=lambda x: True  # Record all episodes
        )

    # Initialize the DDPG agent with environment's state and action dimensions
    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )

    # Load the trained weights into the agent
    agent.load_weights('lunarlander_ddpg_actor_weights_yinh4.pth',
                      'lunarlander_ddpg_critic_weights_yinh4.pth')

    # Run the test episodes
    for test in range(1, num_tests + 1):
        # Reset the environment and get the initial state
        state = env.reset()
        if isinstance(state, tuple):  # Handle new gym return type
            state = state[0]
        episode_reward = 0

        done = False
        truncated = False

        print(f"Starting Test Episode {test}")

        # Run the episode until it's done or truncated
        while not (done or truncated):
            # Select an action based on the current state without exploration noise
            action = agent.select_action(state, noise=False)  # Changed from noise=None to noise=False

            # Take the action and observe the result
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Handle next_state unpacking
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Accumulate the reward
            episode_reward += reward

            # Render the environment (already handled by render_mode='human')

            # Move to the next state
            state = next_state

            # Optional: Add a small sleep to control the rendering speed
            time.sleep(0.02)

        # Print the results of this test episode
        print(f"Test Episode {test}: Reward: {episode_reward:.2f}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()