import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import time
from ppo_yinh4 import PPOAgent
import numpy as np

# set to True to render the environment in a GUI window
GUI_RENDER = False

# Set the number of test episodes
num_tests = 1

def main():
    """Main function to test the trained PPO agent on BipedalWalker-v3."""
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the environment with the exact same wrappers as training
    env = gym.make('BipedalWalker-v3', render_mode='human' if GUI_RENDER else 'rgb_array')
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=0.99)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    
    if not GUI_RENDER:
        # Wrap with video recording
        env = RecordVideo(
            env,
            video_folder=".",
            name_prefix="bipedalwalker_demo_yinh4",
            episode_trigger=lambda x: True
        )

    # Initialize the PPO agent with vectorized environment
    envs = gym.vector.SyncVectorEnv([lambda: env])
    agent = PPOAgent(envs).to(device)

    # Load the trained weights
    agent.load_state_dict(torch.load('bipedalwalker_actor_yinh4.pth', map_location=device))
    agent.eval()

    # Run the test episodes
    total_reward = 0
    for test in range(1, num_tests + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        print(f"Starting Test Episode {test}")

        while not (done or truncated):
            # Add unsqueeze to match the expected batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(state_tensor)
                action = action.cpu().numpy().flatten()

            # Take action in environment
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            if GUI_RENDER:
                time.sleep(0.02)  # Optional: Add a small sleep to control the rendering speed

        print(f"Test Episode {test}: Reward: {episode_reward:.2f}")
        total_reward += episode_reward

    print(f"\nAverage Reward over {num_tests} episodes: {total_reward/num_tests:.2f}")
    env.close()

if __name__ == "__main__":
    main()