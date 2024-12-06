import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import random
import torch.optim as optim

from distutils.util import strtobool
from ppo_yinh4 import PPOAgent

from tqdm import tqdm 

def train_agent(args, agent, optimizer, obs, actions, logprobs, rewards, dones, values, episode_rewards, actor_losses, critic_losses, start_time, window_size, global_step, next_obs, next_done, envs, device):
    progress_bar = tqdm(range(1, args.num_iterations + 1), desc="Training")
    for iteration in progress_bar:
        
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            # Action selection
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Log episode rewards
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_reward = float(info['episode']['r'])
                        episode_rewards.append(episode_reward)
                        # Calculate moving average and throughput
                        current_avg = np.mean(episode_rewards[-window_size:])
                        throughput = int(global_step / (time.time() - start_time))
                        
                        # Early stopping condition
                        if current_avg >= 225:
                            print(f"\nReached target performance! Moving average: {current_avg:.2f}")
                            return
                        
                        # Update progress bar description
                        progress_bar.set_postfix({
                            'step': global_step,
                            'reward': f'{episode_reward:.1f}',
                            f'{min(len(episode_rewards), window_size)}-ep avg reward': f'{current_avg:.1f}',
                            'throughput': f'{throughput} SPS'
                        })

        # Bootstrap value calculation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # Calculate temporal difference error: (reward + discounted_next_value - current_value)
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # Calculate advantage using GAE: accumulate discounted sum of advantages
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            # Total expected returns = advantages + value estimates
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                # Calculate probability ratio between new and old policies
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Normalize advantages if enabled (zero mean, unit variance)
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Calculate PPO clipped objective: min of normal ratio and clipped ratio times advantage
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Calculate value loss with optional clipping
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # Squared error between new value and returns
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    # Clip value function updates similar to policy updates
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    # Use maximum of clipped and unclipped value losses
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # Simple MSE loss if clipping is disabled
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Calculate entropy bonus for exploration
                entropy_loss = entropy.mean()
                # Combine all losses: policy loss - entropy bonus + value loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Store losses for plotting
                actor_losses.append(pg_loss.item())
                critic_losses.append(v_loss.item())

def main(args):
    """
    Main function to train the PPO agent on BipedalWalker-v3 environment.
    """
    # Initialize hyperparameters
    args.batch_size = int(1 * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    print("Hyperparameters:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Determine device to use (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment setup
    env = gym.make("BipedalWalker-v3")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    envs = gym.vector.SyncVectorEnv([lambda: env])

    # Initialize PPO agent and optimizer
    agent = PPOAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup for training data
    obs = torch.zeros((args.num_steps, 1) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, 1) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, 1)).to(device)
    rewards = torch.zeros((args.num_steps, 1)).to(device)
    dones = torch.zeros((args.num_steps, 1)).to(device)
    values = torch.zeros((args.num_steps, 1)).to(device)

    # Tracking variables for performance metrics
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    start_time = time.time()
    window_size = 100  # Define window size for moving average

    # Training loop
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)

    train_agent(
        args, agent, optimizer, obs, actions, logprobs, 
        rewards, dones, values, episode_rewards, actor_losses, 
        critic_losses, start_time, window_size, global_step, 
        next_obs, next_done, envs, device)

    # Save the trained model
    torch.save(agent.state_dict(), 'bipedalwalker_actor_yinh4.pth')
    torch.save(agent.state_dict(), 'bipedalwalker_critic_yinh4.pth')
    envs.close()
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot rewards
    ax1.plot(episode_rewards, label='Episode Reward', alpha=0.6)
    ax1.plot([np.mean(episode_rewards[max(0, i-window_size):i]) for i in range(1, len(episode_rewards)+1)], 
             label='100-episode Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress - Rewards')
    ax1.legend()

    # Plot losses
    ax2.plot(critic_losses, label='Critic Loss', alpha=0.6)
    ax2.plot(actor_losses, label='Actor Loss', alpha=0.6)
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Progress - Losses')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

    # Print final statistics
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")

    envs.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--total-timesteps", type=int, default=2000000, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=2048, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")

    args = parser.parse_args()
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    main(args)