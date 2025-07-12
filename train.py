
# train.py
# Implements a PPO agent using PyTorch to train on the custom CartPole environment.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from env import CartPoleEnv  # Import the custom environment

# --- Hyperparameters ---
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95 # Lambda for Generalized Advantage Estimation
PPO_EPSILON = 0.2 # Clip parameter for PPO
PPO_EPOCHS = 10   # Number of epochs for updating policy
PPO_BATCH_SIZE = 64
ROLLOUT_STEPS = 2048 # Number of steps to collect in each rollout
MAX_EPISODES = 2000
MODEL_PATH = 'ppo_cartpole.pth'

class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO.
    The actor (policy) and critic (value) networks share the initial layers.
    """
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Shared network layers
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        # Actor head: outputs mean of the action distribution
        self.actor_mean = nn.Linear(64, action_dim)
        
        # Critic head: outputs the state value
        self.critic = nn.Linear(64, 1)

        # Action standard deviation (log scale)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        """Forward pass through the network."""
        shared_features = self.shared_net(x)
        action_mean = self.actor_mean(shared_features)
        value = self.critic(shared_features)
        
        # Create a normal distribution for the actions
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        
        return dist, value

def main():
    """Main training loop."""
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment
    env = CartPoleEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize model and optimizer
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Logging
    all_episode_rewards = []
    
    print("Starting training...")
    # --- Training Loop ---
    try:
        for episode in range(MAX_EPISODES):
            # --- Experience Collection (Rollout) ---
            memory = []
            steps_collected = 0
            episode_rewards = []
            
            while steps_collected < ROLLOUT_STEPS:
                state, _ = env.reset()
                done = False
                truncated = False
                current_episode_reward = 0
                
                while not done and not truncated:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        dist, value = model(state_tensor)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)

                    next_state, reward, done, truncated, _ = env.step(action.cpu().numpy()[0])
                    
                    memory.append((state, action, reward, log_prob, value, done, truncated))
                    
                    state = next_state
                    current_episode_reward += reward
                    steps_collected += 1

                episode_rewards.append(current_episode_reward)

            # --- Process Rollout Data ---
            states, actions, rewards, log_probs, values, dones, truncateds = zip(*memory)

            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.cat(list(actions)).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            log_probs = torch.cat(list(log_probs)).to(device)
            values = torch.cat(list(values)).squeeze().to(device)
            dones = torch.FloatTensor(dones).to(device)
            truncateds = torch.FloatTensor(truncateds).to(device)

            # --- Calculate Advantages and Returns (GAE) ---
            advantages = torch.zeros_like(rewards)
            last_gae_lam = 0
            for t in reversed(range(len(rewards) - 1)):
                if t == len(rewards) - 2:
                    next_non_terminal = 1.0 - (dones[t+1] or truncateds[t+1])
                    next_value = values[t+1]
                else:
                    next_non_terminal = 1.0 - (dones[t+1] or truncateds[t+1])
                    next_value = values[t+1]
                    
                delta = rewards[t] + GAMMA * next_value * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            
            returns = advantages + values

            # --- PPO Update ---
            for _ in range(PPO_EPOCHS):
                # Create batches
                indices = np.arange(len(states))
                np.random.shuffle(indices)
                for start in range(0, len(states), PPO_BATCH_SIZE):
                    end = start + PPO_BATCH_SIZE
                    batch_indices = indices[start:end]

                    # Get batch data
                    batch_states = states[batch_indices].to(device)
                    batch_actions = actions[batch_indices].to(device)
                    batch_log_probs = log_probs[batch_indices].to(device)
                    batch_advantages = advantages[batch_indices].to(device)
                    batch_returns = returns[batch_indices].to(device)

                    # Get new log probs and values
                    new_dist, new_values = model(batch_states)
                    new_log_probs = new_dist.log_prob(batch_actions)
                    
                    # --- Policy (Actor) Loss ---
                    ratio = (new_log_probs - batch_log_probs).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # --- Value (Critic) Loss ---
                    critic_loss = (new_values.squeeze() - batch_returns).pow(2).mean()
                    
                    # --- Total Loss ---
                    loss = actor_loss + 0.5 * critic_loss

                    # --- Update ---
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            avg_reward = np.mean(episode_rewards)
            all_episode_rewards.append(avg_reward)
            print(f"Episode {episode+1}/{MAX_EPISODES}, Average Reward: {avg_reward:.2f}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")

    # --- Save Model and Plot ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(all_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig("training_progress.png")
    plt.show()

    env.close()

if __name__ == '__main__':
    main()
