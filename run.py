# run.py
# Loads a trained PPO model and tests its performance on the custom CartPole environment.

import torch
import numpy as np
import time
from env import CartPoleEnv
from train import ActorCritic, MODEL_PATH # Import model and path from training script

# --- Parameters ---
NUM_TEST_EPISODES = 10
DETERMINISTIC = True # Use deterministic actions for testing

def main():
    """Main function to load model and run tests."""
    # Initialize environment with human rendering
    env = CartPoleEnv(render_mode='human')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load the trained model
    model = ActorCritic(obs_dim, action_dim)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Set the model to evaluation mode
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
        env.close()
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        env.close()
        return

    total_rewards = []
    
    print(f"Running {NUM_TEST_EPISODES} test episodes...")
    for episode in range(NUM_TEST_EPISODES):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                dist, _ = model(state_tensor)
                if DETERMINISTIC:
                    # For testing, we take the mean of the distribution (most likely action)
                    action = dist.mean
                else:
                    # Sample from the distribution
                    action = dist.sample()

            state, reward, done, truncated, _ = env.step(action.numpy()[0])
            episode_reward += reward
            env.render()
            
            # Optional: add a small delay to make rendering easier to watch
            time.sleep(0.01)

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    # --- Print Results ---
    avg_reward = np.mean(total_rewards)
    print("\n--- Test Results ---")
    print(f"Average Reward over {NUM_TEST_EPISODES} episodes: {avg_reward:.2f}")
    
    env.close()

if __name__ == '__main__':
    main()

