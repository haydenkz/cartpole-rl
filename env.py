
# env.py
# Defines the custom MuJoCo-based CartPole environment using Gymnasium.

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import mujoco

# The XML model for the CartPole environment.
# This defines the physical components (cart, pole), joints, and sensors.
CARTPOLE_XML = """
<mujoco model="cartpole">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <default>
    <joint limited="true"/>
    <geom contype="0" conaffinity="1" condim="1" friction=".1 .1 .1" solimp=".9 .95 .001" solref=".015 1"/>
  </default>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 0.1" type="plane"/>
    <body name="cart" pos="0 0 0">
      <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
      <geom name="cart" pos="0 0 0" rgba="0.8 0.2 0.2 1" size="0.1 0.1 0.05" type="box"/>
      <body name="pole" pos="0 0 0">
        <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-0.2 0.2" type="hinge"/>
        <geom fromto="0 0 0 0 0 0.5" name="pole" rgba="0.2 0.2 0.8 1" size="0.025" type="capsule"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="slider"/>
  </actuator>
</mujoco>
"""

class CartPoleEnv(gym.Env):
    """
    Custom MuJoCo-based CartPole environment.
    This class follows the Gymnasium API.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 100}

    def __init__(self, render_mode=None):
        super().__init__()

        # Load the MuJoCo model from the XML string.
        try:
            self.model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
        except Exception as e:
            raise IOError(f"Failed to load MuJoCo model from XML. Error: {e}")

        self.data = mujoco.MjData(self.model)

        # Define observation and action spaces.
        # Observation space: [cart_pos, cart_vel, pole_angle, pole_vel]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        # Action space: Continuous force between -1 and 1.
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

    def _get_obs(self):
        """Helper function to extract observations from the simulation state."""
        return np.array([
            self.data.qpos[0],  # Cart position
            self.data.qvel[0],  # Cart velocity
            self.data.qpos[1],  # Pole angle
            self.data.qvel[1]   # Pole angular velocity
        ], dtype=np.float64)

    def set_pole_state(self, angle, angular_velocity):
        """
        Sets the pole's angle and angular velocity directly.
        """
        self.data.qpos[1] = angle
        self.data.qvel[1] = angular_velocity
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        """
        Applies an action to the environment and steps the simulation forward.
        """
        # Apply the action to the actuator.
        self.data.ctrl[0] = action[0]

        # Step the simulation.
        mujoco.mj_step(self.model, self.data)

        # Get the new observation.
        obs = self._get_obs()

        # --- Reward Function ---
        # Reward is a combination of survival and trick incentives.
        
        # Survival reward: 1 for every step the pole is upright.
        survival_reward = 1.0
        
        # Trick reward: Encourage high angular velocity for the pole.
        pole_angular_velocity = obs[3]
        trick_reward = np.abs(pole_angular_velocity) * 0.1 # Scale factor

        reward = survival_reward + trick_reward

        # --- Termination Conditions ---
        cart_pos = obs[0]
        pole_angle = obs[2]

        # Terminate if the pole angle is too large.
        terminated = bool(
            abs(pole_angle) > 0.2  # Approx 12 degrees
        )
        # Terminate if cart goes off the track.
        truncated = bool(
            cart_pos < -1.0 or cart_pos > 1.0
        )

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        """
        super().reset(seed=seed)

        # Reset the simulation state.
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random noise to initial position and velocity
        self.data.qpos[0] = self.np_random.uniform(low=-0.05, high=0.05)
        self.data.qvel[0] = self.np_random.uniform(low=-0.05, high=0.05)
        self.data.qpos[1] = self.np_random.uniform(low=-0.15, high=0.15)
        self.data.qvel[1] = self.np_random.uniform(low=-1.0, high=1.0)

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        try:
            if self.viewer is None and self.render_mode == 'human':
                from mujoco.viewer import launch_passive
                self.viewer = launch_passive(self.model, self.data)


            if self.viewer is not None:
                self.viewer.sync()

        except ImportError:
            gym.logger.warn("mujoco-viewer is not installed. Rendering will be skipped.")
        except Exception as e:
            gym.logger.error(f"Error during rendering: {e}")
            self.close()

    def close(self):
        """
        Cleans up the environment's resources.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == '__main__':
    # Example of how to use the environment
    env = CartPoleEnv(render_mode='human')
    obs, info = env.reset()
    print(f"Initial observation: {obs}")

    # Test setting pole state
    print("\nSetting pole to 45 degrees (0.785 radians) and 0 angular velocity...")
    env.set_pole_state(angle=0.785, angular_velocity=0.0)
    obs, reward, terminated, truncated, info = env.step(np.array([0.0])) # Take a step with no action
    print(f"Observation after setting state: {obs}")

    print("\nRunning for 100 steps with no action to observe pole behavior...")
    for _ in range(100):
        action = np.array([0.0]) # No action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = env.reset()
            break
    print(f"Observation after 100 steps: {obs}")

    env.close()
