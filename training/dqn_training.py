from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import RehabEnv

def train_dqn():
    # Create the environment
    env = DummyVecEnv([lambda: RehabEnv()])

    # Initialize the DQN agent
    model = DQN('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Get the absolute path for saving the model
    model_save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "dqn", "dqn_rehab_model"
    )
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save the trained model
    model.save(model_save_path)

if __name__ == "__main__":
    train_dqn()