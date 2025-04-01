import gym
from environment.custom_env import RehabEnv
from stable_baselines3 import DQN, PPO
import os
import sys

def load_model(model_path):
    # Add .zip extension if not present
    if not model_path.endswith('.zip'):
        model_path = f"{model_path}.zip"
    
    print(f"Attempting to load model from: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        try:
            return DQN.load(model_path) if 'dqn' in model_path else PPO.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

def main():
    env = RehabEnv()
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Use absolute paths with .zip extension
    dqn_model_path = os.path.join(base_path, "models", "dqn", "dqn_rehab_model")
    pg_model_path = os.path.join(base_path, "models", "pg", "ppo_model")
    
    print(f"Current directory: {os.getcwd()}")
    print(f"DQN model path: {dqn_model_path}")
    print(f"PPO model path: {pg_model_path}")
    
    dqn_model = load_model(dqn_model_path)
    pg_model = load_model(pg_model_path)

    obs = env.reset()
    done = False

    while not done:
        action_dqn, _ = dqn_model.predict(obs)
        obs, reward, done, info = env.step(action_dqn)
        env.render()

    obs = env.reset()
    done = False

    while not done:
        action_pg, _ = pg_model.predict(obs)
        obs, reward, done, info = env.step(action_pg)
        env.render()

    env.close()

if __name__ == "__main__":
    main()