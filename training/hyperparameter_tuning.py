import optuna
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import os
import numpy as np
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import RehabEnv

def optimize_dqn(trial):
    """Optimize DQN hyperparameters"""
    # Define the hyperparameters to tune
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'buffer_size': trial.suggest_int('buffer_size', 10000, 100000),
        'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
        'batch_size': trial.suggest_int('batch_size', 32, 256),
        'gamma': trial.suggest_float('gamma', 0.9, 0.99999),
        'exploration_fraction': trial.suggest_float('exploration_fraction', 0.1, 0.5),
        'exploration_initial_eps': trial.suggest_float('exploration_initial_eps', 0.5, 1.0),
        'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.1),
    }

    # Create environment
    env = RehabEnv()
    
    # Create model with trial parameters
    model = DQN('MlpPolicy', env, 
                verbose=0,
                tensorboard_log=f"./dqn_rehab_tensorboard/trial_{trial.number}",
                **params)
    
    try:
        # Train the model
        model.learn(total_timesteps=5000)
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        
        # Clean up
        env.close()
        
        return mean_reward
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')

def optimize_ppo(trial):
    """Optimize PPO hyperparameters"""
    # Define the hyperparameters to tune
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'n_steps': trial.suggest_int('n_steps', 32, 2048),
        'batch_size': trial.suggest_int('batch_size', 32, 256),
        'n_epochs': trial.suggest_int('n_epochs', 5, 20),
        'gamma': trial.suggest_float('gamma', 0.9, 0.99999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 1.0),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.01),
    }

    # Create environment
    env = RehabEnv()
    
    # Create model with trial parameters
    model = PPO('MlpPolicy', env,
                verbose=0,
                tensorboard_log=f"./ppo_rehab_tensorboard/trial_{trial.number}",
                **params)
    
    try:
        # Train the model
        model.learn(total_timesteps=5000)
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        
        # Clean up
        env.close()
        
        return mean_reward
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')

def save_best_params(study, algorithm):
    """Save the best parameters to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, f"best_{algorithm}_params_{timestamp}.txt")
    
    with open(filepath, 'w') as f:
        f.write(f"Best {algorithm} parameters found:\n")
        f.write("============================\n")
        f.write(f"Best Value: {study.best_value}\n\n")
        f.write("Parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nBest parameters saved to: {filepath}")

def main():
    # Number of trials for optimization
    n_trials = 50
    
    print("Starting hyperparameter optimization...")
    
    # Optimize DQN
    print("\nOptimizing DQN...")
    dqn_study = optuna.create_study(direction='maximize')
    dqn_study.optimize(optimize_dqn, n_trials=n_trials)
    save_best_params(dqn_study, 'dqn')
    
    # Optimize PPO
    print("\nOptimizing PPO...")
    ppo_study = optuna.create_study(direction='maximize')
    ppo_study.optimize(optimize_ppo, n_trials=n_trials)
    save_best_params(ppo_study, 'ppo')
    
    print("\nOptimization completed!")
    print("\nBest DQN parameters:")
    print("===================")
    print(f"Best Value: {dqn_study.best_value}")
    print("Parameters:")
    for key, value in dqn_study.best_params.items():
        print(f"{key}: {value}")
        
    print("\nBest PPO parameters:")
    print("===================")
    print(f"Best Value: {ppo_study.best_value}")
    print("Parameters:")
    for key, value in ppo_study.best_params.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()