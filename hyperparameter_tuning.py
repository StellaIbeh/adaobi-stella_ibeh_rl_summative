import os
import json
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import optuna
from datetime import datetime
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import RehabEnv

# Constants
TOTAL_TIMESTEPS = 1000000  # 1 million timesteps
EVAL_EPISODES = 10
TUNING_TIMESTEPS = 200000  # Use fewer timesteps for tuning trials
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_PATH, "models", "tuning_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_env():
    """Create a vectorized environment"""
    return DummyVecEnv([lambda: RehabEnv()])

def objective_dqn(trial):
    """Objective function for DQN hyperparameter optimization"""
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.3)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    target_update_interval = trial.suggest_categorical("target_update_interval", [500, 1000, 5000])
    learning_starts = trial.suggest_categorical("learning_starts", [1000, 5000, 10000])
    
    # Create environment
    env = create_env()
    
    # Create eval environment
    eval_env = RehabEnv()
    
    # Use a timestamp for model path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(RESULTS_DIR, f"dqn_trial_{trial.number}_{timestamp}")
    
    # Create the callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=model_path,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=EVAL_EPISODES
    )
    
    # Initialize the model with the hyperparameters
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        learning_starts=learning_starts,
        verbose=1
    )
    
    try:
        # Train the model
        model.learn(total_timesteps=TUNING_TIMESTEPS, callback=eval_callback)
        
        # Load the best model
        model = DQN.load(os.path.join(model_path, "best_model"))
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES)
        
        # Save the hyperparameters and results
        results = {
            "trial_number": trial.number,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "target_update_interval": target_update_interval,
            "learning_starts": learning_starts,
            "mean_reward": mean_reward,
            "std_reward": std_reward
        }
        
        with open(os.path.join(model_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        return mean_reward
    
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        return float('-inf')  # Return negative infinity for failed trials

def objective_ppo(trial):
    """Objective function for PPO hyperparameter optimization"""
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    
    # Create environment
    env = RehabEnv()  # PPO doesn't require a vectorized environment
    
    # Create eval environment
    eval_env = RehabEnv()
    
    # Use a timestamp for model path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(RESULTS_DIR, f"ppo_trial_{trial.number}_{timestamp}")
    
    # Create the callback for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=model_path,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=EVAL_EPISODES
    )
    
    # Initialize the model with the hyperparameters
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        n_epochs=n_epochs,
        verbose=1
    )
    
    try:
        # Train the model
        model.learn(total_timesteps=TUNING_TIMESTEPS, callback=eval_callback)
        
        # Load the best model
        model = PPO.load(os.path.join(model_path, "best_model"))
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES)
        
        # Save the hyperparameters and results
        results = {
            "trial_number": trial.number,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "n_epochs": n_epochs,
            "mean_reward": mean_reward,
            "std_reward": std_reward
        }
        
        with open(os.path.join(model_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        return mean_reward
    
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        return float('-inf')  # Return negative infinity for failed trials

def train_best_model(best_params, algorithm="dqn"):
    """Train the model with the best hyperparameters for the full timesteps"""
    print(f"\n{'='*50}")
    print(f"Training best {algorithm.upper()} model with {TOTAL_TIMESTEPS} timesteps")
    print(f"Using parameters: {best_params}")
    print(f"{'='*50}\n")
    
    # Create the environment
    if algorithm.lower() == "dqn":
        env = create_env()
        model = DQN("MlpPolicy", env, **best_params, verbose=1)
        model_save_path = os.path.join(BASE_PATH, "models", "dqn", "dqn_rehab_model_tuned")
    else:  # PPO
        env = RehabEnv()
        model = PPO("MlpPolicy", env, **best_params, verbose=1)
        model_save_path = os.path.join(BASE_PATH, "models", "pg", "ppo_model_tuned")
    
    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save the model
    model.save(model_save_path)
    print(f"\nBest {algorithm.upper()} model saved to {model_save_path}")
    
    # Also save a copy with timestamp for reference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = f"{model_save_path}_{timestamp}"
    model.save(timestamped_path)
    
    # Save the parameters
    with open(f"{model_save_path}_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    
    return model_save_path

def analyze_results(study, algorithm):
    """Analyze and print the results of the optimization"""
    print(f"\n{'='*50}")
    print(f"Results for {algorithm.upper()} optimization")
    print(f"{'='*50}")
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best mean reward: {study.best_value}")
    print("\nBest hyperparameters:")
    
    # Extract and print best params
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Create analysis report
    analysis_path = os.path.join(RESULTS_DIR, f"{algorithm}_optimization_results.txt")
    with open(analysis_path, "w") as f:
        f.write(f"Results for {algorithm.upper()} optimization\n")
        f.write("="*50 + "\n\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best mean reward: {study.best_value}\n\n")
        f.write("Best hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        
        # Add justification
        f.write("\nJustification for hyperparameter choices:\n")
        
        if algorithm.lower() == "dqn":
            f.write(f"- Learning Rate ({best_params['learning_rate']:.6f}): Controls step size during optimization. ")
            f.write("This value balances between convergence speed and stability.\n")
            
            f.write(f"- Buffer Size ({best_params['buffer_size']}): Size of replay buffer. ")
            f.write("Larger buffers store more experiences, helping with stable learning from diverse samples.\n")
            
            f.write(f"- Batch Size ({best_params['batch_size']}): Number of samples per gradient update. ")
            f.write("This value balances between speed (larger) and variance (smaller).\n")
            
            f.write(f"- Gamma ({best_params['gamma']:.4f}): Discount factor for future rewards. ")
            f.write("Higher values prioritize long-term rewards, which is important in rehabilitation.\n")
            
            f.write(f"- Exploration Fraction ({best_params['exploration_fraction']:.4f}): ")
            f.write("Fraction of training time spent on exploration, important for discovering optimal postures.\n")
            
            f.write(f"- Final Exploration Rate ({best_params['exploration_final_eps']:.4f}): ")
            f.write("Minimum exploration rate, allowing model to maintain some exploration at end of training.\n")
            
            f.write(f"- Target Update Interval ({best_params['target_update_interval']}): ")
            f.write("How often to update target network, affecting stability of learning.\n")
            
            f.write(f"- Learning Starts ({best_params['learning_starts']}): ")
            f.write("Training starts after this many steps, allowing buffer to fill with experiences first.\n")
        else:  # PPO
            f.write(f"- Learning Rate ({best_params['learning_rate']:.6f}): Controls step size during optimization. ")
            f.write("This value balances between convergence speed and stability.\n")
            
            f.write(f"- n_steps ({best_params['n_steps']}): Number of steps to run for each environment per update. ")
            f.write("This affects the trade-off between bias and variance in policy gradient estimation.\n")
            
            f.write(f"- Batch Size ({best_params['batch_size']}): Number of samples per gradient update. ")
            f.write("This value balances between speed (larger) and variance (smaller).\n")
            
            f.write(f"- Gamma ({best_params['gamma']:.4f}): Discount factor for future rewards. ")
            f.write("Higher values prioritize long-term rewards, which is important in rehabilitation.\n")
            
            f.write(f"- GAE Lambda ({best_params['gae_lambda']:.4f}): ")
            f.write("Factor for trade-off of bias vs variance in Generalized Advantage Estimation.\n")
            
            f.write(f"- Clip Range ({best_params['clip_range']:.4f}): ")
            f.write("Clipping parameter for PPO, constraining policy updates for stability.\n")
            
            f.write(f"- Entropy Coefficient ({best_params['ent_coef']:.4f}): ")
            f.write("Entropy regularization coefficient, encouraging exploration.\n")
            
            f.write(f"- n_epochs ({best_params['n_epochs']}): ")
            f.write("Number of epochs when optimizing the surrogate loss, affecting training stability.\n")
        
        f.write("\nOptimization for this rehabilitation environment requires balancing ")
        f.write("between exploration (finding proper postures) and exploitation (maximizing rewards).\n")
        f.write("These parameters were selected to maximize the agent's ability to guide patients ")
        f.write("toward proper posture while adapting to different patient needs.\n")
    
    print(f"\nDetailed analysis saved to {analysis_path}")
    return best_params

def main():
    # Create the results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create a timestamp for this tuning session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting hyperparameter tuning session at {timestamp}")
    
    # Number of trials
    n_trials = 15  # Adjust based on time constraints
    
    # 1. DQN Optimization
    print("\nStarting DQN hyperparameter optimization...")
    dqn_study = optuna.create_study(direction="maximize", 
                                  study_name=f"dqn_optimization_{timestamp}")
    dqn_study.optimize(objective_dqn, n_trials=n_trials)
    
    # Analyze DQN results
    dqn_best_params = analyze_results(dqn_study, "dqn")
    
    # 2. PPO Optimization
    print("\nStarting PPO hyperparameter optimization...")
    ppo_study = optuna.create_study(direction="maximize", 
                                  study_name=f"ppo_optimization_{timestamp}")
    ppo_study.optimize(objective_ppo, n_trials=n_trials)
    
    # Analyze PPO results
    ppo_best_params = analyze_results(ppo_study, "ppo")
    
    # 3. Train the best models with full timesteps
    dqn_model_path = train_best_model(dqn_best_params, "dqn")
    ppo_model_path = train_best_model(ppo_best_params, "ppo")
    
    # 4. Create a summary report
    summary_path = os.path.join(RESULTS_DIR, f"hyperparameter_tuning_summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Hyperparameter Tuning Summary ({timestamp})\n")
        f.write("="*50 + "\n\n")
        
        f.write("Methodology:\n")
        f.write(f"- Used Optuna for hyperparameter optimization with {n_trials} trials per algorithm\n")
        f.write(f"- Initial tuning with {TUNING_TIMESTEPS} timesteps\n")
        f.write(f"- Final training with {TOTAL_TIMESTEPS} timesteps\n\n")
        
        f.write("DQN Best Parameters:\n")
        for param, value in dqn_best_params.items():
            f.write(f"- {param}: {value}\n")
        f.write(f"Saved model: {dqn_model_path}\n\n")
        
        f.write("PPO Best Parameters:\n")
        for param, value in ppo_best_params.items():
            f.write(f"- {param}: {value}\n")
        f.write(f"Saved model: {ppo_model_path}\n\n")
        
        # Compare the algorithms
        f.write("Algorithm Comparison:\n")
        f.write(f"- DQN best reward: {dqn_study.best_value}\n")
        f.write(f"- PPO best reward: {ppo_study.best_value}\n")
        
        if dqn_study.best_value > ppo_study.best_value:
            f.write("\nDQN outperformed PPO in this environment, suggesting that ")
            f.write("the rehabilitation task benefits from value-based methods that can ")
            f.write("precisely estimate expected rewards for different postures.\n")
        else:
            f.write("\nPPO outperformed DQN in this environment, suggesting that ")
            f.write("the rehabilitation task benefits from policy-based methods that can ")
            f.write("learn a direct mapping from states to actions.\n")
        
        f.write("\nRecommendation:\n")
        f.write("After hyperparameter tuning and training, we recommend using the ")
        f.write("best-performing algorithm for the rehabilitation assistant, as it ")
        f.write("demonstrates superior ability to guide patients through exercises with correct posture.\n")
    
    print(f"\nHyperparameter tuning complete. Summary saved to {summary_path}")
    print(f"Best DQN model saved to {dqn_model_path}")
    print(f"Best PPO model saved to {ppo_model_path}")
    print("\nYou can now run main.py to test the tuned models.")

if __name__ == "__main__":
    main()