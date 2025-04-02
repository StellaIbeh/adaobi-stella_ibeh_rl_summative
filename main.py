import gym
from environment.custom_env import RehabEnv
from stable_baselines3 import DQN, PPO
import os
import sys
import numpy as np
from datetime import datetime

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

def simulate_with_posture(env, model, model_name, posture_type, num_steps=30):
    """Run a simulation with a specific posture type"""
    # Create a unique session ID for this simulation
    env.renderer.session_id = f"{model_name}_{posture_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Reset frames for new simulation
    env.renderer.frames = []
    env.renderer.frame_count = 0
    
    # Reset the figure to ensure it's initialized properly
    env.renderer.fig = None
    
    # Set initial posture based on posture_type
    obs = env.reset()
    
    # Override with specific posture
    if posture_type == "correct":
        # Set a correct posture (well-aligned joints)
        obs[:3] = np.array([0.0, 0.0, 0.0])  # Neutral position
    elif posture_type == "incorrect_slouch":
        # Set an incorrect slouching posture
        obs[:3] = np.array([-0.5, 0.5, -0.3])  # Slouched
    elif posture_type == "incorrect_overextended":
        # Set an overextended posture
        obs[:3] = np.array([1.5, -1.5, 0.8])  # Overextended
    
    # Set the environment state
    env.state = obs
    
    # Override the check_posture method to return True/False based on posture_type
    # But make it dynamic - start with incorrect and let the agent try to correct it
    if posture_type == "correct":
        # Correct posture remains correct
        env.check_posture = lambda: True
    else:
        # For incorrect posture, it becomes correct once the joint angles get close to neutral
        # This lets us see the agent learning to correct posture
        env.check_posture = lambda: np.sum(np.abs(env.state[:3])) < 0.5  # Threshold for "correct" posture
    
    # Run simulation
    print(f"\n=== Running {model_name} Model with {posture_type} posture ===")
    
    total_reward = 0
    for i in range(num_steps):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Show if the posture has been corrected
        posture_status = "Correct" if env.check_posture() else "Incorrect"
        print(f"Step {i+1}: Action={action}, Reward={reward}, Posture={posture_status}, Total Reward={total_reward}")
        
        # Render first, then add title after figure is created
        env.render()
        
        # Now the figure should exist, so add the title
        if env.renderer.fig is not None:
            env.renderer.fig.suptitle(
                f"Rehabilitation Assistant - {model_name} Model - {posture_type.capitalize()} Posture\n" + 
                f"Step: {i+1}/{num_steps}, Action: {action}, Reward: {reward}, Total: {total_reward}",
                fontsize=16, fontweight='bold'
            )
        
        if done:
            break
    
    # Create video file for this simulation
    video_path = env.renderer.finalize_video(model_name, posture_type)
    print(f"Completed {model_name} simulation with {posture_type} posture - Total Reward: {total_reward}")
    return video_path

def main():
    env = RehabEnv()
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Use absolute paths with .zip extension
    dqn_model_path = os.path.join(base_path, "models", "dqn", "dqn_rehab_model")
    pg_model_path = os.path.join(base_path, "models", "pg", "ppo_model")
    
    print(f"Current directory: {os.getcwd()}")
    print(f"DQN model path: {dqn_model_path}")
    print(f"PPO model path: {pg_model_path}")
    
    # Load models
    dqn_model = load_model(dqn_model_path)
    pg_model = load_model(pg_model_path)

    # List of posture types to test
    posture_types = ["correct", "incorrect_slouch", "incorrect_overextended"]
    
    # Run DQN model with different postures
    for posture in posture_types:
        try:
            simulate_with_posture(env, dqn_model, "DQN", posture)
            print(f"Successfully completed DQN simulation with {posture} posture")
        except Exception as e:
            print(f"Error in DQN simulation with {posture} posture: {e}")
    
    # Run PPO model with different postures
    for posture in posture_types:
        try:
            simulate_with_posture(env, pg_model, "PPO", posture)
            print(f"Successfully completed PPO simulation with {posture} posture")
        except Exception as e:
            print(f"Error in PPO simulation with {posture} posture: {e}")
    
    # Try to create combined video, but don't fail if it doesn't work
    try:
        video_path = env.renderer.create_video("complete_simulation")
        if video_path:
            print(f"\nCreated combined video at: {video_path}")
        else:
            print("\nFailed to create combined video, but individual frames and GIFs are available.")
    except Exception as e:
        print(f"\nError creating combined video: {e}")
        print("Individual frames and GIFs are still available in the visualization_output directory.")
    
    # Close environment
    env.close()
    
    print("\nAll simulations completed. Check the visualization_output directory for results.")

if __name__ == "__main__":
    main()