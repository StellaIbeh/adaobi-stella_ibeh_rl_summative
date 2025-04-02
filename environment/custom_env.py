import gym
from gym import spaces
import numpy as np
from environment.rendering import Renderer
import time

class RehabEnv(gym.Env):
    def __init__(self):
        super(RehabEnv, self).__init__()
        
        # Define the state space: joint angles, muscle activation, feedback
        self.observation_space = spaces.Box(low=np.array([-np.pi]*3 + [0]*3), 
                                            high=np.array([np.pi]*3 + [1]*3), 
                                            dtype=np.float32)
        
        # Define the action space: adjust difficulty, encourage, suggest breaks
        self.action_space = spaces.Discrete(3)  # 0: adjust difficulty, 1: encourage, 2: suggest break
        
        # Initialize state variables
        self.state = None
        self.reset()

        # Initialize renderer
        self.renderer = Renderer()

    def step(self, action):
        # Implement the logic for taking a step in the environment
        # Update state based on action and calculate reward
        posture_correct = self.check_posture()  # Placeholder for posture checking logic
        reward = 1 if posture_correct else -1
        
        # Update state (this is a placeholder, implement actual state update logic)
        self.state = self.update_state(action)
        
        done = self.is_done()  # Placeholder for done condition
        return self.state, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))  # Random joint angles
        self.state = np.concatenate((self.state, np.zeros(3)))  # Add muscle activation and feedback
        return self.state

    def render(self, mode='human'):
        # Update the renderer with current state information
        is_correct_posture = self.check_posture()
        
        # Calculate progress based on timesteps (simple placeholder)
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        progress = min(1.0, self.step_count / 100.0)  # Progress maxes at 100 steps
        
        # Pass the entire state to the renderer
        self.renderer.update_progress(progress, is_correct_posture, self.state)
        
        # Add a small delay to make the visualization smoother
        time.sleep(0.1)

    def check_posture(self):
        # Placeholder for posture checking logic
        return True  # Assume posture is correct for now

    def update_state(self, action):
        """Update the state based on the action taken by the agent"""
        # Copy current state 
        new_state = self.state.copy()
        
        # Update joint angles based on action
        if action == 0:  # Adjust difficulty - make small corrections
            new_state[:3] += np.random.uniform(-0.1, 0.1, size=3)
            
            # Increase muscle activation slightly
            new_state[3:6] = np.clip(new_state[3:6] + 0.1, 0, 1)
            
        elif action == 1:  # Encourage - move toward correct posture
            # Move joints toward neutral position (0,0,0)
            new_state[:3] = new_state[:3] * 0.8  # Reduce deviation by 20%
            
            # Increase muscle activation moderately
            new_state[3:6] = np.clip(new_state[3:6] + 0.2, 0, 1)
            
        elif action == 2:  # Suggest break - relax muscles
            # Little change in joint angles
            new_state[:3] += np.random.uniform(-0.05, 0.05, size=3)
            
            # Decrease muscle activation
            new_state[3:6] = np.clip(new_state[3:6] - 0.3, 0, 1)
        
        # Ensure values stay within bounds
        new_state[:3] = np.clip(new_state[:3], -np.pi, np.pi)
        new_state[3:6] = np.clip(new_state[3:6], 0, 1)
        
        return new_state

    def is_done(self):
        # Placeholder for done condition
        return False  # Assume the episode is never done for now

    def set_posture_state(self, posture_type):
        """Set the environment state to simulate different postures"""
        if posture_type == "correct":
            # Well-aligned joints
            self.state[:3] = np.array([0.0, 0.0, 0.0])
            return True
        elif posture_type == "incorrect_slouch":
            # Slouched posture
            self.state[:3] = np.array([-0.5, 0.5, -0.3])
            return False
        elif posture_type == "incorrect_overextended":
            # Overextended posture
            self.state[:3] = np.array([1.5, -1.5, 0.8])
            return False
        else:
            # Default - random posture
            self.state[:3] = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
            return True  # Assume correct by default