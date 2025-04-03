# 🏋️ Rehabilitation Assistant using Reinforcement Learning

## Project Overview

This project implements a reinforcement learning (RL) powered rehabilitation assistant designed to guide patients through physical therapy exercises. By leveraging Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms, the system monitors joint postures and muscle activations, providing adaptive guidance to ensure correct posture during therapy sessions.

## 🧬 Environment Description

The custom Gymnasium environment (`RehabEnv`) simulates rehabilitation scenarios focusing on posture correction:

### State Space
- 6-dimensional vector:
  - 3 joint angles ranging from -π to π
  - 3 muscle activation levels ranging from 0 to 1

### Action Space
- Discrete with 3 possible actions:
  - Adjust difficulty
  - Encourage correct posture
  - Suggest breaks

### Reward Function
- `+1` for achieving correct posture (joint angles close to 0)
- `-1` for incorrect posture deviations

### Posture Types
- **Correct posture:** joint angles near 0
- **Incorrect slouch:** joint angles approximately [-0.5, 0.5, -0.3]
- **Incorrect overextension:** joint angles approximately [1.5, -1.5, 0.8]

## 🚀 Implemented Methods

- **DQN with MlpPolicy**: trained for 10,000 timesteps
- **PPO with MlpPolicy**: trained for 10,000 timesteps

## 🎨 Visualization

An interactive renderer visualizes the environment using Matplotlib, showing:
- Stick figure representation of patient
- Joint angles and muscle activation levels
- Real-time posture correctness feedback

**Generated Visualizations:**
- Individual GIF animations for DQN and PPO models demonstrating posture corrections.

## 📊 Evaluation Metrics

- DQN loss curves
- PPO policy entropy plots
- Cumulative reward trends
- Training stability comparisons
- Success rate in correcting incorrect postures
- Steps required to achieve correct posture

## 📁 Project Structure

```
rehab_rl_project/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment
│   ├── rendering.py             # Visualization system using Matplotlib
├── training/
│   ├── dqn_training.py          # DQN training script
│   ├── pg_training.py           # PPO training script
├── evaluations/
│   ├── evaluate_models.py       # Comprehensive model evaluation
│   ├── generate_plots.py        # Script to generate performance plots
├── videos/
│   ├── dqn_posture.gif          # DQN visualization
│   ├── ppo_posture.gif          # PPO visualization
├── main.py                      # Entry point for running simulations
├── play_video.py                # Utility to playback recorded simulations
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/rehab_rl_project.git
cd rehab_rl_project
pip install -r requirements.txt
```

## ⚙️ Usage Instructions

### Training Models

```bash
# Train DQN model
python training/dqn_training.py

# Train PPO model
python training/pg_training.py
```

### Generating Evaluation Plots

```bash
python evaluations/generate_plots.py
```

### Comprehensive Evaluation

```bash
python evaluations/evaluate_models.py
```

### Running Simulations

```bash
# Run a simulation
python main.py

# Playback recorded simulation
python play_video.py --model dqn
python play_video.py --model ppo
```

## 📈 Results and Analysis

### Comparative Performance

- PPO demonstrated better training stability and faster convergence compared to DQN.
- DQN showed higher variance in initial training phases but achieved comparable results with extended training.
- PPO had higher success rates in consistently correcting postures across diverse scenarios.

### Visual and Numerical Results

- Generated GIFs effectively demonstrate each model's ability to adaptively correct patient posture.
- Evaluation plots provide clear insight into training progress and performance metrics.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

