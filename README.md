# Reinforcement Learning for Rehabilitation Assistant

This project implements a reinforcement learning (RL) solution for an AI-powered rehabilitation assistant designed to help patients perform physical therapy exercises. The project is structured to include a custom Gymnasium environment, training scripts for different RL algorithms, and visualization tools.

## Project Structure

```
student_name_rl_summative
├── environment
│   ├── custom_env.py       # Custom Gymnasium environment for rehabilitation
│   └── rendering.py        # Visualization of the rehabilitation exercises
├── training
│   ├── dqn_training.py     # Training script for Deep Q-Network (DQN)
│   └── pg_training.py      # Training script for Proximal Policy Optimization (PPO)
├── models
│   ├── dqn                 # Directory for storing trained DQN models
│   └── pg                  # Directory for storing trained PPO models
├── main.py                 # Entry point for running the project
├── requirements.txt        # List of project dependencies
└── README.md               # Project documentation
```

## Environment Setup

To set up the project environment, ensure you have Python installed. Then, create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Training Process

The project includes two training scripts:

1. **DQN Training**: Run `dqn_training.py` to train a DQN agent using the custom rehabilitation environment. The trained models will be saved in the `models/dqn` directory.

2. **PPO Training**: Run `pg_training.py` to train a PPO agent. The trained models will be saved in the `models/pg` directory.

## Running the Project

To run the project and visualize the agent's interaction with the rehabilitation environment, execute the `main.py` script:

```bash
python main.py
```

## Results

The results of the training and the agent's performance can be visualized through the rendering module, which provides feedback on posture correctness and exercise progress.

## Acknowledgments

This project utilizes the Gymnasium library for creating the custom environment and Stable Baselines3 for implementing the RL algorithms. Special thanks to the contributors of these libraries for their invaluable work.