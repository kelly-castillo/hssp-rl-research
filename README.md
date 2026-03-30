# HSSP Reinforcement Learning Research

Reinforcement learning experiments built during the Hokie Summer Scholars Program at Virginia Tech.

These projects use [CleanRL](https://github.com/vwxyzjn/cleanrl) as a foundation — a high-quality single-file RL library. CleanRL is not my work; credit goes to Huang et al. (2022). See their [paper](http://jmlr.org/papers/v23/21-1342.html) and [repo](https://github.com/vwxyzjn/cleanrl).

## Projects

- **curriculum_wrapper.py** - Curriculum learning wrapper for RL environments
- **hyperparameter_sandbox_demo.py** - Interactive hyperparameter exploration
- **multiagent_ppo.py** - Multi-agent PPO implementation
- **ppo_safe.py** - Safe reinforcement learning with PPO
- **rl_exploration.py** - Exploration strategy experiments
- **predator_prey_demo.gif** - Visualization of predator-prey environment

## Setup

Clone the repo: 
```bash
git clone https://github.com/kelly-castillo/hssp-rl-research.git
cd hssp-rl-research
```

Create and activate a conda env:
```bash
conda create -yn hssp-rl python=3.10
conda activate hssp-rl
```

Install dependencies:
```bash
pip install cleanrl gymnasium torch streamlit matplotlib numpy pandas
pip install safety-gymnasium # for ppo_safe.py
pip install ipywidgets # for ppo_safe.py Jupyter dashboard
```

## Running the Streamlit Apps
```bash
cd cleanrl
streamlit run cleanrl/hyperparameter_sandbox_demo.py
streamlit run cleanrl/curriculum_wrapper.py
streamlit run cleanrl/rl_exploration.py
streamlit run cleanrl/multiagent_ppo.py
```

## Running the Safety RL Demo

Open `ppo_safe.py` in Jupyter Notebook to use the interactive dashboard, or run it from the command line:
```bash
python ppo_safe.py
```
