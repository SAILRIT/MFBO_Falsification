# Overview

This is the code release for the multi-fidelity Bayesian optimization algorithm for falsification of learning-based control systems, from “Optimizing Falsification for Learning-Based Control Systems: A Multi-Fidelity Bayesian Approach”.

# Benchmarks

## Cart-pole
We developed two higher fidelity simulators, based on the primary cart-pole offered by Gym, available at: [Gymnasium Cartpole](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py).  
We falsified the proximal policy optimization (PPO) trained for the highest fidelity simulator which is available at: [Stable Baselines PPO](https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/ppo.rst).

**Dependencies:** `numpy`, `stable_baselines3`, `gymnasium==0.28.1`, `emukit`, `GPy`

## Lunar Lander

We developed two other fidelity levels for the original lunar lander environment available at: [Gymnasium Lunar Lander](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py).  
We falsify against the deep deterministic policy gradient (DDPG) algorithm trained for the highest fidelity simulator available at: [Stable Baselines DDPG](https://github.com/Stable-Baselines-Team/stable-baselines/blob/master/docs/modules/ddpg.rst).

**Dependencies:** `numpy`, `stable_baselines3`, `gymnasium`, `emukit`, `gym[box2d]`, `GPy`

For this case study, we used Conda environment for managing dependencies. You can use these commands:
```bash
conda create -y -n gymbox
conda activate gymbox
conda install -y conda-forge::gymnasium-box2d

## Highway

We developed three levels of fidelity for the original highway driving benchmark available at: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv). Each level operates at a different frequency and has a varying number of cars. We falsify deep Q-learning (DQN) (available at [DQN Documentation](https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/dqn.rst)) for the ego vehicle's behavior, which is trained on the high-fidelity simulator.

**Dependencies:** `numpy`, `stable_baselines3`, `gymnasium`, `emukit`, `gym[box2d]`, `GPy`, `highway_env`,


For this environment, first run:
```bash
pip install highway_env
highway_env.register_highway_envs()
