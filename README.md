# Overview

This is the code release for the multi-fidelity Bayesian optimization algorithm for falsification of learning-based control systems, from “Optimizing Falsification for Learning-Based Control Systems: A Multi-Fidelity Bayesian Approach”.

# Benchmarks

## Cart-pole
We developed two higher fidelity simulators, based on the primary cart-pole offered by Gym, available at: [Gymnasium Cartpole](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py).  
We falsified the proximal policy optimization (PPO) trained for the highest fidelity simulator which is available at: [Stable Baselines PPO](https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/ppo.rst).

**Dependencies:** `numpy`, `stable_baselines3`, `gymnasium==0.28.1`, `emukit`, `GPy`
