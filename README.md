# Overview

This is the code release for the multi-fidelity Bayesian optimization algorithm for falsification of learning-based control systems, from “Optimizing Falsification for Learning-Based Control Systems: A Multi-Fidelity Bayesian Approach”.

## Benchmarks

### Cart-pole
We developed two higher fidelity simulators, based on the primary cart-pole offered by Gym, available at: [Gymnasium Cartpole](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py).  
We falsified the proximal policy optimization (PPO) trained for the highest fidelity simulator which is available at: [Stable Baselines PPO](https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/ppo.rst).

**Dependencies:** `numpy`, `stable_baselines3`, `gymnasium==0.28.1`, `emukit`, `GPy`

### Lunar Lander

We developed two other fidelity levels for the original lunar lander environment available at: [Gymnasium Lunar Lander](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py).  
We falsify against the deep deterministic policy gradient (DDPG) algorithm trained for the highest fidelity simulator available at: [Stable Baselines DDPG](https://github.com/Stable-Baselines-Team/stable-baselines/blob/master/docs/modules/ddpg.rst).

**Dependencies:** `numpy`, `stable_baselines3`, `gymnasium`, `emukit`, `gym[box2d]`, `GPy`

You can use these commands:
```
conda create -y -n gymbox;
conda activate gymbox;
conda install -y conda-forge::gymnasium-box2d;
```

### Highway, Merge, and Roundabout scenarios

We developed three levels of fidelity for the original highway, merge, and roundabout driving scenarios available at: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv). FOr example, in case of highway, each simulator operates at a different frequency and has a varying number of cars. We falsify deep Q-learning (DQN) (available at [DQN Documentation](https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/dqn.rst)) for the ego vehicle's behavior, which is trained on the high-fidelity simulator.

**Dependencies:** `numpy`, `stable_baselines3`, `gymnasium`, `emukit`, `gym[box2d]`, `GPy`, `highway_env`,

For this environment, first run:
```
pip install highway_env
highway_env.register_highway_envs()
```
## Running Multi-fidelity Bayesian Optimization
For implementing multi-fidelity BO, for all three case studies, first install emukit:
```
pip install emukit
```
You also need to install stable-baselines3[extra]:
```
pip install stable-baselines3[extra]
```

## Citing Us:
The final version of the paper is available at: [MFBO_for_Falsification](https://arxiv.org/pdf/2409.08097).

```
@article{shahrooei2024optimizing,
  title={Optimizing Falsification for Learning-Based Control Systems: A Multi-Fidelity Bayesian Approach},
  author={Shahrooei, Zahra and Kochenderfer, Mykel J and Baheri, Ali},
  journal={arXiv preprint arXiv:2409.08097},
  year={2024}
}
```
