
import gymnasium as gym
import sys
sys.modules["gym"] = gym
import time
import argparse
import stable_baselines3
from stable_baselines3 import PPO
import math
import emukit
import numpy as np
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch
import copy
import pprint
import highway_env

# highway_env.register_highway_envs()
import warnings
warnings.filterwarnings('ignore')
########-----------------HFS-------------------##################

envh = gym.make("merge-v0")

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "normalize": False,
        "features": ["presence", "x", "y", "vx", "vy"],
        "see_behind": True,
        "absolute": True,
        "order": "sorted"
    }
}


envh.unwrapped.config["reward_speed_range"]=[25, 30]
envh.unwrapped.config["high_speed_reward"]=0.6
envh.unwrapped.config["simulation_frequency"]= 15
# envh.configure(config)
envh.unwrapped.configure(config)
# model = DQN.load("dqn_merge_model_new")
model = DQN.load("dqn_merge_model")

# model = DQN.load("dqn_merge_model_twelve")
# model = DQN.load("dqn_merge_model_eigh")



from highway_env.vehicle.behavior import IDMVehicle
def compute_trajHf(**kwargs):
        IDMVehicle.COMFORT_ACC_MAX=4
        IDMVehicle.COMFORT_ACC_MIN=-5.5
        IDMVehicle.TIME_WANTED=1
        IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN=0.1
        IDMVehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED=2
        obs, _ = envh.reset(seed=0)
        if 'v_ego' in kwargs:
          # 'env.vehicle.speed'
            v0 = obs[0][3] - kwargs['v_ego']
            obs[0][3]=v0

        if 'v_x_ob_1' in kwargs:
            v1 = obs[1][3]+kwargs['v_x_ob_1']
            obs[1][3]=v1
        if 'v_x_ob_2' in kwargs:
            v2 = obs[2][3] + kwargs['v_x_ob_2']
            obs[2][3]= v2
        if 'v_x_ob_3' in kwargs:
            v3 = obs[3][3] + kwargs['v_x_ob_3']
            obs[3][3]=v3
        if 'v_x_ob_4' in kwargs:
            v4 = obs[4][3] + kwargs['v_x_ob_4']
            obs[4][3]= v4
        if 'x_ob_1' in kwargs:
            x1 = obs[0][1]+ kwargs['x_ob_1']
            obs[1][1] = x1
        if 'x_ob_2' in kwargs:
            x2 = obs[0][1]+ kwargs['x_ob_2']
            obs[2][1] = x2
        if 'x_ob_3' in kwargs:
            x3 = obs[0][1] + kwargs['x_ob_3']
            obs[3][1] = x3
        if 'x_ob_4' in kwargs:
            x4 = obs[0][1]+ kwargs['x_ob_4']
            obs[4][1]= x4

        ob = obs[:, 1:3].reshape(-1)
        trajHf=[ob]
        # trajHf=[]
        reward = 0
        done = truncated = False
        capture_interval =0.1
        t = 0
        while done==False:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, info = envh.step(action)
                ob = obs[:, 1:3].reshape(-1)
                reward = r + reward
                trajHf.append(ob)
                t += capture_interval
                if done or truncated:
                    break
        additional_data = {'reward':reward}
        return trajHf, additional_data

def sutH(x0):
        return compute_trajHf(v_ego=x0[0],v_x_ob_1=x0[1],v_x_ob_2=x0[2],v_x_ob_3=x0[3],v_x_ob_4=x0[4],
             x_ob_1=x0[5],x_ob_2=x0[6],x_ob_3=x0[7],x_ob_4=x0[8])
    # return compute_trajHf(v_x_ob_1=x0[0],v_x_ob_2=x0[1],v_x_ob_3=x0[2],v_x_ob_4=x0[3],
    #          x_ob_1=x0[4],x_ob_2=x0[5],x_ob_3=x0[6],x_ob_4=x0[7])

#############---------------------LFS-------------##############

envl = gym.make("merge-v0")

configl = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "normalize": False,
        "features": ["presence", "x", "y", "vx", "vy"],
        "see_behind": True,
        "absolute": True,
        "order": "sorted"
    }
}


envl.unwrapped.config["reward_speed_range"]=[25, 30]
envl.unwrapped.config["high_speed_reward"]=0.6
envl.unwrapped.config["simulation_frequency"]= 11
# envh.configure(config)
envl.unwrapped.configure(configl)

def compute_trajLf(**kwargs):
        IDMVehicle.COMFORT_ACC_MAX=3
        IDMVehicle.COMFORT_ACC_MIN=-5
        IDMVehicle.TIME_WANTED=1.5
        IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN=0.2
        IDMVehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.5
        obs, _ = envl.reset(seed=0)

        
        if 'v_ego' in kwargs:
          # 'env.vehicle.speed'
            v0 = obs[0][3] - kwargs['v_ego']
            obs[0][3]=v0

        if 'v_x_ob_1' in kwargs:
            v1 = obs[1][3]+kwargs['v_x_ob_1']
            obs[1][3]=v1
        if 'v_x_ob_2' in kwargs:
            v2 = obs[2][3] + kwargs['v_x_ob_2']
            obs[2][3]= v2
        if 'v_x_ob_3' in kwargs:
            v3 = obs[3][3] + kwargs['v_x_ob_3']
            obs[3][3]=v3
        if 'v_x_ob_4' in kwargs:
            v4 = obs[4][3] + kwargs['v_x_ob_4']
            obs[4][3]= v4
        if 'x_ob_1' in kwargs:
            x1 = obs[0][1]+ kwargs['x_ob_1']
            obs[1][1] = x1
        if 'x_ob_2' in kwargs:
            x2 = obs[0][1]+ kwargs['x_ob_2']
            obs[2][1] = x2
        if 'x_ob_3' in kwargs:
            x3 = obs[0][1] + kwargs['x_ob_3']
            obs[3][1] = x3
        if 'x_ob_4' in kwargs:
            x4 = obs[0][1]+ kwargs['x_ob_4']
            obs[4][1]= x4


        ob = obs[:, 1:3].reshape(-1)
        trajLf=[ob]
        # trajLf=[]
        reward = 0
        done = truncated = False
        capture_interval =0.1
        t = 0
        while done==False:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, info = envl.step(action)
                

                noise_x_1 = np.random.normal(6.2,0.5,1)
                noise_x_2 = np.random.normal(6.2,0.5,1)
                noise_x_3 = np.random.normal(6.2,0.5,1)
                noise_x_4 = np.random.normal(6.2,0.5,1)

                noise_v_1=np.random.normal(2,0.5,1)
                noise_v_2=np.random.normal(2,0.5,1)
                noise_v_3=np.random.normal(2,0.5,1)
                noise_v_4=np.random.normal(2,0.5,1)
                
                obs[1][1]=obs[1][1]+ noise_x_1
                obs[2][1]=obs[2][1]+ noise_x_2
                obs[3][1]=obs[3][1]+ noise_x_3
                obs[4][1]=obs[4][1]+ noise_x_4
                obs[1][3]=obs[1][3]+ noise_v_1
                obs[2][3]=obs[2][3]+ noise_v_2
                obs[3][3]=obs[3][3]+ noise_v_3
                obs[4][3]=obs[4][3]+ noise_v_4
                ob = obs[:, 1:3].reshape(-1)
                reward = r + reward
                trajLf.append(ob)
                t += capture_interval
                if done or truncated:
                    break
        additional_data = {'reward':reward}
        return trajLf, additional_data
def sutL(x0):
        return compute_trajLf(v_ego=x0[0],v_x_ob_1=x0[1],v_x_ob_2=x0[2],v_x_ob_3=x0[3],v_x_ob_4=x0[4],
             x_ob_1=x0[5],x_ob_2=x0[6],x_ob_3=x0[7],x_ob_4=x0[8])
    # return compute_trajLf(v_x_ob_1=x0[0],v_x_ob_2=x0[1],v_x_ob_3=x0[2],v_x_ob_4=x0[3],
    #          x_ob_1=x0[4],x_ob_2=x0[5],x_ob_3=x0[6],x_ob_4=x0[7])

########################---------MID FS#############################
envm = gym.make("merge-v0", render_mode='rgb_array')

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "normalize": False,
        "features": ["presence", "x", "y", "vx", "vy"],
        "see_behind": True,
        "absolute": True,
        "order": "sorted"
    }
}

envm.unwrapped.config["reward_speed_range"]=[25, 30]
envm.unwrapped.config["high_speed_reward"]=0.6
envm.unwrapped.config["simulation_frequency"]= 13
# envh.configure(config)
envm.unwrapped.configure(config)



def compute_trajMf(**kwargs):
        IDMVehicle.COMFORT_ACC_MAX=3.5
        IDMVehicle.COMFORT_ACC_MIN=-5.25
        IDMVehicle.TIME_WANTED=1.25
        IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN=0.15
        IDMVehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.75
        obs, _ = envm.reset(seed=0)

        
        if 'v_ego' in kwargs:
          # 'env.vehicle.speed'
            v0 = obs[0][3] - kwargs['v_ego']
            obs[0][3]=v0

        if 'v_x_ob_1' in kwargs:
            v1 = obs[1][3]+kwargs['v_x_ob_1']
            obs[1][3]=v1
        if 'v_x_ob_2' in kwargs:
            v2 = obs[2][3] + kwargs['v_x_ob_2']
            obs[2][3]= v2
        if 'v_x_ob_3' in kwargs:
            v3 = obs[3][3] + kwargs['v_x_ob_3']
            obs[3][3]=v3
        if 'v_x_ob_4' in kwargs:
            v4 = obs[4][3] + kwargs['v_x_ob_4']
            obs[4][3]= v4
        if 'x_ob_1' in kwargs:
            x1 = obs[0][1]+ kwargs['x_ob_1']
            obs[1][1] = x1
        if 'x_ob_2' in kwargs:
            x2 = obs[0][1]+ kwargs['x_ob_2']
            obs[2][1] = x2
        if 'x_ob_3' in kwargs:
            x3 = obs[0][1] + kwargs['x_ob_3']
            obs[3][1] = x3
        if 'x_ob_4' in kwargs:
            x4 = obs[0][1]+ kwargs['x_ob_4']
            obs[4][1]= x4
        ob = obs[:, 1:3].reshape(-1)
        trajMf=[ob]
        # trajMf=[]
        reward = 0
        done = truncated = False
        capture_interval =0.1
        t = 0
        while done==False:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, info = envm.step(action)
                noise_x_1 = np.random.normal(4,0.6,1)
                noise_x_2 = np.random.normal(4,0.6,1)
                noise_x_3 = np.random.normal(4,0.6,1)
                noise_x_4 = np.random.normal(4,0.6,1)
                noise_v_1=np.random.normal(1,0.3,1)
                noise_v_2=np.random.normal(1,0.3,1)
                noise_v_3=np.random.normal(1,0.3,1)
                noise_v_4=np.random.normal(1,0.3,1)
                obs[1][1]=obs[1][1] + noise_x_1
                obs[2][1]=obs[2][1] + noise_x_2
                obs[3][1]=obs[3][1] + noise_x_3
                obs[4][1]=obs[4][1] + noise_x_4
                obs[1][3]=obs[1][3] + noise_v_1
                obs[2][3]=obs[2][3] + noise_v_2
                obs[3][3]=obs[3][3] + noise_v_3
                obs[4][3]=obs[4][3] + noise_v_4
                ob = obs[:, 1:3].reshape(-1)
                reward = r + reward
                trajMf.append(ob)
                t += capture_interval
                if done or truncated:
                    break
        additional_data = {'reward':reward}
        return trajMf, additional_data
def sutM(x0):
        return compute_trajMf(v_ego=x0[0],v_x_ob_1=x0[1],v_x_ob_2=x0[2],v_x_ob_3=x0[3],v_x_ob_4=x0[4],
             x_ob_1=x0[5],x_ob_2=x0[6],x_ob_3=x0[7],x_ob_4=x0[8])

######## - - - - - --- - -   Utils ----- ---  - - - - ##############

def sample_from(count, bounds, sampler=None):
    if sampler is None:
        sampler = lambda num: np.random.random(num)

    sampled_01 = sampler(count*len(bounds))
    sampled_01.resize(count,len(bounds))
    sampled_01 = sampled_01.T
    sampled_lb = [sampled_01[i]*(b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]
    return np.array(sampled_lb).T



bounds = [(0,1)]
bounds.append((8,12))
bounds.append((8,12))
bounds.append((8,12))
bounds.append((8,12))
# bounds.append((30,40))
### noise on initial positions
bounds.append((10, 12))
bounds.append((10, 12))
bounds.append((10, 12))
bounds.append((10, 12))

from emukit.core import ContinuousParameter, ParameterSpace,InformationSourceParameter

bound = ParameterSpace([ContinuousParameter('v_ego', 0,1),ContinuousParameter('v_x_ob_1', 8,12),
           ContinuousParameter('v_x_ob_2',8,12),
           ContinuousParameter('v_x_ob_3', 8,12),
           ContinuousParameter('v_x_ob_4', 8,12),
           ContinuousParameter('x_ob_1',10, 12),
           ContinuousParameter('x_ob_2', 10, 12),
           ContinuousParameter('x_ob_3', 10, 12),
           ContinuousParameter('x_ob_4', 10, 12),InformationSourceParameter(3)])

import GPy
from emukit.model_wrappers import GPyModelWrapper
# Class Tree Node!
class tree_node():
    def __init__(self, children, f=None, df=None):
        self.children = children
        self.f = f
        self.df = df

    def evaluate(self, X,  **kwargs):
        self.cn_data = [child.evaluate(X, **kwargs) for child in self.children]
        return self.f(np.array(self.cn_data), axis=0)

    def eval_df(self, X, **kwargs):
        loc = self.df(np.array(self.cn_data), axis=0)
        cn_df_data = [child.eval_df(X, **kwargs) for child in self.children]
        return cn_df_data[loc]

    def init_GPs(self, X, trajs, **kwargs):
        for child in self.children:
            child.init_GPs(X, trajs, **kwargs)

    def update_GPs(self, X, trajs, **kwargs):
        for child in self.children:
            child.update_GPs(X, trajs, **kwargs)

    def eval_robustness(self, trajs):
        cn_data = [child.eval_robustness(trajs) for child in self.children]
        return self.f(np.array(cn_data), axis=0)

    def find_GP_func(self):
        cn_data = [child.find_GP_func() for child in self.children]
        return self.f(np.array(cn_data), axis=0)

# Different types of nodes!
# Max and Min Node
class max_node(tree_node):
    def __init__(self,children, f=np.amax, df=np.argmax):
        super(max_node, self).__init__(children, f,df)

class min_node(tree_node):
    def __init__(self, children, f=np.amin, df=np.argmin):
        super(min_node, self).__init__(children, f,df)

# Predicate Node
class pred_node(tree_node):
    def __init__(self, children=None, f=None):
        super(pred_node, self).__init__(children, f)
        self.Y = []

    def evaluate(self, X, **kwargs):
        X = np.atleast_2d(X)

        # If mode is True evaluate in GP mode

        if 'k' in kwargs:
            k=kwargs['k']
        else:
            k = 10

        m, v = self.GP.predict(X)
        return m - k*np.sqrt(v)

    def eval_df(self, X, **kwargs):
        X = np.atleast_2d(X)

        if 'k' in kwargs:
            k = kwargs['k']
        else:
            k = 10
        m,v = self.GP.predict(X)
        dm, dv = self.GP.predictive_gradients(X)
        dm = dm[:, :, 0]
        return dm - (k/2)*(dv/np.sqrt(v))

    def init_GPs(self, X, trajs, **kwargs):
        for traj in trajs:
            self.Y.append(self.f(traj))
        self.Y = np.array(self.Y)
        self.Y.resize(len(self.Y),1)
        if 'kernel' in kwargs:
            kernel = kwargs['kernel']
        else:
            kernel = GPy.kern.Matern32(X.shape[1])
        if 'normalizer' in kwargs:
            normalizer=kwargs['normalizer']
        else:
            normalizer=False
        self.GP = GPy.models.GPRegression(X= X, Y=self.Y,
                                          kernel=copy.deepcopy(kernel),
                                          normalizer=normalizer)

        if 'optimize_restarts' in kwargs:
            self.GP.optimize_restarts(kwargs['optimize_restarts'])
        else:
            self.GP.optimize()

    def update_GPs(self, X, trajs, **kwargs):
        ys = []

        trajs = np.atleast_2d(trajs)
        for traj in trajs:
            ys.append(self.f(traj))
        ys = np.array(ys)
        ys.resize(len(ys), 1)
        self.Y = np.vstack((self.Y, ys))

        self.GP.set_XY(X, self.Y)

        if 'optimize_restarts' in kwargs:
            self.GP.optimize_restarts(kwargs['optimize_restarts'])
        else:
            self.GP.optimize()

    def eval_robustness(self, trajs):
        # trajs = np.atleast_2d(trajs)
        Y = np.array([self.f(traj) for traj in trajs])
        return Y.reshape(len(Y), 1)


    def find_GP_func(self):
        return self.GP.Y

# Assign costs
from emukit.core.acquisition import Acquisition
# Define cost of different fidelities as acquisition function
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        x_cost = np.array([self.costs[i] for i in fidelity_index])
        return x_cost[:, None]

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)


import emukit
from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch

from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
low_fidelity_cost =6
mid_fidelity_cost = 15
high_fidelity_cost = 23

global x_array,y_array
class test_module:
    global y_array
    def __init__(self,sutl,suth,sutm, bounds, spec=None,f_tree=None,
                 normalizer=False,seed=None, **kwargs):

        self.system_under_test_L=sutl
        self.system_under_test_H=suth
        self.system_under_test_M=sutm
        self.f_tree=f_tree
        self.bounds = bounds
        self.normalizer=normalizer
        self.seed=seed


        if 'cost_model' in kwargs:
            self.cost_model = kwargs['cost_model']
        else:
            self.cost_model = lambda x: 1

        if 'init_sample' in kwargs:
            self.init_sample = kwargs['init_sample']
        else:
            self.init_sample = 2*len(bounds)


        if 'with_ns' in kwargs:
            self.with_ns = kwargs['with_ns']
        else:
            self.with_ns = False


        if 'exp_weight' in kwargs:
            self.k = kwargs['exp_weight']
        else:
            self.k = 10

        # Optimize retsrats for hyper parameter optimization for GPs
        if 'optimize_restarts' in kwargs:
            self.optimize_restarts = kwargs['optimize_restarts']
        else:
            self.optimize_restarts = 1


        if 'XL' in kwargs:
            self.XL = kwargs['XL']
        else:
            self.XL = []

        if 'XH' in kwargs:
            self.XH = kwargs['XH']
        else:
            self.XH = []

        if 'XM' in kwargs:
            self.XM = kwargs['XM']
        else:
            self.XM = []

    def initialize(self):
        global low_exp_num
        global high_exp_num
        global mid_exp_num
        global real_low_ce
        global real_high_ce
        global real_mid_ce
        global valid_low_ce
        global valid_high_ce
        global valid_mid_ce
        global real_ce
        real_ce=[]
        real_high_ce=0
        real_mid_ce=0
        real_low_ce=0
        valid_low_ce=0
        valid_high_ce=0
        valid_mid_ce=0
        global all_ce_high
        global all_ce_low
        global all_ce_mid

        global min_phi_obs
        min_phi_obs=[]
        all_ce_low=0
        all_ce_high=0
        all_ce_mid=0
        global X_ns
        global XL, XH,YL, YH ,YM,XM
        if len(self.XL) == 0:
            XL = sample_from(self.init_sample, self.bounds)
            self.XL = XL
        if len(self.XM) == 0:
            # XM=XL
            jj=self.init_sample//2
            XM = np.atleast_2d(np.random.permutation(XL)[:jj])
            self.XM = XM

        if len(self.XH) == 0:
            # XH=XL
            o=self.init_sample//3
            XH = np.atleast_2d(np.random.permutation(XM)[:o])
            self.XH = XH
        # print(XH)
        global trajsL,trajsH
        global XL_ns, XH_ns,YL, YH
        trL = []
        trH = []
        trM=[]

        for x in self.XL:

            # start_t=time.time()
            trL.append(self.system_under_test_L(x))
        # print(f"this is LF: {trL}")
            # end_t=time.time()
            # total_time = end_t - start_t
            # print("This is LF")
            # print("\n"+ str(total_time))
        self.f_acqu=self.f_tree[0]
        YL = self.f_acqu.eval_robustness(trL)

        for x in self.XH:
            # start_t=time.time()
            trH.append(self.system_under_test_H(x))
        # print(f"this is HF: {trH}")
            # end_t=time.time()
            # total_time = end_t - start_t
            # print("This is HF")
            # print("\n"+ str(total_time))
        self.f_acqu=self.f_tree[2]
        YH = self.f_acqu.eval_robustness(trH)

        for x in self.XM:
            # start_t=time.time()
            trM.append(self.system_under_test_M(x))
            # end_t=time.time()
            # total_time = end_t - start_t
            # print("This is MF")
            # print("\n"+ str(total_time))
        # print(f"this is MF: {trM}")
        self.f_acqu=self.f_tree[1]
        YM = self.f_acqu.eval_robustness(trM)

        low_exp_num=self.init_sample
        mid_exp_num=self.init_sample//2
        high_exp_num=self.init_sample//3

        trl=[]
        trh=[]
        trm=[]

        for x in self.XL:
               trl.append(self.system_under_test_L(x))
               trh.append(self.system_under_test_H(x))
        self.f_acqu=self.f_tree[0]
        f_xlow=self.f_acqu.eval_robustness(trl)
        min_phi_obs.append(f_xlow)
        all_ce_low=all_ce_low+np.sum(f_xlow< 0)
        self.f_acqu=self.f_tree[2]
        f_xhigh=self.f_acqu.eval_robustness(trh)
        for fl, fh in zip(f_xlow, f_xhigh):
    # Iterate over elements in the row
                    for i in range(len(fl)):
                      if fl[i] < 0 and fh[i] < 0:  # Check corresponding elements
                          real_low_ce = 1 + real_low_ce
                          real_ce.append(fl[i])
        # if (f_xlow<0):
        #   all_ce_low=all_ce_low+1


        # if (f_xlow<0) and (f_xhigh<0):
        #   real_low_ce=1+real_low_ce
        trjh=[]
        trjl=[]

        for x in self.XM:
               trm.append(self.system_under_test_M(x))
               trjh.append(self.system_under_test_H(x))
               trjl.append(self.system_under_test_L(x))

        self.f_acqu=self.f_tree[1]
        f_xmid=self.f_acqu.eval_robustness(trm)
        min_phi_obs.append(f_xmid)
        self.f_acqu=self.f_tree[2]
        f_xhigh=self.f_acqu.eval_robustness(trjh)
        # self.f_acqu=self.f_tree[0]
        # f_x_low=self.f_acqu.eval_robustness(trjl)
        all_ce_mid=all_ce_mid + np.sum(f_xmid< 0)

        for fm, fh in zip( f_xmid, f_xhigh):
                    for i in range(len(fm)):
                      if fm[i]<0 and fh[i]<0:
                          real_mid_ce=1 + real_mid_ce
                          real_ce.append(fm[i])

        # for fl, fm, fh in zip(f_x_low, f_xmid, f_xhigh):
        #             for i in range(len(fm)):
        #               if fl[i] > 0 and fm[i] < 0:  # Check corresponding elements
        #                   all_ce_mid=all_ce_mid + 1
        #               if fm[i]<0 and fh[i]<0 and fl[i]>0:
        #                   real_mid_ce=1+real_mid_ce


        # if (f_xmid<0) and (f_x_low>0):
        #   all_ce_mid=all_ce_mid + 1

        # if (f_xmid<0) and (f_x_low>0) and (f_xhigh<0):
        #   real_mid_ce=1 + real_mid_ce
        th=[]
        tm=[]
        tl=[]
        for x in self.XH:
               th.append(self.system_under_test_H(x))
               tl.append(self.system_under_test_L(x))
               tm.append(self.system_under_test_M(x))



        self.f_acqu=self.f_tree[2]
        f_x_high=self.f_acqu.eval_robustness(th)
        min_phi_obs.append(f_x_high)
        # self.f_acqu=self.f_tree[0]
        # f_x_low=self.f_acqu.eval_robustness(tl)
        # self.f_acqu=self.f_tree[1]
        # f_x_m=self.f_acqu.eval_robustness(tm)
        all_ce_high=all_ce_high + np.sum(f_x_high< 0)
        real_high_ce = real_high_ce + np.sum(f_x_high< 0)
        for i in range(len(f_x_high)):
            if f_x_high[i]<0:
                real_ce.append(f_x_high[i])
        
        # for fl, fm, fh in zip(f_x_low, f_x_m, f_x_high):
        #     for i in range(len(fh)):
        #         if fl[i]>0 and fm[i]>0 and fh[i]<0:
        #             all_ce_high=all_ce_high + 1
        #             real_high_ce=1+real_high_ce

        # if (f_x_low>0) and (f_x_m>0) and (f_x_high<0):
        #   real_high_ce=1 + real_high_ce
        #   all_ce_high=all_ce_high + 1


        global x_array,y_array
        x_array, y_array = convert_xy_lists_to_arrays([XL,XM, XH], [YL,YM, YH])

    global XL, XH,YL, YH,YM,XM
    global y_array
    global real_ce
    global flag_n
    global cm
    flag_n=0
    cm=0   


    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            global XL, XH,YL, YH,XM,YM
            global low_exp_num
            global high_exp_num
            global mid_exp_num
            global all_ce_high
            global all_ce_low
            global all_ce_mid
            global real_low_ce
            global real_high_ce
            global real_mid_ce
            global valid_low_ce
            global valid_high_ce
            global valid_mid_ce
            global real_ce
            global flag_n
            global cm
            print('BO iteration:', ib)
            global x_array,y_array
            kern_low = GPy.kern.RBF(len(bounds),ARD=True)
            kern_mid = GPy.kern.RBF(len(bounds),ARD=True)
            kern_err = GPy.kern.RBF(len(bounds),ARD=True)
            
            # kern_low.lengthscale.constrain_bounded(0.01, 0.5)
            # kern_mid.lengthscale.constrain_bounded(0.01, 0.5)
            # kern_err.lengthscale.constrain_bounded(0.01, 0.5)
            
            multi_fidelity_kernel = LinearMultiFidelityKernel([kern_mid,kern_low, kern_err])
            gpy_model = GPyLinearMultiFidelityModel(x_array, y_array, multi_fidelity_kernel, 3,None)
            gpy_model.mixed_noise.Gaussian_noise.fix(0.01)
            gpy_model.mixed_noise.Gaussian_noise_1.fix(0.01)
            gpy_model.mixed_noise.Gaussian_noise_2.fix(0.01)
            GPmodel = GPyMultiOutputWrapper(gpy_model, 3, 1, verbose_optimization=True)
            GPmodel.optimize()
            cost_acquisition = Cost([low_fidelity_cost,mid_fidelity_cost, high_fidelity_cost])
            acquisition = MultiInformationSourceEntropySearch(GPmodel, bound) / cost_acquisition
            acquisition_optimizer=MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(bound), bound)
            new_x,val_acq=acquisition_optimizer.optimize(acquisition)
            # print(new_x)
            TL=[]
            TH=[]
            TM=[]
            THH=[]
            THHH=[]

            if new_x[0][-1]==0.:
               print("This is low-fidelity")
               x=new_x[0][0:9]
               X_L=XL
               XL=np.vstack((X_L, x))
               low_exp_num=1+low_exp_num
               TL.append(self.system_under_test_L(x))
               self.f_acqu=self.f_tree[0]
               f_xl=self.f_acqu.eval_robustness(TL)
               min_phi_obs.append(f_xl)
               if (flag_n==0):
                       cm= cm + 6
               if f_xl<0:
                 print('ce detected!')
                 all_ce_low=all_ce_low+1
               self.f_acqu=self.f_tree[2]
               TH.append(self.system_under_test_H(x))
               f_test_ce=self.f_acqu.eval_robustness(TH)
               if (f_xl<0) and (f_test_ce<0):
                 print("Wow, it's real!")
                 real_ce.append(f_xl)
                 flag_n=1
                #  print("It's a valid counterexample found by low fidelity simulator")
                 valid_low_ce=1 + valid_low_ce

               #print(f"f_xl= {f_xl}")
               Y_L=YL
               YL=np.vstack((Y_L, f_xl))
               x_array, y_array = convert_xy_lists_to_arrays([XL,XM, XH], [YL,YM, YH])
            elif new_x[0][-1]==1.:
               print("This is mid-fidelity")
               xkk=new_x[0][0:9]
               X_M=XM
               XM=np.vstack((X_M, xkk))
               mid_exp_num=1 + mid_exp_num
               TM.append(self.system_under_test_M(xkk))
               self.f_acqu=self.f_tree[1]
               f_xm=self.f_acqu.eval_robustness(TM)
               min_phi_obs.append(f_xm)
               if (flag_n==0):
                       cm= cm + 15
               if f_xm<0:
                 all_ce_mid=all_ce_mid + 1
                 print('ce found!')
               self.f_acqu=self.f_tree[2]
               THH.append(self.system_under_test_H(xkk))
               f_test_ce=self.f_acqu.eval_robustness(THH)
               if (f_xm<0) and (f_test_ce<0):
                 print("It's a real ce! found by mid fidelity simulator")
                 valid_mid_ce=1+valid_mid_ce
                 flag_n=1
                 real_ce.append(f_xm)

               #print(f"f_xl= {f_xl}")
               Y_M=YM
               YM=np.vstack((Y_M, f_xm))
               x_array, y_array = convert_xy_lists_to_arrays([XL,XM, XH], [YL,YM, YH])
            else:
               print("This is high-fidelity")
               a=new_x[0][0:9]
               X_H=XH
               XH=np.vstack((X_H, a))
               high_exp_num =1 + high_exp_num
               THHH.append(self.system_under_test_H(a))
               self.f_acqu=self.f_tree[2]
               f_xh=self.f_acqu.eval_robustness(THHH)
               min_phi_obs.append(f_xh)
               if (flag_n==0):
                       cm= cm + 23
               if f_xh<0:
                  print("It's a ce!")
                  flag_n=1
                  valid_high_ce=1 + valid_high_ce
                  all_ce_high=all_ce_high + 1
                  real_ce.append(f_xh)
               #print(f"f_xh= {f_xh}")
               Y_H=YH
               YH=np.vstack((Y_H, f_xh))
               x_array, y_array = convert_xy_lists_to_arrays([XL,XM, XH], [YL,YM, YH])
        global sum_real_h_ce
        sum_real_h_ce=0
        global sum_real_m_ce
        sum_real_m_ce=0
        global sum_real_l_ce
        sum_real_l_ce=0
        global sum_real_ce
        global MF_c
        global min_val
        global new_mf_c
        new_mf_c=0
        # MF_c=0
        # sum_real_ce=0
        sum_real_h_ce=(valid_high_ce)+(real_high_ce)
        sum_real_m_ce=(valid_mid_ce)+(real_mid_ce)
        sum_real_l_ce=(valid_low_ce)+(real_low_ce)
        sum_real_ce=(sum_real_l_ce)+(sum_real_m_ce)+(sum_real_h_ce)
        global all_ce

        all_ce=all_ce_high + all_ce_low + all_ce_mid
        # print(f" number of real conuter examples is: {sum_real_ce}")
        MF_c=23*(high_exp_num)+15*(mid_exp_num)+6*(low_exp_num)
        new_mf_c=MF_c + 23*(all_ce_mid+all_ce_low)
        # print(f" the cost is {MF_c}")
        min_val = y_array.min()

# Safety specification in paper:
# 1. Either the car remains within the initial condition of state and velocity
# 2. Reaches the goal asap
from numpy import mean
import warnings
warnings.filterwarnings('ignore')
mf_new_cost=[]
min_phi=[]
MFBO_cost=[]
real_num_ce=[]
all_ce_3f=[]
############### specifications for lf

def pred1(trajLf):
    # print(f"this is trajHf in pred1: {trajHf}")
    trajLf= trajLf[0]
    # print(traj)
    x_ego=np.array(trajLf).T[0]
    y_ego=np.array(trajLf).T[1]
    x_ob1=np.array(trajLf).T[2]
    x_ob2=np.array(trajLf).T[4]
    x_ob3=np.array(trajLf).T[6]
    x_ob4=np.array(trajLf).T[8]
    y_ob1=np.array(trajLf).T[3]
    y_ob2=np.array(trajLf).T[5]
    y_ob3=np.array(trajLf).T[7]
    y_ob4=np.array(trajLf).T[9]
    x_f=[]

    for i in range(len(y_ego)):
        x_goal=[]
        if math.floor(y_ego[i])==0:
            y_ego[i]=0
        if math.floor(y_ego[i])==4 or math.floor(y_ego[i])==3:
            y_ego[i]=4
        if math.floor(y_ego[i])==8 or math.floor(y_ego[i])==7:
            y_ego[i]=8
        if math.floor(y_ego[i])==-5 or math.floor(y_ego[i])==-4:
            y_ego[i]=-4
        if math.floor(y_ego[i])==11 or math.floor(y_ego[i])==12:
            y_ego[i]=12

        if math.floor(y_ob1[i])==0:
            y_ob1[i]=0
        if math.floor(y_ob1[i])==4 or math.floor(y_ob1[i])==3:
            y_ob1[i]=4
        if math.floor(y_ob1[i])==8 or math.floor(y_ob1[i])==7:
            y_ob1[i]=8
        if math.floor(y_ob1[i])==-5 or math.floor(y_ob1[i])==-4:
            y_ob1[i]=-4
        if math.floor(y_ob1[i])==11 or math.floor(y_ob1[i])==12:
            y_ob1[i]=12

        if math.floor(y_ob2[i])==0:
            y_ob2[i]=0
        if math.floor(y_ob2[i])==4 or math.floor(y_ob2[i])==3:
            y_ob2[i]=4
        if math.floor(y_ob2[i])==8 or math.floor(y_ob2[i])==7:
            y_ob2[i]=8
        if math.floor(y_ob2[i])==-5 or math.floor(y_ob2[i])==-4:
            y_ob2[i]=-4
        if math.floor(y_ob2[i])==11 or math.floor(y_ob2[i])==12:
            y_ob2[i]=12

        if math.floor(y_ob3[i])==0:
            y_ob3[i]=0
        if math.floor(y_ob3[i])==4 or math.floor(y_ob3[i])==3:
            y_ob3[i]=4
        if math.floor(y_ob3[i])==8 or math.floor(y_ob3[i])==7:
            y_ob3[i]=8
        if math.floor(y_ob3[i])==-5 or math.floor(y_ob3[i])==-4:
            y_ob3[i]=-4
        if math.floor(y_ob3[i])==11 or math.floor(y_ob3[i])==12:
            y_ob3[i]=12

        if math.floor(y_ob4[i])==0:
            y_ob4[i]=0
        if math.floor(y_ob4[i])==4 or math.floor(y_ob4[i])==3:
            y_ob4[i]=4
        if math.floor(y_ob4[i])==8 or math.floor(y_ob4[i])==7:
            y_ob4[i]=8
        if math.floor(y_ob4[i])==-5 or math.floor(y_ob4[i])==-4:
            y_ob4[i]=-4
        if math.floor(y_ob4[i])==11 or math.floor(y_ob4[i])==12:
            y_ob4[i]=12

        if y_ego[i]==y_ob1[i]:
            dist1=x_ob1[i]-x_ego[i]
            if dist1<= -10.4:
                x_goal.append(abs(dist1) -10.4)
            elif(dist1> 10.4):
                x_goal.append(dist1 -10.4)
            elif(0 <dist1 <=10.4):
                x_goal.append(dist1 -10.4)
            else:
                x_goal.append(abs(dist1) - 10.4)

        if y_ego[i]==y_ob2[i]:
            dist2=x_ob2[i]-x_ego[i]
            if dist2 <= -10.4:
                x_goal.append(abs(dist2) -10.4)
            elif(dist2 > 10.4):
                x_goal.append(dist2 -10.4)
            elif(0 < dist2 <=10.4):
                x_goal.append(dist2 - 10.4)
            else:
                x_goal.append(abs(dist2) - 10.4)

        if y_ego[i]==y_ob3[i]:
            dist3=x_ob3[i]-x_ego[i]
            if dist3 <= -10.4:
                x_goal.append(abs(dist3) - 10.4)
            elif(dist3 >10.4):
                x_goal.append(dist3 -10.4)
            elif(0 < dist3 <=10.4):
                x_goal.append(dist3 - 10.4)
            else:
                x_goal.append(abs(dist3) - 10.4)

        if y_ego[i]==y_ob4[i]:
            dist4=x_ob4[i]-x_ego[i]
            if dist4<= -10.4:
                x_goal.append(abs(dist4) - 10.4)
            elif(dist4>10.4):
                x_goal.append(dist4 - 10.4)
            elif(0 <dist4 <=10.4):
                x_goal.append(dist4 - 10.4)
            else:
                x_goal.append(abs(dist4) - 10.4)
        if x_goal!=[]:
            x_f.append(np.min(x_goal))
        else:
            x_f.append(20)

    return np.min(x_f)

#####################Specifications for MF

def pred2(trajMf):
    # print(f"this is trajHf in pred1: {trajHf}")
    trajMf= trajMf[0]
    # print(traj)
    x_ego=np.array(trajMf).T[0]
    y_ego=np.array(trajMf).T[1]
    x_ob1=np.array(trajMf).T[2]
    x_ob2=np.array(trajMf).T[4]
    x_ob3=np.array(trajMf).T[6]
    x_ob4=np.array(trajMf).T[8]
    y_ob1=np.array(trajMf).T[3]
    y_ob2=np.array(trajMf).T[5]
    y_ob3=np.array(trajMf).T[7]
    y_ob4=np.array(trajMf).T[9]
    x_f=[]

    for i in range(len(y_ego)):
        x_goal=[]
        if math.floor(y_ego[i])==0:
            y_ego[i]=0
        if math.floor(y_ego[i])==4 or math.floor(y_ego[i])==3:
            y_ego[i]=4
        if math.floor(y_ego[i])==8 or math.floor(y_ego[i])==7:
            y_ego[i]=8
        if math.floor(y_ego[i])==-5 or math.floor(y_ego[i])==-4:
            y_ego[i]=-4
        if math.floor(y_ego[i])==11 or math.floor(y_ego[i])==12:
            y_ego[i]=12

        if math.floor(y_ob1[i])==0:
            y_ob1[i]=0
        if math.floor(y_ob1[i])==4 or math.floor(y_ob1[i])==3:
            y_ob1[i]=4
        if math.floor(y_ob1[i])==8 or math.floor(y_ob1[i])==7:
            y_ob1[i]=8
        if math.floor(y_ob1[i])==-5 or math.floor(y_ob1[i])==-4:
            y_ob1[i]=-4
        if math.floor(y_ob1[i])==11 or math.floor(y_ob1[i])==12:
            y_ob1[i]=12

        if math.floor(y_ob2[i])==0:
            y_ob2[i]=0
        if math.floor(y_ob2[i])==4 or math.floor(y_ob2[i])==3:
            y_ob2[i]=4
        if math.floor(y_ob2[i])==8 or math.floor(y_ob2[i])==7:
            y_ob2[i]=8
        if math.floor(y_ob2[i])==-5 or math.floor(y_ob2[i])==-4:
            y_ob2[i]=-4
        if math.floor(y_ob2[i])==11 or math.floor(y_ob2[i])==12:
            y_ob2[i]=12

        if math.floor(y_ob3[i])==0:
            y_ob3[i]=0
        if math.floor(y_ob3[i])==4 or math.floor(y_ob3[i])==3:
            y_ob3[i]=4
        if math.floor(y_ob3[i])==8 or math.floor(y_ob3[i])==7:
            y_ob3[i]=8
        if math.floor(y_ob3[i])==-5 or math.floor(y_ob3[i])==-4:
            y_ob3[i]=-4
        if math.floor(y_ob3[i])==11 or math.floor(y_ob3[i])==12:
            y_ob3[i]=12

        if math.floor(y_ob4[i])==0:
            y_ob4[i]=0
        if math.floor(y_ob4[i])==4 or math.floor(y_ob4[i])==3:
            y_ob4[i]=4
        if math.floor(y_ob4[i])==8 or math.floor(y_ob4[i])==7:
            y_ob4[i]=8
        if math.floor(y_ob4[i])==-5 or math.floor(y_ob4[i])==-4:
            y_ob4[i]=-4
        if math.floor(y_ob4[i])==11 or math.floor(y_ob4[i])==12:
            y_ob4[i]=12
            
        if y_ego[i]==y_ob1[i]:
            dist1=x_ob1[i]-x_ego[i]
            if dist1<= -10.4:
                x_goal.append(abs(dist1) -10.4)
            elif(dist1> 10.4):
                x_goal.append(dist1 -10.4)
            elif(0 <dist1 <=10.4):
                x_goal.append(dist1 -10.4)
            else:
                x_goal.append(abs(dist1) - 10.4)

        if y_ego[i]==y_ob2[i]:
            dist2=x_ob2[i]-x_ego[i]
            if dist2 <= -10.4:
                x_goal.append(abs(dist2) -10.4)
            elif(dist2 > 10.4):
                x_goal.append(dist2 -10.4)
            elif(0 < dist2 <=10.4):
                x_goal.append(dist2 - 10.4)
            else:
                x_goal.append(abs(dist2) - 10.4)

        if y_ego[i]==y_ob3[i]:
            dist3=x_ob3[i]-x_ego[i]
            if dist3 <= -10.4:
                x_goal.append(abs(dist3) - 10.4)
            elif(dist3 >10.4):
                x_goal.append(dist3 -10.4)
            elif(0 < dist3 <=10.4):
                x_goal.append(dist3 - 10.4)
            else:
                x_goal.append(abs(dist3) - 10.4)

        if y_ego[i]==y_ob4[i]:
            dist4=x_ob4[i]-x_ego[i]
            if dist4<= -10.4:
                x_goal.append(abs(dist4) - 10.4)
            elif(dist4>10.4):
                x_goal.append(dist4 - 10.4)
            elif(0 <dist4 <=10.4):
                x_goal.append(dist4 - 10.4)
            else:
                x_goal.append(abs(dist4) - 10.4)
        if x_goal!=[]:
            x_f.append(np.min(x_goal))
        else:
            x_f.append(20)

    return np.min(x_f)


# ###################Specifications for hf

def pred3(trajHf):
    # print(f"this is trajHf in pred1: {trajHf}")
    trajHf= trajHf[0]
    # print(traj)
    x_ego=np.array(trajHf).T[0]
    y_ego=np.array(trajHf).T[1]
    x_ob1=np.array(trajHf).T[2]
    x_ob2=np.array(trajHf).T[4]
    x_ob3=np.array(trajHf).T[6]
    x_ob4=np.array(trajHf).T[8]
    y_ob1=np.array(trajHf).T[3]
    y_ob2=np.array(trajHf).T[5]
    y_ob3=np.array(trajHf).T[7]
    y_ob4=np.array(trajHf).T[9]
    x_f=[]

    for i in range(len(y_ego)):
        x_goal=[]
        if math.floor(y_ego[i])==0:
            y_ego[i]=0
        if math.floor(y_ego[i])==4 or math.floor(y_ego[i])==3:
            y_ego[i]=4
        if math.floor(y_ego[i])==8 or math.floor(y_ego[i])==7:
            y_ego[i]=8
        if math.floor(y_ego[i])==-5 or math.floor(y_ego[i])==-4:
            y_ego[i]=-4
        if math.floor(y_ego[i])==11 or math.floor(y_ego[i])==12:
            y_ego[i]=12

        if math.floor(y_ob1[i])==0:
            y_ob1[i]=0
        if math.floor(y_ob1[i])==4 or math.floor(y_ob1[i])==3:
            y_ob1[i]=4
        if math.floor(y_ob1[i])==8 or math.floor(y_ob1[i])==7:
            y_ob1[i]=8
        if math.floor(y_ob1[i])==-5 or math.floor(y_ob1[i])==-4:
            y_ob1[i]=-4
        if math.floor(y_ob1[i])==11 or math.floor(y_ob1[i])==12:
            y_ob1[i]=12

        if math.floor(y_ob2[i])==0:
            y_ob2[i]=0
        if math.floor(y_ob2[i])==4 or math.floor(y_ob2[i])==3:
            y_ob2[i]=4
        if math.floor(y_ob2[i])==8 or math.floor(y_ob2[i])==7:
            y_ob2[i]=8
        if math.floor(y_ob2[i])==-5 or math.floor(y_ob2[i])==-4:
            y_ob2[i]=-4
        if math.floor(y_ob2[i])==11 or math.floor(y_ob2[i])==12:
            y_ob2[i]=12

        if math.floor(y_ob3[i])==0:
            y_ob3[i]=0
        if math.floor(y_ob3[i])==4 or math.floor(y_ob3[i])==3:
            y_ob3[i]=4
        if math.floor(y_ob3[i])==8 or math.floor(y_ob3[i])==7:
            y_ob3[i]=8
        if math.floor(y_ob3[i])==-5 or math.floor(y_ob3[i])==-4:
            y_ob3[i]=-4
        if math.floor(y_ob3[i])==11 or math.floor(y_ob3[i])==12:
            y_ob3[i]=12

        if math.floor(y_ob4[i])==0:
            y_ob4[i]=0
        if math.floor(y_ob4[i])==4 or math.floor(y_ob4[i])==3:
            y_ob4[i]=4
        if math.floor(y_ob4[i])==8 or math.floor(y_ob4[i])==7:
            y_ob4[i]=8
        if math.floor(y_ob4[i])==-5 or math.floor(y_ob4[i])==-4:
            y_ob4[i]=-4
        if math.floor(y_ob4[i])==11 or math.floor(y_ob4[i])==12:
            y_ob4[i]=12

        if y_ego[i]==y_ob1[i]:
            dist1=x_ob1[i]-x_ego[i]
            if dist1<= -10.4:
                x_goal.append(abs(dist1) -10.4)
            elif(dist1> 10.4):
                x_goal.append(dist1 -10.4)
            elif(0 <dist1 <=10.4):
                x_goal.append(dist1 -10.4)
            else:
                x_goal.append(abs(dist1) - 10.4)

        if y_ego[i]==y_ob2[i]:
            dist2=x_ob2[i]-x_ego[i]
            if dist2 <= -10.4:
                x_goal.append(abs(dist2) -10.4)
            elif(dist2 > 10.4):
                x_goal.append(dist2 -10.4)
            elif(0 < dist2 <=10.4):
                x_goal.append(dist2 - 10.4)
            else:
                x_goal.append(abs(dist2) - 10.4)

        if y_ego[i]==y_ob3[i]:
            dist3=x_ob3[i]-x_ego[i]
            if dist3 <= -10.4:
                x_goal.append(abs(dist3) - 10.4)
            elif(dist3 >10.4):
                x_goal.append(dist3 -10.4)
            elif(0 < dist3 <=10.4):
                x_goal.append(dist3 - 10.4)
            else:
                x_goal.append(abs(dist3) - 10.4)

        if y_ego[i]==y_ob4[i]:
            dist4=x_ob4[i]-x_ego[i]
            if dist4<= -10.4:
                x_goal.append(abs(dist4) - 10.4)
            elif(dist4>10.4):
                x_goal.append(dist4 - 10.4)
            elif(0 <dist4 <=10.4):
                x_goal.append(dist4 - 10.4)
            else:
                x_goal.append(abs(dist4) - 10.4)
        if x_goal!=[]:
            x_f.append(np.min(x_goal))
        else:
            x_f.append(20)

    return np.min(x_f)


########## Non_smooth method ############
rand_num = list(range(1, 750))

#########################################
parser = argparse.ArgumentParser(description='Takes and integer as random seed and runs the code')
parser.add_argument('-r', metavar='N', type=int, help='Index to pick from the rand_num')

args = parser.parse_args()
print("Number of elements in the random seed list %d" % len(rand_num) )
print("The index from random seed list : %d" % args.r)
print("Value picked: %d" % rand_num[args.r])
rand_num2=[rand_num[args.r]]


for r in rand_num2:
      np.random.seed(r)
      node4_lf = pred_node(f=pred1)
      node4_mf = pred_node(f=pred2)
      node4_hf = pred_node(f=pred3)
      node=[node4_lf,node4_mf ,node4_hf]
      TM_ns = test_module(bounds=bounds,suth=lambda x0: sutH(x0),sutm=lambda x0: sutM(x0) ,sutl=lambda x0: sutL(x0),
                          f_tree = node,init_sample =33, optimize_restarts=5,with_ns=True, exp_weight=5, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(140)
      MFBO_cost.append(MF_c)
      all_ce_3f.append(all_ce)
      real_num_ce.append(sum_real_ce)
      min_phi.append(min_val)
      mf_new_cost.append(new_mf_c)
      
      print(f"this is number of hf runs: {high_exp_num}")
      print(f"robustness value of validated ces: {real_ce}")
      print(f"this is number of lf runs: {low_exp_num}")
      print(f"this is number of mf runs: {mid_exp_num}")
      print(f"number of ces on HF: {sum_real_h_ce}")
      print(f"number of ces on MF: {sum_real_m_ce}")
      print(f"number of ces on LF: {sum_real_l_ce}")
      print(f"this all counterexamples: {all_ce_3f}")
   
      print(f" number of valid counterexamples is : {real_num_ce}")
      print(f"cost is {MFBO_cost}")
      print(f"all validation cost: {mf_new_cost}")
      print(f"min of phi after * BO iterations: {min_phi}")
      print(f" this is minvalue of optimization: {min_phi_obs}")
      print(f"cost of first ce: {cm}")
      print("goodluck")
