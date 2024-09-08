
#!/usr/bin/env python3
import gymnasium as gym
# import sys
# sys.modules["gym"] = gym
import time
import argparse
import stable_baselines3
from stable_baselines3 import PPO
import emukit
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
# from stable_baselines3.common.vec_env import DummyVecEnv
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch


from emukit.core import ContinuousParameter, ParameterSpace,InformationSourceParameter
import numpy as np
from gym import spaces
import math
import warnings
warnings.filterwarnings('ignore')
import pprint
import highway_env
# import highway_env

# highway_env.register_highway_envs()

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
envh.unwrapped.configure(config)

# model = DQN.load("dqn_merge_model_150_k")

# model = DQN.load("dqn_merge_model_new")
model = DQN.load("dqn_merge_model")

from highway_env.vehicle.kinematics import Vehicle

from highway_env.vehicle.behavior import IDMVehicle
def compute_trajHf(**kwargs):
        # obs, _ = envh.reset(seed=0)
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
        reward = 0
        done = truncated = False
        capture_interval =1
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
#########----------------------------LF---------------------------###################
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
# # Longitudinal policy parameters
# ACC_MAX = 6.0  # [m/s2]
# """Maximum acceleration."""

# COMFORT_ACC_MAX = 3.0  # [m/s2]
# """Desired maximum acceleration."""

# COMFORT_ACC_MIN = -5.0  # [m/s2]
# """Desired maximum deceleration."""

# DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
# """Desired jam distance to the front vehicle."""

# TIME_WANTED = 1.5  # [s]
# """Desired time gap to the front vehicle."""
    # Lateral policy parameters
# LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
# LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]

def compute_trajLf(**kwargs):
        IDMVehicle.COMFORT_ACC_MAX=3
        IDMVehicle.COMFORT_ACC_MIN=-5
        IDMVehicle.TIME_WANTED=1.5
        IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN=0.2
        IDMVehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.5
        obs, _ = envl.reset(seed=0)

        # if 'max_s' and 'min_s' in kwargs:
        #     Vehicle.MAX_SPEED =kwargs['max_s']
        #     Vehicle.MIN_SPEED =kwargs['min_s']

        # if 'lc' in kwargs:
        #     Vehicle.MAX_LENGTH =kwargs['lc']

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
        reward = 0
        done = truncated = False
        capture_interval =0.1
        t = 0
        while done==False:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, info = envl.step(action)
                # obs, r, done, truncated, info = envl.step(action)

                noise_x_1 = np.random.normal(6.2,0.5,1)
                noise_x_2 = np.random.normal(6.2,0.5,1)
                noise_x_3 = np.random.normal(6.2,0.5,1)
                noise_x_4 = np.random.normal(6.2,0.5,1)

                noise_v_1=np.random.normal(2,0.5,1)
                noise_v_2=np.random.normal(2,0.5,1)
                noise_v_3=np.random.normal(2,0.5,1)
                noise_v_4=np.random.normal(2,0.5,1)
                # noise_v_0=np.random.normal(1,0.5,1)
                # obs[0][3]=obs[0][3]+ noise_v_0
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
           ContinuousParameter('x_ob_4', 10, 12),InformationSourceParameter(2)])

#Function tree ##
import numpy as np
import copy
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
low_fidelity_cost = 6
high_fidelity_cost =23

global x_array,y_array
class test_module:
    global y_array
    def __init__(self,sutl,suth, bounds, spec=None,f_tree=None,
                 normalizer=False,seed=None, **kwargs):

        self.system_under_test_L=sutl
        self.system_under_test_H=suth
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

    def initialize(self):
        global low_exp_num
        global high_exp_num
        global real_low_ce
        global real_high_ce
        global valid_low_ce
        global valid_high_ce
        real_high_ce=0
        global all_ce_high
        global all_ce_low
        global min_phi_obs
        min_phi_obs=[]
        all_ce_low=0
        all_ce_high=0
        global real_ce
        real_ce=[]

        ttL=[]
        ttH=[]
        real_low_ce=0
        valid_low_ce=0
        valid_high_ce=0
        global X_ns
        global XL, XH,YL, YH
        if len(self.XL) == 0:
            XL = sample_from(self.init_sample, self.bounds)
            self.XL = XL

        if len(self.XH) == 0:
            # XH=XL
            o=self.init_sample//3
            XH = np.atleast_2d(np.random.permutation(XL)[:o])
            self.XH = XH
        # print(XH)
        global trajsL,trajsH
        global XL_ns, XH_ns,YL, YH
        trajsL = []
        trajsH = []
        # print("This is L/F time")
        for x in self.XL:
            # start_t=time.time()
            trajsL.append(self.system_under_test_L(x))

        self.f_acqu=self.f_tree[0]
        YL = self.f_acqu.eval_robustness(trajsL)
        # print("This is HF time")
        for x in self.XH:
            # start_t=time.time()
            trajsH.append(self.system_under_test_H(x))

        self.f_acqu=self.f_tree[1]
        YH = self.f_acqu.eval_robustness(trajsH)
        low_exp_num=self.init_sample
        high_exp_num=self.init_sample//3
        trl=[]
        trh=[]
        for x in self.XL:
               trl.append(self.system_under_test_L(x))
               trh.append(self.system_under_test_H(x))

        self.f_acqu=self.f_tree[0]
        f_xlow=self.f_acqu.eval_robustness(trl)
        min_phi_obs.append(f_xlow)
        self.f_acqu=self.f_tree[1]
        f_xhigh=self.f_acqu.eval_robustness(trh)
        # if (f_xlow<0):
        #   all_ce_low=all_ce_low+1
        # self.f_acqu=self.f_tree[1]
        all_ce_low=all_ce_low + np.sum(f_xlow< 0)

        for fl, fh in zip(f_xlow, f_xhigh):

              for i in range(len(fl)):
                if fl[i] < 0 and fh[i] < 0:  # Check corresponding elements
                    real_low_ce += 1
                    real_ce.append(fl[i])


        # if (f_xlow<0) and (f_xhigh<0):
        #   real_low_ce=1+real_low_ce
        trah=[]
        tral=[]
        for x in self.XH:

               trah.append(self.system_under_test_H(x))
               tral.append(self.system_under_test_L(x))

        self.f_acqu=self.f_tree[1]
        f_x_high=self.f_acqu.eval_robustness(trah)
        # f_x_high=self.f_acqu.eval_robustness(traj_H)
        min_phi_obs.append(f_x_high)
        # traj_L=self.system_under_test_L(x)
        # self.f_acqu=self.f_tree[0]
        # f_x_low=self.f_acqu.eval_robustness(tral)
        all_ce_high=all_ce_high + np.sum(f_x_high< 0)
        real_high_ce = real_high_ce + np.sum(f_x_high< 0)
        for i in range(len(f_x_high)):
            if f_x_high[i]<0:
                real_ce.append(f_x_high[i])

        # for fll, fhh in zip(f_x_low, f_x_high):
        #       for i in range(len(fll)):
        #         if fll[i] > 0 and fhh[i] < 0:  # Check corresponding elements
        #             real_high_ce += 1
        #             all_ce_high += 1

        # if (f_x_low>0) and (f_x_high<0):
        #   real_high_ce=1+real_high_ce
        #   all_ce_high=all_ce_high + 1
        global x_array,y_array
        x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])

    global XL, XH,YL, YH
    global y_array
    global real_ce
    global flag_n
    global cm
    flag_n=0
    cm=0

    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            global XL, XH,YL, YH
            global low_exp_num
            global high_exp_num
            global real_low_ce
            global real_high_ce
            global valid_low_ce
            global valid_high_ce
            global all_ce_high
            global all_ce_low
            global real_ce
            global flag_n
            global cm
            print('BO iteration:', ib)
            global x_array,y_array
            kern_low = GPy.kern.RBF(len(bounds),ARD=True)
            #kern_low.lengthscale.constrain_bounded(0.01, 0.5)
            kern_err = GPy.kern.RBF(len(bounds),ARD=True)
            #kern_err.lengthscale.constrain_bounded(0.01, 0.5)
            multi_fidelity_kernel = LinearMultiFidelityKernel([kern_low, kern_err])
            gpy_model = GPyLinearMultiFidelityModel(x_array, y_array, multi_fidelity_kernel, 2,None)
            gpy_model.mixed_noise.Gaussian_noise.fix(0.1)
            gpy_model.mixed_noise.Gaussian_noise_1.fix(0.1)
            GPmodel = GPyMultiOutputWrapper(gpy_model, 2, 1, verbose_optimization=True)
            GPmodel.optimize()
            cost_acquisition = Cost([low_fidelity_cost, high_fidelity_cost])
            acquisition = MultiInformationSourceEntropySearch(GPmodel, bound) / cost_acquisition
            acquisition_optimizer=MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(bound), bound)
            new_x,val_acq=acquisition_optimizer.optimize(acquisition)
            # print(new_x)
            tr_L=[]
            tr_H=[]
            th=[]

            if new_x[0][-1]==0.:
               print("This is low-fidelity")
               x=new_x[0][0:9]
               X_L=XL
               XL=np.vstack((X_L, x))
               low_exp_num=1+low_exp_num
               tr_L.append(self.system_under_test_L(x))
               self.f_acqu=self.f_tree[0]
               f_xl=self.f_acqu.eval_robustness(tr_L)
               min_phi_obs.append(f_xl)
               if (flag_n==0):
                       cm= cm + 6
               if f_xl<0:
                 all_ce_low=all_ce_low + 1
                 print("It's a counterexample!!!")
                 print("-----------------------")
               self.f_acqu=self.f_tree[1]
               tr_H.append(self.system_under_test_H(x))
               f_test_ce=self.f_acqu.eval_robustness(tr_H)
               if (f_xl<0) and (f_test_ce<0):
                 print("It's a valid counterexample")
                 valid_low_ce=1 + valid_low_ce
                 real_ce.append(f_xl)
                 flag_n=1

               #print(f"f_xl= {f_xl}")
               Y_L=YL
               YL=np.vstack((Y_L, f_xl))
               x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
            else:
               a=new_x[0][0:9]
               print("This is high-fidelity")
               X_H=XH
               XH=np.vstack((X_H, a))
               high_exp_num =1+high_exp_num
               th.append(self.system_under_test_H(a))
               self.f_acqu=self.f_tree[1]
               f_xh=self.f_acqu.eval_robustness(th)
               min_phi_obs.append(f_xh)
               if (flag_n==0):
                       cm= cm + 23

               if f_xh<0:
                  print("It's a counterexample!!!")
                  all_ce_high=all_ce_high + 1
                  real_ce.append(f_xh)
                  valid_high_ce=1+valid_high_ce
                  flag_n=1
               #print(f"f_xh= {f_xh}")
               Y_H=YH
               YH=np.vstack((Y_H, f_xh))
               x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
        global n
        n=0
        global sume_real_h_ce
        sume_real_h_ce=0
        global init_ce_lf
        global sum_real_ce
        global MF_c
        MF_c=0
        sum_real_ce=0
        init_ce_lf=0
        global min_val
        global new_mf_c
        new_mf_c=0
        min_val = y_array.min()
        sume_real_h_ce=(valid_high_ce)+(real_high_ce)

        sum_real_ce=(valid_high_ce)+(valid_low_ce)+(real_high_ce)+(real_low_ce)
        global all_ce
        all_ce=all_ce_high+all_ce_low
        MF_c=23*(high_exp_num)+ 6*(low_exp_num)
        new_mf_c=MF_c + 23*(all_ce_low)

# Safety specification in paper:
# 1. Either the car remains within the initial condition of state and velocity
# 2. Reaches the goal asap
from numpy import mean
import warnings
warnings.filterwarnings('ignore')
min_phi=[]
MFBO_cost=[]
real_num_ce=[]
all_ce_2f=[]
mf_new_cost=[]
############### specifications for lf
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

# ###################Specifications for hf

def pred2(trajHf):
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

########## NOn_Smooth method ############
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
      node4_hf =pred_node(f=pred2)
      node=[node4_lf,node4_hf]

      TM_ns = test_module(bounds=bounds,suth=lambda x0: sutH(x0), sutl=lambda x0: sutL(x0),
                          f_tree = node,init_sample =45, optimize_restarts=2, with_ns=True, exp_weight=2, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(140)
      min_phi.append(min_val)
      print(f"min of phi after * BO iterations: {min_phi}")
      MFBO_cost.append(MF_c)
      all_ce_2f.append(all_ce)
      real_num_ce.append(sum_real_ce)
      mf_new_cost.append(new_mf_c)
      print(f"robustness value of validated ces: {real_ce}")
      print(f"all validation cost: {mf_new_cost}")
      print(f"number of ces on HF: {sume_real_h_ce}")
      print(f"number of hf runs: {high_exp_num}")
      print(f"number of Lf runs: {low_exp_num}")
      print(f" number of valid counterexamples are : {real_num_ce}")
      print(f"cost is {MFBO_cost}")
      print(f"this all counterexamples: {all_ce_2f}")
      print(f" this is minvalue of optimization: {min_phi_obs}")
      print(f"cost of first ce: {cm}")
