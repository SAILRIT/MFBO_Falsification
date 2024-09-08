

import highway_env

# highway_env.register_highway_envs()
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import DQN
import warnings
warnings.filterwarnings('ignore')
import pprint

#!/usr/bin/env python3
import gymnasium as gym
import sys
sys.modules["gym"] = gym
import warnings
warnings.filterwarnings('ignore')
# from gym import spaces
# from tqdm.notebook import trange
from stable_baselines3 import DDPG
import numpy as np
import math
from stable_baselines3.common.vec_env import DummyVecEnv
import emukit
import time
import argparse
import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from matplotlib import pyplot as plt
import numpy as np
import copy
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch

from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
# from sklearn.decomposition import KernelPCA
from emukit.core.optimization.optimizer import Optimizer
from emukit.core.optimization.optimizer import OptLbfgs
import copy
import pprint

import highway_env
# highway_env.register_highway_envs()

envh = gym.make("merge-v0", render_mode='rgb_array')

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
from highway_env.vehicle.behavior import IDMVehicle
model = DQN.load("dqn_merge_model")
# model = DQN.load("dqn_merge_model_twelve")
# model = DQN.load("dqn_merge_model_new")

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


def sut(x0):
      return compute_trajHf(v_ego=x0[0],v_x_ob_1=x0[1],v_x_ob_2=x0[2],v_x_ob_3=x0[3],v_x_ob_4=x0[4],
             x_ob_1=x0[5],x_ob_2=x0[6],x_ob_3=x0[7],x_ob_4=x0[8])
    # return compute_trajHf(v_x_ob_1=x0[0],v_x_ob_2=x0[1],v_x_ob_3=x0[2],v_x_ob_4=x0[3],
    #          x_ob_1=x0[4],x_ob_2=x0[5],x_ob_3=x0[6],x_ob_4=x0[7])


########-------------- Utils----------------##############
import numpy as np
def sample_from(count, bounds, sampler=None):
    if sampler is None:
        sampler = lambda num: np.random.random(num)

    sampled_01 = sampler(count*len(bounds))
    sampled_01.resize(count,len(bounds))
    sampled_01 = sampled_01.T
    sampled_lb = [sampled_01[i]*(b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]

    return np.array(sampled_lb).T

# bounds = [(1,2)] #noise on initial velocities


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
           ContinuousParameter('x_ob_4', 10, 12)])


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

ns_Failure_count=[]
ns_details_r3 = []
num_nonsmooth=[]

from emukit.model_wrappers import GPyModelWrapper
import GPy
class test_module:
    def __init__(self, sut, bounds, spec=None,f_tree=None,optimizer=None, normalizer=False,seed=None, **kwargs):
        self.system_under_test = sut

        # Choosing the optimizer function
        if spec is None:
            self.f_acqu = f_tree
        else:
            self.spec = spec
            # To implement parser to convert from specification to the function f

        self.bounds = bounds
        self.normalizer=normalizer
        self.seed=seed

        if 'cost_model' in kwargs:
            self.cost_model = kwargs['cost_model']
        else:
            self.cost_model = lambda x: 1

        # Choosing the optimizers
        if 'opt_name' in kwargs:
            self.optimizer = select_opt(kwargs['opt_name'])(bounds, **kwargs)
        elif optimizer is None:
            self.optimizer = sample_opt(bounds=bounds, cost=self.cost_model)
        else:
            self.optimizer = optimizer

        # Number of samples for initializing GPs
        if 'init_sample' in kwargs:
            self.init_sample = kwargs['init_sample']
        else:
            self.init_sample = 2*len(bounds)

        # Model GPs for the top level requirement, potentially modeling
        # non-smooth function
        if 'with_ns' in kwargs:
            self.with_ns = kwargs['with_ns']
        else:
            self.with_ns = False


        # Exploration weight for GP-LCB
        if 'exp_weight' in kwargs:
            self.k = kwargs['exp_weight']
        else:
            self.k = 10

        # Optimize retsrats for hyper parameter optimization for GPs
        if 'optimize_restarts' in kwargs:
            self.optimize_restarts = kwargs['optimize_restarts']
        else:
            self.optimize_restarts = 1

        if 'kernel_type' in kwargs:
            self.kernel = kwargs['kernel_type'](len(bounds))
        else:
            self.kernel = GPy.kern.Matern32(len(bounds), ARD=True)
            # self.kernel=GPy.kern.RBF(len(bounds),ARD=True)


        # Sending in pre sampled data
        if 'X' in kwargs:
            self.X = kwargs['X']
        else:
            self.X = []


    def initialize(self):
        global min_phi_obs
        min_phi_obs=[]
        global num_ce
        global Y
        num_ce=0
        global X_ns
        if len(self.X) == 0:
            X = sample_from(self.init_sample, self.bounds)
            self.X = X

        trjs = []
        for x in self.X:
            trjs.append(self.system_under_test(x))
        Y = self.f_acqu.eval_robustness(trjs)
        min_phi_obs.append(Y)
        for i in range(len(Y)):
          if Y[i][0]<0:
            num_ce+=1

        if self.with_ns:
            self.ns_X = copy.deepcopy(self.X)
            X_ns = copy.deepcopy(self.ns_X)
            self.ns_GP = GPy.models.GPRegression(X_ns, Y,kernel=copy.deepcopy(self.kernel),normalizer=self.normalizer)
            self.ns_GP.Gaussian_noise.fix(0.1)
            self.ns_GP.Gaussian_noise.variance.fix(0.1)
            
            self.ns_GP.optimize_restarts(self.optimize_restarts)
            global Hf_model
            Hf_model = GPyModelWrapper(self.ns_GP)
            

    global p
    p=[]
    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            print('BO iteration:', ib)
            global X_ns, Y
            global num_ce
            # trs=[]
            if self.with_ns:
              global Hf_model
              Hf_acq=EntropySearch(Hf_model, bound)
              x,_ = self.optimizer.optimize(Hf_acq,None)
              # trs.append(self.system_under_test(x_i) for x_i in x)
              trs=[self.system_under_test(x_i) for x_i in x]
              f_x = self.f_acqu.eval_robustness(trs)
              min_phi_obs.append(f_x)
              if f_x <0:
                print("counterexample found")
                num_ce=num_ce + 1
              #print(f_x)
              self.ns_X = np.vstack((self.ns_X, x))
              X_ns = self.ns_X
              Y_L=Y
              Y=np.vstack((Y_L, f_x))
              self.ns_GP.set_XY(X_ns,Y)
              self.ns_GP.optimize_restarts(self.optimize_restarts)
              Hf_model = GPyModelWrapper(self.ns_GP)
        global min_val
        if self.with_ns:
          self.ns_min_val = self.ns_GP.Y.min()

          min_val=self.ns_min_val

# Safety specification in paper:
# 1. Either the car remains within the initial condition of state and velocity
# 2. Reaches the goal asap
from emukit.core.optimization import GradientAcquisitionOptimizer
nums=[]
num_ce_real=[]
min_phi=[]

from collections import defaultdict

def pred1(trajHf):
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

#####---------------##################
rand_num = list(range(1,750))

parser = argparse.ArgumentParser(description='Takes and integer as random seed and runs the code')
parser.add_argument('-r', metavar='N', type=int, help='Index to pick from the rand_num')
args = parser.parse_args()
print("Number of elements in the random seed list %d" % len(rand_num) )
print("The index from random seed list : %d" % args.r)
print("Value picked: %d" % rand_num[args.r])
rand_num2=[rand_num[args.r]]

for r in rand_num2:
      np.random.seed(r)
      node4 = pred_node(f=pred1)
      TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0), optimizer=GradientAcquisitionOptimizer(bound),f_tree = node4,init_sample =60, with_ns=True, optimize_restarts=1, exp_weight=10,seed=r, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(140)
      min_phi.append(min_val)
      # print(f"min of phi after * BO iterations: {min_phi}")
      num_ce_real.append(num_ce)
      print(f'number of counterexamples: {num_ce_real}')
      # print(f" this is minvalue of optimization: {min_phi_obs}")
