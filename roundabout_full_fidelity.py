import gymnasium as gym
import stable_baselines3
import warnings
warnings.filterwarnings('ignore')
import pprint
import numpy as np
import math
import emukit
import time
import argparse
import gymnasium
from stable_baselines3 import DQN
from matplotlib import pyplot as plt
import copy
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
# from sklearn.decomposition import KernelPCA
from emukit.core.optimization.optimizer import Optimizer
from emukit.core.optimization.optimizer import OptLbfgs
from emukit.core import ContinuousParameter, ParameterSpace,InformationSourceParameter
import highway_env
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle

# highway_env.register_highway_envs()

envh = gym.make("roundabout-v0")
# obs, _ = envh.reset(seed=0)
# print(obs)


# pprint.pprint(envh.unwrapped.config)

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "normalize": False,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-50, 50],
            "vy": [-50, 50]
        },
        "see_behind": True,
        "absolute": True,
        "order": "sorted"
    }
}

envh.unwrapped.config["simulation_frequency"]= 15
envh.unwrapped.configure(config)

model = DQN.load("round_dqn_100k")
# model = DQN.load("round_dqn_k_new")


bounds = [(5,7)]
### noise on initial positions
bounds.append((-5,5))
bounds.append((-5,5))
bounds.append((-5,5))
bounds.append((-5,5))

bound = ParameterSpace([ContinuousParameter('lc', 5,7),
           ContinuousParameter('x_ob_1',-5,5),
           ContinuousParameter('x_ob_2', -5,5),
           ContinuousParameter('x_ob_3', -5,5),
           ContinuousParameter('x_ob_4', -5,5)])

def compute_trajHf(**kwargs):
    obs, _ = envh.reset(seed=0)
    IDMVehicle.LANE_CHANGE_DELAY= 0.8
    IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN = 0.1 # [m/s2]
    IDMVehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0  # [m/s2]
    
    if 'lc' in kwargs:
            Vehicle.LENGTH =kwargs['lc']

    # print(obs)
    if 'v_ego' in kwargs:
        v0 = obs[0][3] + kwargs['v_ego']
        obs[0][3]=v0

    if 'v_x_ob_1' in kwargs:
        v1 = obs[1][3]+ kwargs['v_x_ob_1']
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
        x1 = obs[0][1] + kwargs['x_ob_1']
        obs[1][1] = x1
    if 'x_ob_2' in kwargs:
        x2 = obs[0][1] + kwargs['x_ob_2']
        obs[2][1] = x2
    if 'x_ob_3' in kwargs:
        x3 = obs[0][1] + kwargs['x_ob_3']
        obs[3][1] = x3
    if 'x_ob_4' in kwargs:
        x4 = obs[0][1] + kwargs['x_ob_4']
        obs[4][1]= x4
    ob = obs[:, 1:5].reshape(-1)  
    trajHf = [ob]
    reward = 0
    done = truncated = False
    capture_interval = 1
    t = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = envh.step(action)
        ob = obs[:, 1:5].reshape(-1)  # Continue to extract x, y, vx, vy
        reward += r
        trajHf.append(ob)
        t += capture_interval
        envh.render()
        if done or truncated:
            break

    additional_data = {'reward': reward}
    return trajHf, additional_data


def sut(x0):
          return compute_trajHf(lc= x0[0],
             x_ob_1=x0[1],x_ob_2=x0[2],x_ob_3=x0[3],x_ob_4=x0[4])
        # return compute_trajHf(v_ego=x0[0],v_x_ob_1=x0[1],v_x_ob_2=x0[2],v_x_ob_3=x0[3],v_x_ob_4=x0[4],
        #      x_ob_1=x0[5],x_ob_2=x0[6],x_ob_3=x0[7],x_ob_4=x0[8])

########-------------- Utils----------------##############

def sample_from(count, bounds, sampler=None):
    if sampler is None:
        sampler = lambda num: np.random.random(num)

    sampled_01 = sampler(count*len(bounds))
    sampled_01.resize(count,len(bounds))
    sampled_01 = sampled_01.T
    sampled_lb = [sampled_01[i]*(b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]

    return np.array(sampled_lb).T

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


        # Sending in pre sampled data
        if 'X' in kwargs:
            self.X = kwargs['X']
        else:
            self.X = []


    def initialize(self):
        global min_phi_obs
        min_phi_obs=[]
        global num_ce
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
            self.ns_GP.Gaussian_noise.fix(1)
            global Hf_model
            Hf_model = GPyModelWrapper(self.ns_GP)

    global p
    p=[]
    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            print('BO iteration:', ib)
            global X_ns
            global num_ce
            trs=[]
            if self.with_ns:
              global Hf_model
              Hf_acq=EntropySearch(Hf_model, bound)
              x,_ = self.optimizer.optimize(Hf_acq,None)
              # trs.append(self.system_under_test(x_i) for x_i in x)
              trs.extend(self.system_under_test(x_i) for x_i in x)
              f_x = self.f_acqu.eval_robustness(trs)
              min_phi_obs.append(f_x)
              if f_x <0:
                print("counterexample found")
                num_ce=num_ce + 1
              #print(f_x)
              self.ns_X = np.vstack((self.ns_X, x))
              X_ns = self.ns_X
              Y=np.vstack((self.ns_GP.Y, f_x))
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

# from collections import defaultdict

def pred1(trajHf):
    trajHf = trajHf[0]  # Extract the trajectory data from the tuple
    min_pred1 = float('inf')

    for observation in trajHf:
        observation = np.array(observation)
        observation = observation.reshape(5, 4)  # 5 vehicles, 4 features (x, y, vx, vy)
        ego_y = observation[0][1]  # y position of ego vehicle

        for i in range(1, 5):  # Iterate over the 4 other vehicles
            y_i = observation[i][1]
            pred_value = abs(y_i - ego_y) - 2.2
            min_pred1 = min(min_pred1, pred_value)

    return min_pred1

def pred2(trajHf):
    trajHf = trajHf[0]  # Extract the trajectory data from the tuple
    min_pred2 = float('inf')

    for observation in trajHf:
        observation = np.array(observation)
        observation = observation.reshape(5, 4)  # 5 vehicles, 4 features (x, y, vx, vy)
        ego_x = observation[0][0]  # x position of ego vehicle

        for i in range(1, 5):  # Iterate over the 4 other vehicles
            x_i = observation[i][0]
            pred_value = abs(x_i - ego_x) - 4.44
            min_pred2 = min(min_pred2, pred_value)

    return min_pred2

def pred3(trajHf):
    trajHf = trajHf[0]  # Extract the trajectory data from the tuple
    min_pred3 = float('inf')

    for observation in trajHf:
        observation = np.array(observation)
        observation = observation.reshape(5, 4)  # 5 vehicles, 4 features (x, y, vx, vy)
        ego_x = observation[0][0]  # x position of ego vehicle

        for i in range(1, 5):  # Iterate over the 4 other vehicles
            x_i = observation[i][0]
            pred_value = abs(x_i - ego_x) - 2.2
            min_pred3 = min(min_pred3, pred_value)

    return min_pred3

def pred4(trajHf):
    trajHf = trajHf[0]  # Extract the trajectory data from the tuple
    min_pred4 = float('inf')

    for observation in trajHf:
        observation = np.array(observation)
        observation = observation.reshape(5, 4)  # 5 vehicles, 4 features (x, y, vx, vy)
        ego_y = observation[0][1]  # y position of ego vehicle

        for i in range(1, 5):  # Iterate over the 4 other vehicles
            y_i = observation[i][1]
            pred_value = abs(y_i - ego_y) - 4.44
            min_pred4 = min(min_pred4, pred_value)

    return min_pred4



#####---------------##################
rand_num = list(range(0,750))
parser = argparse.ArgumentParser(description='Takes and integer as random seed and runs the code')
parser.add_argument('-r', metavar='N', type=int, help='Index to pick from the rand_num')
args = parser.parse_args()
print("Number of elements in the random seed list %d" % len(rand_num) )
print("The index from random seed list : %d" % args.r)
print("Value picked: %d" % rand_num[args.r])
rand_num2=[rand_num[args.r]]


for r in rand_num2:
      np.random.seed(r)
      # node4 = pred_node(f=pred1)
      node1 = pred_node(f=pred1)
      node2 = pred_node(f=pred2)
      node3 = pred_node(f=pred3)
      node4 = pred_node(f=pred4)


      nodeA=max_node(children=[node1, node2])
      nodeB=max_node(children=[node3, node4])
      node=min_node(children=[nodeA,nodeB])

      TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0), optimizer=GradientAcquisitionOptimizer(bound),f_tree = node,init_sample =60, with_ns=True, optimize_restarts=1, exp_weight=5,seed=r, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(140)
      min_phi.append(min_val)
      print(f"min of phi after * BO iterations: {min_phi}")
      num_ce_real.append(num_ce)
      print(f'number of counterexamples: {num_ce_real}')
      print(f" this is minvalue of optimization: {min_phi_obs}")
