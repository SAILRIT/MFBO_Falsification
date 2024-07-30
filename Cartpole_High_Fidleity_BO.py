#!/usr/bin/env python3 
import stable_baselines3
import gym
import GPy
from stable_baselines3 import PPO
import gym
import numpy as np
import emukit
import time 
import argparse



env = gym.make("CartPole-v1")
env.env.kinematics_integrator = "semi-implicit euler"
env.env.force_mag = 20.0

# modelppo = PPO("MlpPolicy", env, verbose=1)
# modelppo.learn(total_timesteps=25000)
# modelppo.save("PPO_CP")


modelppo = PPO.load("PPO_CP.zip")


#Trajectories
# SUT
import numpy as np
from gym import spaces
def compute_traj(max_steps, **kwargs):
    env.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
    if 'masspole' in kwargs:
        env.env.masspole = kwargs['masspole']
        env.env.total_mass = env.env.masspole + env.env.masscart
        env.env.polemass_length = env.env.masspole * env.env.length
    if 'length' in kwargs:
        env.env.length = kwargs['length']
        env.env.polemass_length = env.env.masspole * env.env.length
    # if 'force_mag' in kwargs:
    #     env.env.force_mag = kwargs['force_mag']
    traj = [ob]
    reward = 0
    done=False
    iter_time = 0
    while done==False:
    # for _ in range(max_steps):
        iter_time += 1
        pi, _ = modelppo.predict(ob)
        action=pi
        ob, r, terminated, truncated, info=env.step(action)
        # ob, r, done, _ = env.step(action)
        reward = r+reward
        traj.append(ob)
        done = terminated or iter_time > max_steps
        if done:
            break
    additional_data = {'reward':reward, 'mass':env.env.total_mass}
    return traj, additional_data

def sut(max_steps,x0, ead=False):
    return compute_traj(max_steps,init_state=x0[0:4], masspole=x0[4],length=x0[5])

######## Utils ##############
import numpy as np
def sample_from(count, bounds, sampler=None):
    if sampler is None:
        sampler = lambda num: np.random.random(num)

    sampled_01 = sampler(count*len(bounds))
    sampled_01.resize(count,len(bounds))
    sampled_01 = sampled_01.T
    sampled_lb = [sampled_01[i]*(b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]

    return np.array(sampled_lb).T

bounds = [(-2, 2)]  # Bounds on the state
bounds.append((-0.05, 0.05)) # Bounds on the 
bounds.append((-0.2, 0.2)) # Bounds on the
bounds.append((-0.05, 0.05)) # Bounds on the 

bounds.append((0.05, 0.15)) # Bounds on the mass of the pole
bounds.append((0.4, 0.6)) # Bounds on the length of the pole
# bounds.append((18, 22)) # Bounds on the force magnitude
from emukit.core import ContinuousParameter, ParameterSpace,InformationSourceParameter
bound = ParameterSpace([ContinuousParameter('x', -2,2),
           ContinuousParameter('vx', -0.05,0.05),
           ContinuousParameter('theta',-0.2,0.2), 
           ContinuousParameter('d_theta', -0.05,0.05),
           ContinuousParameter('mass', 0.05,0.15),
           ContinuousParameter('len', 0.4,0.6)])

#Function tree ##
import numpy as np
import GPy
import copy

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

from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch
from emukit.model_wrappers import GPyModelWrapper
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
# from sklearn.decomposition import KernelPCA
from emukit.core.optimization.optimizer import Optimizer
from emukit.core.optimization.optimizer import OptLbfgs
import copy

class test_module:
    def __init__(self, sut, bounds, spec=None,f_tree=None,optimizer=None,
                 normalizer=False,seed=None, **kwargs):
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

        trajs = []
        for x in self.X:
            trajs.append(self.system_under_test(x))
        Y = self.f_acqu.eval_robustness(trajs)
        min_phi_obs.append(Y)
        # for i in range(len(Y)):
        #   if Y[i][0]<0:
        #     num_ce+=1

        if self.with_ns:
            self.ns_X = copy.deepcopy(self.X)
            X_ns = copy.deepcopy(self.ns_X)
            self.ns_GP = GPy.models.GPRegression(X_ns, Y,
                                        kernel=copy.deepcopy(self.kernel),
                                        normalizer=self.normalizer)
            global Hf_model
            Hf_model = GPyModelWrapper(self.ns_GP)

    global p
    p=[]
    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            print('BO iteration:', ib)
            global X_ns
            global num_ce
            # trm=[]
            if self.with_ns:
              global Hf_model
              Hf_acq=EntropySearch(Hf_model, bound)
              x,_ = self.optimizer.optimize(Hf_acq,None)
              trm=[self.system_under_test(x_i) for x_i in x]
              f_x = self.f_acqu.eval_robustness(trm)
              min_phi_obs.append(f_x)
              if f_x <0:
                #print("counterexample found")
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
          global n
          n=0
          for i in range(len(self.ns_GP.Y)):
            if self.ns_GP.Y[i][0]<0:
              n+=1

# Safety specification in paper:
# 1. Either the car remains within the initial condition of state and velocity
# 2. Reaches the goal asap
from emukit.core.optimization import GradientAcquisitionOptimizer
nums=[]
num_ce_real=[]
min_phi=[]
import warnings
warnings.filterwarnings('ignore')

def pred2(traj):
    traj_ = traj[0]
    mass = traj[1]['mass']
    v_s = np.array(traj_).T[1]
    return min(1 - np.abs(mass*v_s))

def pred3(traj):
    traj=traj[0]
    theta=np.array(traj).T[2]
    return min(0.157 - np.abs(theta))

def pred4(traj):
    traj=traj[0]
    x_s = np.array(traj).T[0]
    return min(1 - np.abs(x_s))

##################### Non Smooth method ######################
rand_num = list(range(1, 751))

# rand_num=[1]

#####---------------##################
parser = argparse.ArgumentParser(description='Takes and integer as random seed and runs the code')
parser.add_argument('-r', metavar='N', type=int, help='Index to pick from the rand_num')

args = parser.parse_args()
print("Number of elements in the random seed list %d" % len(rand_num) )
print("The index from random seed list : %d" % args.r)
print("Value picked: %d" % rand_num[args.r])

rand_num2=[rand_num[args.r]]

for r in rand_num2:
      np.random.seed(r)
      node1 = pred_node(f=lambda traj: pred2(traj))
      node2 = pred_node(f=lambda traj: pred3(traj))
      node3 = pred_node(f=lambda traj: pred4(traj))
      node4 = min_node(children= [node1, node3, node2])
      TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(450,x0), optimizer=GradientAcquisitionOptimizer(bound),f_tree = node4,init_sample =60, with_ns=True,
                          optimize_restarts=1, exp_weight=2,seed=r, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(140)
      min_phi.append(min_val)
      nums.append(n)
      print(f"min of phi after * BO iterations: {min_phi}")
      # num_ce_real.append(num_ce)
      print(f"number of counterexamples: {nums}")
      print(f" this is minvalue of optimization: {min_phi_obs}")
    #   print(np.mean(num_ce_real))
