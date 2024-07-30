
# !pip install stable-baselines3[extra]
# !pip install gymnasium==0.28.1
# # !pip install stable_baselines3
# !pip install emukit
# !pip install gym
# !pip install swig
# !pip install gym[box2d]

# !pip install GPy

#!/usr/bin/env python3
import time
import argparse
import stable_baselines3
import gym
# !pip install gpy
# import GPy
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3 import DDPG
import emukit
import time
import math
from gym import spaces

envh = gym.make('LunarLanderContinuous-v2')
envh.enable_wind=True
envh.wind_power=19.9
envh.turbulence_power=1.99

envm=gym.make('LunarLanderContinuous-v2')
envm.enable_wind=True
envm.wind_power=6

# envm.turbulence_power=0.99
# model = DDPG("MlpPolicy", envh, action_noise=None, verbose=1)
# model.learn(total_timesteps=100000, log_interval=10)
# model.save("ddpg_LunarLanderContinuous-v2")

modelddpg = DDPG.load("ddpg_LunarLanderContinuous-v2")

def compute_trajHf(max_steps,**kwargs):
    envh.reset()
    if 'init_state' in kwargs:
        envh.env.lander.position=kwargs['init_state']
    if 'init_velocity' in kwargs:
        envh.env.lander.linearVelocity = kwargs['init_velocity']
    # State perturbation
    if 'state_per' in kwargs:
        state_per = kwargs['state_per']
    # Velocity perturbation
    if 'vel_per' in kwargs:
        vel_per = kwargs['vel_per']
    # env.env.lander.
    envh.env.lander.position[0] = envh.env.lander.position[0] + state_per[0]
    envh.env.lander.position[1] = envh.env.lander.position[1] + state_per[1]
    envh.lander.linearVelocity[0]=envh.lander.linearVelocity[0]+vel_per[0]
    envh.lander.linearVelocity[1]=envh.lander.linearVelocity[1]+vel_per[1]
    ob=np.array([envh.env.lander.position[0],envh.env.lander.position[1],
                     envh.lander.linearVelocity[0],envh.lander.linearVelocity[1],envh.env.lander.angle,
                     envh.env.lander.angularVelocity,0,0])
    iter_time = 0
    r = 0
    done=False
    trajHf = [ob]
    # while done==False:
    while done==False:
        iter_time += 1
        po, _ = modelddpg.predict(ob)
        #action, _ = pi.act(False, ob)
        action=po
        # ob, reward, terminated, _ = envh.step(action)
        ob, reward, terminated, truncated, info=envh.step(action)

        trajHf.append(ob)
        r+= reward
        done = terminated or iter_time >= max_steps
        if done:
            break
    return trajHf, {'reward':r}

def sutH(max_steps,x0):
    state_per = np.zeros(2)
    state_per[0:2] += x0[0:2]
    vel_per = np.zeros(2)
    vel_per[0:2] += x0[2:4]
    return compute_trajHf(max_steps, state_per=state_per,vel_per=vel_per)
###################----------------------
def compute_trajMf(max_steps,**kwargs):
    envm.reset()
    if 'init_state' in kwargs:
        envm.env.lander.position=kwargs['init_state']
    if 'init_velocity' in kwargs:
        envm.env.lander.linearVelocity = kwargs['init_velocity']
    # State perturbation
    if 'state_per' in kwargs:
        state_per = kwargs['state_per']
    # Velocity perturbation
    if 'vel_per' in kwargs:
        vel_per = kwargs['vel_per']
    # env.env.lander.
    envm.env.lander.position[0] = envm.env.lander.position[0] + state_per[0]
    envm.env.lander.position[1] = envm.env.lander.position[1] + state_per[1]
    envm.lander.linearVelocity[0]=envm.lander.linearVelocity[0]+vel_per[0]
    envm.lander.linearVelocity[1]=envm.lander.linearVelocity[1]+vel_per[1]
    ob=np.array([envm.env.lander.position[0],envm.env.lander.position[1],
                     envm.lander.linearVelocity[0],envm.lander.linearVelocity[1],envm.env.lander.angle,
                     envm.env.lander.angularVelocity,0,0])
    iter_time = 0
    r = 0
    done=False
    trajMf = [ob]
    while done==False:
        iter_time += 1
        po, _ = modelddpg.predict(ob)
        action=po
        # ob, reward, terminated, _ = envm.step(action)
        ob, reward, terminated, truncated, info=envm.step(action)
        noise_x = np.random.normal(0,0.05,1)
        noise_y = np.random.normal(0,0.05,1)
        noise_v_x=np.random.normal(0,0.2,1)
        noise_v_y=np.random.normal(0,0.2,1)
        pi_n = math.pi
        noise_theta=np.random.normal(0,0.03*pi_n,1)
        noise_dtheta=np.random.normal(0,0.1,1)

        ob[0]+=noise_x
        ob[1]+=noise_y
        ob[2]+=noise_v_x
        ob[3]+=noise_v_y
        ob[4]+=noise_theta
        ob[5]+=noise_dtheta
        trajMf.append(ob)
        r+= reward
        done = terminated or iter_time >= max_steps
        if done:
            break
    return trajMf, {'reward':r}

def sutM(max_steps,x0):
    state_per = np.zeros(2)
    state_per[0:2] += x0[0:2]
    vel_per = np.zeros(2)
    vel_per[0:2] += x0[2:4]
    return compute_trajMf(max_steps, state_per=state_per,vel_per=vel_per)
################-----------------------------------------------------
envl=gym.make('LunarLanderContinuous-v2')
def compute_trajLf(max_steps,**kwargs):
    envl.reset()
    if 'init_state' in kwargs:
        envl.env.lander.position=kwargs['init_state']
    if 'init_velocity' in kwargs:
        envl.env.lander.linearVelocity = kwargs['init_velocity']
    # State perturbation
    if 'state_per' in kwargs:
        state_per = kwargs['state_per']
    # Velocity perturbation
    if 'vel_per' in kwargs:
        vel_per = kwargs['vel_per']
    # env.env.lander.
    envl.env.lander.position[0] = envl.env.lander.position[0] + state_per[0]
    envl.env.lander.position[1] = envl.env.lander.position[1] + state_per[1]
    envl.lander.linearVelocity[0]=envl.lander.linearVelocity[0]+vel_per[0]
    envl.lander.linearVelocity[1]=envl.lander.linearVelocity[1]+vel_per[1]
    ob=np.array([envl.env.lander.position[0],envl.env.lander.position[1],
                     envl.lander.linearVelocity[0],envl.lander.linearVelocity[1],envl.env.lander.angle,
                     envl.env.lander.angularVelocity,0,0])
    iter_time = 0
    r = 0
    done=False
    trajLf = [ob]
    # while done==False:
    while done==False:
        iter_time += 1
        po, _ = modelddpg.predict(ob)
        #action, _ = pi.act(False, ob)
        action=po
        ob, reward, terminated, truncated, info=envl.step(action)
        # ob, reward, terminated, _=envl.step(action)
        noise_x = np.random.normal(0,0.12,1)
        noise_y = np.random.normal(0,0.12,1)
        noise_v_x=np.random.normal(0,0.5,1)
        noise_v_y=np.random.normal(0,0.5,1)
        pi_n = math.pi
        noise_theta=np.random.normal(0,0.08*pi_n,1)
        noise_dtheta=np.random.normal(0,0.3,1)
        ob[0]+=noise_x
        ob[1]+=noise_y
        ob[2]+=noise_v_x
        ob[3]+=noise_v_y
        ob[4]+=noise_theta
        ob[5]+=noise_dtheta
        trajLf.append(ob)
        r+= reward
        done = terminated or iter_time >= max_steps
        if done:
            break
    return trajLf, {'reward':r}

def sutL(max_steps,x0):
    state_per = np.zeros(2)
    state_per[0:2] += x0[0:2]
    vel_per = np.zeros(2)
    vel_per[0:2] += x0[2:4]
    return compute_trajLf(max_steps, state_per=state_per,vel_per=vel_per)

######## - - - - - --- - -   Utils ----- ---  - - - - ##############
def sample_from(count, bounds, sampler=None):
    if sampler is None:
        sampler = lambda num: np.random.random(num)

    sampled_01 = sampler(count*len(bounds))
    sampled_01.resize(count,len(bounds))
    sampled_01 = sampled_01.T
    sampled_lb = [sampled_01[i]*(b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]
    return np.array(sampled_lb).T

bounds = [(-0.5, 0.5)] # Bounds on the X perturbations
bounds.append((0, 3)) # Bounds on the Y perturbations
bounds.append((-2, 2)) ## Bounds on the x velocity perturbations
bounds.append((0, 2)) # Bounds on the y velocity perturbations

from emukit.core import ContinuousParameter, ParameterSpace,InformationSourceParameter

bound = ParameterSpace([ContinuousParameter('p_x', -0.5,0.5),
           ContinuousParameter('p_y', 0,3),
           ContinuousParameter('vx', -2,2),
           ContinuousParameter('vy', 0,2),InformationSourceParameter(3)])

#Function tree ##

import copy
import GPy
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

from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
low_fidelity_cost = 202
high_fidelity_cost = 916
mid_fidelity_cost=452

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
        if 'XM' in kwargs:
            self.XM = kwargs['XM']
        else:
            self.XM = []

        if 'XH' in kwargs:
            self.XH = kwargs['XH']
        else:
            self.XH = []

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

        global all_ce_high
        global all_ce_low
        global all_ce_mid

        global min_phi_obs
        min_phi_obs=[]
        all_ce_low=0
        all_ce_high=0
        all_ce_mid=0

        real_high_ce=0
        real_mid_ce=0
        real_low_ce=0
        valid_low_ce=0
        valid_high_ce=0
        valid_mid_ce=0
        global XL,XM, XH,YL,YM, YH
        if len(self.XL) == 0:
            XL = sample_from(self.init_sample, self.bounds)
            self.XL = XL
        if len(self.XM) == 0:
            # XM=XL
            o=self.init_sample//2
            XM= np.atleast_2d(np.random.permutation(XL)[:o])
            self.XM = XM

        if len(self.XH) == 0:
            # XH=XL
            no=self.init_sample//3
            XH = np.atleast_2d(np.random.permutation(XM)[:no])
            self.XH = XH
        # print(XH)
        global trajsL,trajsH,trajsM
        global YL, YH
        trajsL = []
        trajsH = []
        trajsM=[]
        for x in self.XL:
            trajsL.append(self.system_under_test_L(x))
        self.f_acqu=self.f_tree[0]
        YL = self.f_acqu.eval_robustness(trajsL)
        for x in self.XM:
            trajsM.append(self.system_under_test_M(x))
        self.f_acqu=self.f_tree[1]
        YM = self.f_acqu.eval_robustness(trajsM)

        for x in self.XH:
            trajsH.append(self.system_under_test_H(x))
        self.f_acqu=self.f_tree[2]
        YH = self.f_acqu.eval_robustness(trajsH)

        low_exp_num=self.init_sample
        mid_exp_num=self.init_sample//2
        high_exp_num=self.init_sample//3
        ##############-------initial counterexamples----------####
        trL = []
        trH = []
        trM=[]
        for x in self.XL:
               trL.append(self.system_under_test_L(x))
               trH.append(self.system_under_test_H(x))

        self.f_acqu=self.f_tree[0]
        f_xlow=self.f_acqu.eval_robustness(trL)
        min_phi_obs.append(f_xlow)
        all_ce_low=all_ce_low+np.sum(f_xlow< 0)
        # if (f_xlow<0):
        #   all_ce_low=all_ce_low+1

        self.f_acqu=self.f_tree[2]

        f_xhigh=self.f_acqu.eval_robustness(trH)
        for fl, fh in zip(f_xlow, f_xhigh):
                    for i in range(len(fl)):
                      if fl[i] < 0 and fh[i] < 0:  # Check corresponding elements
                          real_low_ce += 1

        # if (f_xlow<0) and (f_xhigh<0):
        #   real_low_ce=1 + real_low_ce
        trjh=[]
        trjl=[]
        for x in self.XM:
               trM.append(self.system_under_test_M(x))
               trjh.append(self.system_under_test_H(x))
               trjl.append(self.system_under_test_L(x))



        self.f_acqu=self.f_tree[1]
        f_xmid=self.f_acqu.eval_robustness(trM)
        min_phi_obs.append(f_xmid)
        self.f_acqu=self.f_tree[2]
        f_xhigh=self.f_acqu.eval_robustness(trjh)
        # self.f_acqu=self.f_tree[0]
        # f_x_low=self.f_acqu.eval_robustness(trjl)
        all_ce_mid=all_ce_mid+np.sum(f_xmid< 0)

        for fm, fh in zip( f_xmid, f_xhigh):
                    for i in range(len(fm)):
                      if fm[i]<0 and fh[i]<0:
                          real_mid_ce=1+real_mid_ce
        th=[]
        # tm=[]
        # tl=[]

        for x in self.XH:
               th.append(self.system_under_test_H(x))
               # tl.append(self.system_under_test_L(x))
               # tm.append(self.system_under_test_M(x))


        self.f_acqu=self.f_tree[2]
        f_x_high=self.f_acqu.eval_robustness(th)
        min_phi_obs.append(f_x_high)
        # self.f_acqu=self.f_tree[0]
        # f_x_low=self.f_acqu.eval_robustness(tl)
        # self.f_acqu=self.f_tree[1]
        # f_x_m=self.f_acqu.eval_robustness(tm)
        all_ce_high=all_ce_high+np.sum(f_x_high< 0)
        real_high_ce = real_high_ce+np.sum(f_x_high< 0)

        # for fl, fm, fh in zip(f_x_low, f_x_m, f_x_high):
        #     for i in range(len(fh)):
        #         if fl[i]>0 and fm[i]>0 and fh[i]<0:
        #             all_ce_high=all_ce_high + 1
        #             real_high_ce=1+real_high_ce

        global x_array,y_array
        x_array, y_array = convert_xy_lists_to_arrays([XL,XM,XH], [YL,YM,YH])

    global XL, XH,YL, YH
    global y_array, x_array

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
            print('BO iteration:', ib)
            global x_array,y_array
            kern_low = GPy.kern.RBF(4,ARD=True)
            #kern_low.lengthscale.constrain_bounded(0.01, 0.5)
            kern_err = GPy.kern.RBF(4,ARD=True)
            kernel_mid=GPy.kern.RBF(4,ARD=True)
            #kern_err.lengthscale.constrain_bounded(0.01, 0.5)
            multi_fidelity_kernel = LinearMultiFidelityKernel([kernel_mid,kern_low, kern_err])
            gpy_model = GPyLinearMultiFidelityModel(x_array, y_array, multi_fidelity_kernel, 3,None)
            gpy_model.mixed_noise.Gaussian_noise.fix(0.0001)
            gpy_model.mixed_noise.Gaussian_noise_1.fix(0.0001)
            gpy_model.mixed_noise.Gaussian_noise_2.fix(0.0001)
            GPmodel = GPyMultiOutputWrapper(gpy_model, 3, 1, verbose_optimization=True)
            GPmodel.optimize()
            cost_acquisition = Cost([low_fidelity_cost,mid_fidelity_cost, high_fidelity_cost])
            acquisition = MultiInformationSourceEntropySearch(GPmodel, bound) / cost_acquisition
            acquisition_optimizer=MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(bound), bound)
            new_x,val_acq=acquisition_optimizer.optimize(acquisition)
            #print(new_x)
            TL=[]
            TH=[]
            TM=[]
            THH=[]
            THHH=[]

            if new_x[0][-1]==0.:
               print("This is low-fidelity")
               x=new_x[0][0:4]
               X_L=XL
               XL=np.vstack((X_L, x))
               low_exp_num=1+low_exp_num
               TL.append(self.system_under_test_L(x))
               self.f_acqu=self.f_tree[0]
               f_xl=self.f_acqu.eval_robustness(TL)
               min_phi_obs.append(f_xl)
               if f_xl<0:
                 all_ce_low=all_ce_low+1
               self.f_acqu=self.f_tree[2]
               TH.append(self.system_under_test_H(x))
               f_test_ce=self.f_acqu.eval_robustness(TH)
               if (f_xl<0) and (f_test_ce<0):
                #  print("It's a valid counterexample")
                 valid_low_ce=1+valid_low_ce

               Y_L=YL
               YL=np.vstack((Y_L, f_xl))
               x_array, y_array = convert_xy_lists_to_arrays([XL,XM,XH], [YL,YM,YH])
            elif new_x[0][-1]==1.:
               print("This is mid-fidelity")
               #print(f"f_xl= {f_xl}")
               xm=new_x[0][0:4]
               X_M=XM
               XM=np.vstack((X_M, xm))
               mid_exp_num=1+mid_exp_num
               TM.append(self.system_under_test_M(xm))
               self.f_acqu=self.f_tree[1]
               f_xm=self.f_acqu.eval_robustness(TM)
               min_phi_obs.append(f_xm)
               if f_xm<0:
                 all_ce_mid=all_ce_mid + 1
            #    print(f"f_xm is: {f_xm}")
               Y_M=YM
               YM=np.vstack((Y_M, f_xm))
               x_array, y_array = convert_xy_lists_to_arrays([XL,XM,XH], [YL,YM,YH])
               self.f_acqu=self.f_tree[2]
               THH.append(self.system_under_test_H(xm))
               f_test_ce=self.f_acqu.eval_robustness(THH)
               if (f_xm<0) and (f_test_ce<0):
                #  print("It's a valid counterexample")
                #  print(f"f_xm is: {f_xm}")
                 valid_mid_ce=1+ valid_mid_ce

            else:
               a=new_x[0][0:4]
               print("This is high-fidelity")
               X_H=XH
               XH=np.vstack((X_H, a))
               high_exp_num =1+high_exp_num
               THHH.append(self.system_under_test_H(a))
               self.f_acqu=self.f_tree[2]
               f_xh=self.f_acqu.eval_robustness(THHH)
               min_phi_obs.append(f_xh)
               Y_H=YH
               if f_xh<0:
                  valid_high_ce=1+valid_high_ce
                  all_ce_high=all_ce_high + 1
            #    print(f"f_xh= {f_xh}")
               YH=np.vstack((Y_H, f_xh))
               x_array, y_array = convert_xy_lists_to_arrays([XL,XM,XH], [YL,YM,YH])
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
        sum_real_h_ce=(valid_high_ce)+(real_high_ce)
        sum_real_m_ce=(valid_mid_ce)+(real_mid_ce)
        sum_real_l_ce=(valid_low_ce)+(real_low_ce)

        sum_real_ce=(sum_real_l_ce)+(sum_real_m_ce)+(sum_real_h_ce)
        global all_ce

        all_ce=all_ce_high + all_ce_low + all_ce_mid

        # print(f" number of real conuter examples is: {sum_real_ce}")
        MF_c=9.16*(high_exp_num)+4.52*(mid_exp_num)+2.02*(low_exp_num)
        new_mf_c=MF_c + 9.16*(all_ce_mid+all_ce_low)
        # print(f" the cost is {MF_c}")
        min_val = y_array.min()

# Safety specification in paper:
# 1. Either the car remains within the initial condition of state and velocity
# 2. Reaches the goal asap
from numpy import mean
import math
from math import pi
import warnings
warnings.filterwarnings('ignore')
min_phi=[]
MFBO_cost=[]
real_num_ce=[]
all_ce_3f=[]
############### specifications for lf
def pred1(trajLf):
    traj = trajLf[0]
    for state in traj:
        last_state = state[0]
    #print(f'last_state ========= {last_state}')
    return 0.1-np.abs(last_state)

def pred2(trajLf):
      traj = trajLf[0]
      for state in traj:
         theta=state[4]
      return (pi/4)-np.abs(theta)

def pred3(trajLf):
      traj =trajLf[0]
      for state in traj:
        d=state[5]
      return 0.2-np.abs(d)

####################Specifications for MF
def pred4(trajMf):
    traj = trajMf[0]
    for state in traj:
        last_state = state[0]
    #print(f'last_state ========= {last_state}')
    return 0.1-np.abs(last_state)

def pred5(trajMf):
      traj = trajMf[0]
      for state in traj:
         theta=state[4]
      return (pi/4)-np.abs(theta)

def pred6(trajMf):
      traj =trajMf[0]
      for state in traj:
        d=state[5]
      return 0.2-np.abs(d)
####################Specifications for hf

def pred7(trajHf):
    traj = trajHf[0]
    for state in traj:
        last_state = state[0]
    #print(f'last_state ========= {last_state}')
    return 0.1-np.abs(last_state)

def pred8(trajHf):
      traj = trajHf[0]
      for state in traj:
         theta=state[4]
      return (pi/4)-np.abs(theta)

def pred9(trajHf):
      traj = trajHf[0]
      for state in traj:
        d=state[5]
      return 0.2-np.abs(d)

########## NOn_Smooth method ############
rand_num = list(range(1, 751))

parser = argparse.ArgumentParser(description='Takes and integer as random seed and runs the code')
parser.add_argument('-r', metavar='N', type=int, help='Index to pick from the rand_num')

args = parser.parse_args()
print("Number of elements in the random seed list %d" % len(rand_num) )
print("The index from random seed list : %d" % args.r)
print("Value picked: %d" % rand_num[args.r])

rand_num2=[rand_num[args.r]]
mf_new_cost=[]

for r in rand_num2:
      np.random.seed(r)
      node0 = pred_node(f=pred3)
      node1 = pred_node(f=pred2)
      node2 = pred_node(f=pred1)
      node4_lf = min_node(children=[node1, node2,node0])

      node0_M = pred_node(f=pred6)
      node1_M = pred_node(f=pred5)
      node2_M = pred_node(f=pred4)
      node4_mf = min_node(children=[node1_M, node2_M,node0_M])

      node0_h = pred_node(f=pred7)
      node1_h = pred_node(f=pred8)
      node2_h = pred_node(f=pred9)
      node4_hf = min_node(children=[node0_h,node1_h,node2_h])
      node=[node4_lf,node4_mf,node4_hf]

      TM_ns = test_module(bounds=bounds,suth=lambda x0: sutH(800,x0),sutm=lambda x0: sutM(400,x0),sutl=lambda x0: sutL(200,x0),
                          f_tree = node,init_sample =33, with_ns=True, exp_weight=2, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(140)
      MFBO_cost.append(MF_c)
      all_ce_3f.append(all_ce)
      real_num_ce.append(sum_real_ce)
      min_phi.append(min_val)
      mf_new_cost.append(new_mf_c)
      print(f"MF all ces: {all_ce_mid}")
      print(f"LF all ces: {all_ce_low}")
      print(f"all validation cost: {mf_new_cost}")
      print(f"this is number of hf runs: {high_exp_num}")
      print(f"this is number of lf runs: {low_exp_num}")
      print(f"this is number of mf runs: {mid_exp_num}")
      print(f"number of ces of HF: {sum_real_h_ce}")
      print(f"number of ces of MF: {sum_real_m_ce}")
      print(f"number of ces of LF: {sum_real_l_ce}")
      print(f" number of valid counterexamples is : {real_num_ce}")
      print(f"cost is {MFBO_cost}")
      print(f"min of phi after 50 BO iterations: {min_phi}")
      print(f"this all counterexamples: {all_ce_3f}")
      print(f" this is minvalue of optimization: {min_phi_obs}")
      print("goodluck")