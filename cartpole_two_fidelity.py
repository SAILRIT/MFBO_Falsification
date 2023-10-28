#!/usr/bin/env python3
import time
import argparse
import stable_baselines3
import gym
from stable_baselines3 import PPO
import emukit
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch
from emukit.model_wrappers import GPyModelWrapper
from emukit.core import ContinuousParameter, ParameterSpace,InformationSourceParameter
import numpy as np
from gym import spaces
import math

# !pip uninstall scipy
# !pip install scipy==1.4.1
# import cartpole
# from cartpole import CartPoleEnv

envh = gym.make("CartPole-v1")
envh.env.kinematics_integrator = "semi-implicit euler"
envh.env.force_mag=20

modelppo = PPO.load("PPO_CP.zip")

def compute_trajHf(max_steps, **kwargs):
    envh.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        envh.env.state = ob
    if 'masspole' in kwargs:
        envh.env.masspole = kwargs['masspole']
        envh.env.total_mass = envh.env.masspole + envh.env.masscart
        envh.env.polemass_length = envh.env.masspole * envh.env.length
    if 'length' in kwargs:
        envh.env.length = kwargs['length']
        envh.env.polemass_length = envh.env.masspole * envh.env.length
    # if 'force_mag' in kwargs:
    #     envh.env.force_mag = kwargs['force_mag']
    trajHf = [ob]
    reward = 0
    iter_time = 0
    done=False
    while done==False:
    # for _ in range(max_steps):
        iter_time += 1
        pi, _ = modelppo.predict(ob)
        action=pi
        ob, r, terminated, truncated, info=envh.step(action)
        reward = reward+r
        trajHf.append(ob)

        done = terminated or iter_time >= max_steps
        if done:
            break
    additional_data = {'reward':reward, 'mass':envh.env.total_mass}
    return trajHf, additional_data

def sutH(max_steps,x0):
    return compute_trajHf(max_steps,init_state=x0[0:4], masspole=x0[4],length=x0[5])

envl = gym.make("CartPole-v1")
envl.env.force_mag=10

def compute_trajLf(max_steps, **kwargs):
    envl.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        envl.env.state = ob
    if 'masspole' in kwargs:
        envl.env.masspole = kwargs['masspole']
        envl.env.total_mass = envl.env.masspole + envl.env.masscart
        envl.env.polemass_length = envl.env.masspole * envl.env.length
    if 'length' in kwargs:
        envl.env.length = kwargs['length']
        envl.env.polemass_length = envl.env.masspole * envl.env.length
    # if 'force_mag' in kwargs:
    #     envl.env.force_mag = kwargs['force_mag']

        iter_time = 0
        reward = 0
        done=False
        trajLf = [ob]
        # for _ in range(max_steps):
        while done==False:
            iter_time += 1
            pi, _ = modelppo.predict(ob)
            action=pi
            # ob, r, done, _ = envl.step(action)
            ob, r, terminated, truncated, info=envl.step(action)
            reward = reward+r
            noise_x = np.random.normal(0,0.25,1)
            pi_n = math.pi
            noise_theta=np.random.normal(0,0.015*pi_n,1)
            ob[0]+=noise_x
            ob[2]+=noise_theta
            ob[0]=round(ob[0],2)
            ob[1]=round(ob[1],2)
            ob[2]=round(ob[2],2)
            ob[3]=round(ob[3],2)
            trajLf.append(ob)
            done = terminated or iter_time >= max_steps
            if done:
                break
        additional_data = {'reward':reward, 'mass':envl.env.total_mass}
        return trajLf, additional_data

def sutL(max_steps,x0):
    return compute_trajLf(max_steps,init_state=x0[0:4], masspole=x0[4],length=x0[5])

######## - - - - - --- - -   Utils ----- ---  - - - - ##############
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

# bound = ParameterSpace([ContinuousParameter('x', -2,2),
#            ContinuousParameter('vx', -0.05,0.05),
#            ContinuousParameter('theta',-0.2,0.2), 
#            ContinuousParameter('d_theta', -0.05,0.05),
#            ContinuousParameter('mass', 0.05,0.15),
#            ContinuousParameter('len', 0.4,0.6),
#            ContinuousParameter('mag', 18.00,22.00),InformationSourceParameter(2)])
bound = ParameterSpace([ContinuousParameter('x', -2,2),
           ContinuousParameter('vx', -0.05,0.05),
           ContinuousParameter('theta',-0.2,0.2), 
           ContinuousParameter('d_theta', -0.05,0.05),
           ContinuousParameter('mass', 0.05,0.15),
           ContinuousParameter('len', 0.4,0.6),InformationSourceParameter(2)])

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
        trajs = np.atleast_2d(trajs)
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
low_fidelity_cost = 271
high_fidelity_cost = 5641

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

        for x in self.XL:
               trajL=self.system_under_test_L(x)
               self.f_acqu=self.f_tree[0]
               f_xlow=self.f_acqu.eval_robustness(trajL)
               min_phi_obs.append(f_xlow)
               if (f_xlow<0):
                 all_ce_low=all_ce_low+1
               self.f_acqu=self.f_tree[1]
               trajH=self.system_under_test_H(x)
               f_xhigh=self.f_acqu.eval_robustness(trajH)
               if (f_xlow<0) and (f_xhigh<0):
                 real_low_ce=1+real_low_ce

        for x in self.XH:
               self.f_acqu=self.f_tree[1]
               traj_H=self.system_under_test_H(x)
               f_x_high=self.f_acqu.eval_robustness(traj_H)
               f_x_high=self.f_acqu.eval_robustness(traj_H)
               min_phi_obs.append(f_x_high)
               traj_L=self.system_under_test_L(x)
               self.f_acqu=self.f_tree[0]
               f_x_low=self.f_acqu.eval_robustness(traj_L)
               if (f_x_low>0) and (f_x_high<0):
                 real_high_ce=1+real_high_ce
                 all_ce_high=all_ce_high + 1


        global x_array,y_array
        x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
       
    global XL, XH,YL, YH
    global y_array

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
            print('BO iteration:', ib)
            global x_array,y_array
            kern_low = GPy.kern.RBF(6,ARD=True)
            #kern_low.lengthscale.constrain_bounded(0.01, 0.5)
            kern_err = GPy.kern.RBF(6,ARD=True)
            #kern_err.lengthscale.constrain_bounded(0.01, 0.5)
            multi_fidelity_kernel = LinearMultiFidelityKernel([kern_low, kern_err])
            gpy_model = GPyLinearMultiFidelityModel(x_array, y_array, multi_fidelity_kernel, 2,None)
            gpy_model.mixed_noise.Gaussian_noise.fix(0.000001)
            gpy_model.mixed_noise.Gaussian_noise_1.fix(0.000001)
            GPmodel = GPyMultiOutputWrapper(gpy_model, 2, 1, verbose_optimization=True)
            GPmodel.optimize()
            cost_acquisition = Cost([low_fidelity_cost, high_fidelity_cost])
            acquisition = MultiInformationSourceEntropySearch(GPmodel, bound) / cost_acquisition
            acquisition_optimizer=MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(bound), bound)
            new_x,val_acq=acquisition_optimizer.optimize(acquisition)
            # print(new_x)
            
            if new_x[0][-1]==0.:
               print("This is low-fidelity") 
               x=new_x[0][0:6] 
               X_L=XL 
               XL=np.vstack((X_L, x))
               low_exp_num=1+low_exp_num
               trajsL=self.system_under_test_L(x)
               self.f_acqu=self.f_tree[0]
               f_xl=self.f_acqu.eval_robustness(trajsL)
               min_phi_obs.append(f_xl)
               if f_xl<0:
                 all_ce_low=all_ce_low+1
               self.f_acqu=self.f_tree[1]
               trajsH=self.system_under_test_H(x)
               f_test_ce=self.f_acqu.eval_robustness(trajsH)
               if (f_xl<0) and (f_test_ce<0):
                 print("It's a valid counterexample")
                 valid_low_ce=1+valid_low_ce
               #print(f"f_xl= {f_xl}")
               Y_L=YL
               YL=np.vstack((Y_L, f_xl))    
               x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
            else:
               a=new_x[0][0:6]
               print("This is high-fidelity") 
               X_H=XH
               XH=np.vstack((X_H, a))
               high_exp_num =1+high_exp_num
               trajsH=self.system_under_test_H(a)
               self.f_acqu=self.f_tree[1]
               f_xh=self.f_acqu.eval_robustness(trajsH)
               min_phi_obs.append(f_xh)
               all_ce_high=all_ce_high + 1
               if f_xh<0:
                  valid_high_ce=1+valid_high_ce
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
        min_val = y_array.min()
        sume_real_h_ce=(valid_high_ce)+(real_high_ce)
        sum_real_ce=(valid_high_ce)+(valid_low_ce)+(real_high_ce)+(real_low_ce)
        global all_ce
        all_ce=all_ce_high+all_ce_low
        MF_c=56.41*(high_exp_num)+2.71*(low_exp_num)

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

############### specifications for lf
def pred2(trajLf):
    traj_ = trajLf[0]
    mass = trajLf[1]['mass']
    v_s = np.array(traj_).T[1]
    return min(1 - np.abs(mass*v_s))

def pred3(trajLf):
    traj=trajLf[0]
    theta=np.array(traj).T[2]
    return min(0.157 - np.abs(theta))

def pred4(trajLf):
    traj=trajLf[0]
    x_s = np.array(traj).T[0]
    return min(1 - np.abs(x_s))
####################Specifications for hf
def pred5(trajHf):
    traj_ = trajHf[0]
    mass = trajHf[1]['mass']
    v_s = np.array(traj_).T[1]
    return min(1 - np.abs(mass*v_s))
   

def pred6(trajHf):
    traj=trajHf[0]
    theta=np.array(traj).T[2]
    return min(0.157 - np.abs(theta))

def pred7(trajHf):
    traj=trajHf[0]
    x_s = np.array(traj).T[0]
    return min(1 - np.abs(x_s))

########## NOn_Smooth method ############
#####rand_num=[479359, 3786994, 5481201, 4071722, 5835020, 7657249, 5042017, 1652934, 769047, 8148169, 4981496, 1313397, 8084240, 637072, 206367, 3312771, 1598542, 3679713, 2671523, 2852110, 5215106, 163720, 3854956, 499245, 6284359, 715456, 4026284, 254424, 4998063, 3404367, 752015, 8053736, 1143640, 4755279, 3928661, 1963690, 9917347, 4141578, 8898752, 1030191, 6925304, 2721285, 193090, 7277555, 9203760, 2573754, 7591557, 5062511, 8127290, 9080893, 3680442, 7643592, 9564781, 3494973, 4480626, 2841177, 7551223, 1340475, 7210384, 5023283, 6433320, 8746900, 8676207, 1733610, 6973115, 3127908, 7584720, 5603415, 8464209, 7674101, 993926, 4119494, 7172135, 2998587, 7334394, 8712555, 2162903, 1162402, 7865528, 8565067, 9735101, 8645054, 6179863, 5330074, 2856147, 5848034, 2187439, 7958020, 1332020, 6038921, 2517015, 9673098, 710943, 7565010, 2974425, 6655503, 5125780, 3787084, 5950652, 6008005, 1532944, 8586709, 471050, 8935740, 7576614, 1008070, 3559284, 7698377, 7390126, 6096987, 6911848, 2997461, 5305194, 515240, 2333903, 6990388, 1460750, 1948951, 8594846, 6485600, 886483, 9456530, 2521821, 3495235, 9643328, 7916779, 9114303, 119985, 158796, 773501, 2648596, 8341187, 8659666, 2431803, 491351, 1881700, 3193191, 4008480, 4637133, 4006738, 2450607, 6384370, 8329989, 7543136, 9713438, 6222888, 4539439, 9211945, 567462, 1030463, 5273893, 2931059, 2511667, 4416649, 1616650, 9547318, 9280204, 1381385, 6628588, 8664913, 7364206, 750620, 92828, 4983741, 4759616, 5172439, 294445, 8957495, 959415, 4935409, 9733796, 6925471, 3363191, 7092421, 3050287, 5238619, 4426282, 4319957, 5805850, 8759746, 447465, 1825521, 6309446, 3650698, 2543327, 6777968, 167472, 7742785, 9125847, 1882952, 7856533, 6223969, 8841581, 6171567, 960190, 552447, 2031373, 8038120, 4053983, 4054370, 9657373, 5361767, 4547214, 8018470, 6024014, 6534529, 958088, 4448155, 687590, 5408564]
####rand_num=[75485, 14384, 97099, 41166, 60097, 54622, 60660, 74172, 12922, 58625, 95057, 21039, 99732, 28242, 99032, 56523, 23046, 40994, 96495, 46950, 84826, 38959, 88098, 63638, 41078, 19315, 90626, 93319, 53898, 28958, 22120, 81389, 83214, 38094, 51756, 14627, 76394, 82143, 60059, 5572, 60112, 53931, 93502, 4750, 82449, 58063, 31927, 96971, 10399, 63184, 98777, 67161, 20633, 87217, 81007, 33798, 36779, 32647, 18627, 24514, 30039, 17078, 30701, 28609, 8393, 31187, 43409, 69401, 27473, 74980, 5077, 8800, 66233, 25006, 9937, 99534, 18269, 15273, 75752, 56972, 34924, 35912, 47609, 7930, 15386, 11804, 74529, 58978, 97844, 56139, 55917, 61882, 30354, 4530, 27174, 73014, 83112, 34152, 51353, 59109, 15327, 74007, 72333, 33577, 38599, 46968, 25915, 78038, 76246, 61710, 31215, 19173, 20160, 21316, 5957, 33769, 34095, 66969, 49700, 20959, 71335, 48479, 25021, 31676, 4829, 15815, 48322, 51510, 21167, 90595, 66864, 90129, 81597, 30198, 65058, 36499, 94451, 73915, 98950, 25906, 297, 79995, 16629, 34073, 86410, 29959, 33285, 71325, 13256, 84755]
rand_num=[3969675, 1129371, 2946276, 1238808, 129519, 968526, 4712957, 1495789, 4424084, 305169, 123095, 4362912, 618681, 426568, 4318216, 3420140, 4376872, 445558, 639664, 2573739, 1697528, 4280774, 1159611, 312704, 281844, 1575098, 3233622, 1542407, 4054422, 4742535, 1818570, 2746352, 478027, 4649798, 2681668, 1081513, 1835505, 506430, 4204609, 1163602, 455678, 3972889, 4271006, 3231785, 4502324, 1406983, 822040, 3947416, 1419252, 4258678, 4861650, 3266363, 4051878, 432617, 1811568, 3219384, 542721, 10876, 4738663, 1586398, 1019791, 1484715, 4257438, 3441514, 2796034, 1505731, 1454526, 1155004, 2013356, 2650683, 1890670, 4954160, 1120676, 1927071, 865123, 2112185, 1025842, 2000204, 3054922, 4333539, 4601199, 4350871, 3883109, 1262734, 4318961, 281688, 4570134, 2334354, 3741087, 3966315, 4220896, 2101102, 1945892, 1528275, 1639211, 1321534, 867633, 3741408, 635068, 2801483, 654136, 3578880, 2748637, 6383, 186152, 4940048, 287730, 32312, 4051798, 1454602, 2717920, 1849901, 3687303, 478993, 2104806, 4898772, 3339832, 2433012, 4783725, 990744, 4212376, 3417468, 4841428, 3191654, 1915990, 1356266, 2131290, 4864184, 2570743, 35843, 1793615, 4275349, 62181, 2744752, 1518368, 169270, 3947661, 805986, 3823919, 777249, 2324581, 2100703, 2203392, 3759242, 408554, 4157409, 3900738, 4477156, 372741, 1809129, 3133040, 1701520, 2578858, 2520038, 2064326, 1589454, 2499389, 4725330, 4615177, 1916336, 2269194, 4255552, 3409092, 3730567, 95397, 2415878, 3073522, 156900, 98648, 1552033, 4621042, 4134387, 3882716, 260826, 2875079, 4868879, 3561294, 3269287, 1373098, 3227621, 1117100, 4132397, 4598477, 1891712, 2209768, 1552776, 1277399, 2016085, 2004447, 2584097, 95383, 2371357, 3906078, 3708807, 586469, 4246894, 1621233, 1682636, 1602637, 4494482, 3411518, 3561744, 275471, 988747, 3291909, 2308068, 4520345, 3584080, 1755221, 3619548, 3435429, 1638136, 4980539, 1503112, 3325157, 4307667, 3006330, 54936, 3128916, 1898693, 2774785, 113184, 3606963, 4373709, 4920974, 4090211, 4365012, 1910438, 4630947, 1313338, 4966574, 2059262, 4902290, 3203337, 457692, 4105195, 1711790, 3472411, 3425340, 643980, 4724389, 252729, 3917747, 4436077, 2039837, 2874662, 1648134, 2004029, 4986903, 2353103, 3822432, 892997, 263966, 4707916, 656621, 4547779, 2033323, 4519390, 4908670, 3316318, 3311564, 4903589, 3603778, 4402637, 2682204, 647693, 2602352, 4770686, 4558878, 3361771, 1580594, 1764284, 2317998, 1351370, 1092947, 4785183, 4840855, 2555207, 985069, 1323258, 2075252, 4052424, 737071, 4462651, 447775, 4516944, 1080467, 2348243, 1447577, 2335854, 1368960, 3494435, 3084457, 4337770, 990633, 1929967, 1184840, 3671016, 2089345, 3134789]

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
      node1 = pred_node(f=lambda trajLf: pred2(trajLf))
      node2 = pred_node(f=lambda trajLf: pred3(trajLf))
      node3 = pred_node(f=lambda trajLf: pred4(trajLf))
      node4_lf = max_node(children= [node1, node3, node2])

      node1_h = pred_node(f=lambda trajHf: pred2(trajHf))
      node2_h = pred_node(f=lambda trajHf: pred3(trajHf))
      node3_h = pred_node(f=lambda trajHf: pred4(trajHf))
      node4_hf = max_node(children= [node1_h, node3_h, node2_h])
      node=[node4_lf,node4_hf]

      TM_ns = test_module(bounds=bounds,suth=lambda x0: sutH(450,x0), sutl=lambda x0: sutL(150,x0), 
                          f_tree = node,init_sample =33, with_ns=True, exp_weight=2, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(66)
      min_phi.append(min_val)
      print(f"min of phi after * BO iterations: {min_phi}")
      MFBO_cost.append(MF_c)
      all_ce_2f.append(all_ce)
      real_num_ce.append(sum_real_ce)
      print(f" number of valid counterexamples are : {real_num_ce}")
      print(f"cost is {MFBO_cost}")
      print(f"this all counterexamples: {all_ce_2f}")
      print(f" this is minvalue of optimization: {min_phi_obs}")
