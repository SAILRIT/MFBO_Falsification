
# !pip install stable-baselines3[extra]
# !pip install gymnasium==0.28.1
# !pip install stable_baselines3
# !pip install emukit

# !pip install gym
# !pip install GPy

#!/usr/bin/env python3
import time
import argparse
import stable_baselines3
import gym
from stable_baselines3 import PPO
import emukit
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch

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
        # ob, r, terminated, _=envh.step(action)
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
            # ob, r, terminated, _=envl.step(action)
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
        trajs_L = []
        trajs_H = []
        # print("This is L/F time")
        for x in self.XL:
            # start_t=time.time()
            trajs_L.append(self.system_under_test_L(x))

        self.f_acqu=self.f_tree[0]
        YL = self.f_acqu.eval_robustness(trajs_L)
        # print("This is HF time")
        for x in self.XH:
            # start_t=time.time()
            trajs_H.append(self.system_under_test_H(x))

        self.f_acqu=self.f_tree[1]
        YH = self.f_acqu.eval_robustness(trajs_H)
        low_exp_num=self.init_sample
        high_exp_num=self.init_sample//3
        traL=[]
        traH=[]

        for x in self.XL:
               traL.append(self.system_under_test_L(x))
               traH.append(self.system_under_test_H(x))
        self.f_acqu=self.f_tree[0]
        f_xlow=self.f_acqu.eval_robustness(traL)
        min_phi_obs.append(f_xlow)
        all_ce_low=all_ce_low+np.sum(f_xlow< 0)
        self.f_acqu=self.f_tree[1]
        f_xhigh=self.f_acqu.eval_robustness(traH)
        for fl, fh in zip(f_xlow, f_xhigh):
    # Iterate over elements in the row
              for i in range(len(fl)):
                if fl[i] < 0 and fh[i] < 0:  # Check corresponding elements
                    real_low_ce += 1
        trL=[]
        trH=[]

        for x in self.XH:

               trH.append(self.system_under_test_H(x))

        self.f_acqu=self.f_tree[1]
        f_x_high=self.f_acqu.eval_robustness(trH)
        min_phi_obs.append(f_x_high)
        all_ce_high=all_ce_high+np.sum(f_x_high< 0)
        real_high_ce = real_high_ce+np.sum(f_x_high< 0)


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
            tr_L=[]
            tr_H=[]
            th=[]

            if new_x[0][-1]==0.:
               print("This is low-fidelity")
               x=new_x[0][0:6]
               X_L=XL
               XL=np.vstack((X_L, x))
               low_exp_num=1+low_exp_num
               tr_L.append(self.system_under_test_L(x))
               self.f_acqu=self.f_tree[0]
               f_xl=self.f_acqu.eval_robustness(tr_L)
               min_phi_obs.append(f_xl)
               if f_xl<0:
                 all_ce_low=all_ce_low+1
               self.f_acqu=self.f_tree[1]
               tr_H.append(self.system_under_test_H(x))
               f_test_ce=self.f_acqu.eval_robustness(tr_H)
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
               th.append(self.system_under_test_H(a))
               self.f_acqu=self.f_tree[1]
               f_xh=self.f_acqu.eval_robustness(th)
               min_phi_obs.append(f_xh)

               if f_xh<0:
                  valid_high_ce=1+valid_high_ce
                  all_ce_high=all_ce_high + 1
               #print(f"f_xh= {f_xh}")
               Y_H=YH
               YH=np.vstack((Y_H, f_xh))
               x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
        global n
        n=0
        global sume_real_h_ce
        global sume_real_l_ce
        sume_real_l_ce=0
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
        sume_real_l_ce=(valid_low_ce)+(real_low_ce)
        sum_real_ce=(valid_high_ce)+(valid_low_ce)+(real_high_ce)+(real_low_ce)
        global all_ce
        all_ce=all_ce_high+all_ce_low
        MF_c=56.41*(high_exp_num)+2.71*(low_exp_num)
        new_mf_c=MF_c + 56.41*(all_ce_low)

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

rand_num = list(range(1, 751))
#########################################
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
      node1 = pred_node(f=lambda trajLf: pred2(trajLf))
      node2 = pred_node(f=lambda trajLf: pred3(trajLf))
      node3 = pred_node(f=lambda trajLf: pred4(trajLf))
      node4_lf = max_node(children= [node1, node3, node2])

      node1_h = pred_node(f=lambda trajHf: pred5(trajHf))
      node2_h = pred_node(f=lambda trajHf: pred6(trajHf))
      node3_h = pred_node(f=lambda trajHf: pred7(trajHf))
      node4_hf = max_node(children= [node1_h, node3_h, node2_h])
      node=[node4_lf,node4_hf]

      TM_ns = test_module(bounds=bounds,suth=lambda x0: sutH(450,x0), sutl=lambda x0: sutL(150,x0),
                          f_tree = node,init_sample =45, with_ns=True, exp_weight=2, normalizer=True)
      TM_ns.initialize()
      TM_ns.run_BO(140)
      min_phi.append(min_val)
      mf_new_cost.append(new_mf_c)
      MFBO_cost.append(MF_c)
      all_ce_2f.append(all_ce)
      real_num_ce.append(sum_real_ce)
      print(f"all validation cost: {mf_new_cost}")
      print(f"min of phi after * BO iterations: {min_phi}")
      print(f"number of hf runs: {high_exp_num}")
      print(f"number of Lf runs: {low_exp_num}")
      print(f"number of valid ces on HF: {sume_real_h_ce}")
      print(f"number of valid ces on LF: {sume_real_l_ce}")
      print(f"number of all low ces: {all_ce_low}")

      print(f" number of valid counterexamples are : {real_num_ce}")
      print(f"cost is {MFBO_cost}")
      print(f"this all counterexamples: {all_ce_2f}")
      print(f" this is minvalue of optimization: {min_phi_obs}")