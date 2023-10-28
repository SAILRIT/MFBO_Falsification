#!/usr/bin/env python3
import time
import argparse
import stable_baselines3
import gym
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')
import emukit
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch
from emukit.model_wrappers import GPyModelWrapper
import numpy as np
from gym import spaces
# !pip uninstall scipy
# !pip install scipy==1.4.1

# Training part
envh = gym.make("MountainCarContinuous-v0")
# envh.env.power = 0.0011

modelppo = PPO.load("PPO_MC")

def compute_trajHf(**kwargs):
    envh.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        envh.env.state = ob
    if 'goal_pos' in kwargs:
        gp = kwargs['goal_pos']
        envh.env.goal_position = gp
    if 'max_speed' in kwargs:
        ms = kwargs['max_speed']
        envh.env.max_speed = ms
        envh.env.low_state = \
            np.array([envh.env.min_position, - envh.env.max_speed])
        envh.env.high_state = \
            np.array([envh.env.max_position, envh.env.max_speed])
        envh.env.observation_space = \
            spaces.Box(envh.env.low_state, envh.env.high_state)
    # if 'power' in kwargs:
    #     pow = kwargs['power']
    #     envh.env.power = pow
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf

    iter_time = 0
    r = 0
    done=False
    trajHf = [ob]
    for _ in range(max_steps):
    # while done==False:
        iter_time += 1
        #action, _ = pi.act(False, ob)
        pi, _ = modelppo.predict(ob)
        action=pi
        ob, reward, terminated, truncated, info=envh.step(action)
        trajHf.append(ob)
        r= reward +r
        done = terminated or iter_time >= max_steps
        if done:
            break
    return trajHf, {'reward':r, 'iter_time': iter_time}

def sutH(x0, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf
    return compute_trajHf(max_steps=max_steps, init_state=x0[0:2],goal_pos=x0[2], max_speed=x0[3])

envl = gym.make("MountainCarContinuous-v0")
envl.env.power = 0.0017
def compute_trajLf(**kwargs):
    envl.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        envl.env.state = ob
    if 'goal_pos' in kwargs:
        gp = kwargs['goal_pos']
        envl.env.goal_position = gp
    if 'max_speed' in kwargs:
        ms = kwargs['max_speed']
        envl.env.max_speed = ms
        envl.env.low_state = \
            np.array([envl.env.min_position, - envl.env.max_speed])
        envl.env.high_state = \
            np.array([envl.env.max_position, envl.env.max_speed])
        envl.env.observation_space = \
            spaces.Box(envl.env.low_state, envl.env.high_state)
    # if 'power' in kwargs:
    #     pow = kwargs['power']
    #     envl.env.power = pow
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf

    iter_time = 0
    r = 0
    done=False
    trajLf = [ob]
    # while done==False:
    for _ in range(max_steps):
        iter_time += 1
        #action, _ = pi.act(False, ob)
        pi, _ = modelppo.predict(ob)
        action=pi
        ob, reward, terminated, truncated, info=envl.step(action)
        # print (ob)
        n_xm = np.random.normal(0.09,0.03,1)
        n_vm=np.random.normal(0.003,0.002,1)
        ob[0]=n_xm+ob[0]
        ob[1]=n_vm+ob[1]
        # print(ob)

        ob[0]=round(ob[0],4)
        ob[1]=round(ob[1],4)

        trajLf.append(ob)
        r= reward +r
        done =terminated or iter_time >= max_steps
        if done:
            break
    return trajLf, {'reward':r, 'iter_time': iter_time}

def sutL(x0, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf
    return compute_trajLf(max_steps=max_steps, init_state=x0[0:2],goal_pos=x0[2], max_speed=x0[3])

########   Utils ##############

def sample_from(count, bounds, sampler=None):
    if sampler is None:
        sampler = lambda num: np.random.random(num)

    sampled_01 = sampler(count*len(bounds))
    sampled_01.resize(count,len(bounds))
    sampled_01 = sampled_01.T
    sampled_lb = [sampled_01[i]*(b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]

    sampled_01_h = sampler(count//2*len(bounds))
    sampled_01_h.resize(count//2,len(bounds))
    sampled_01_h = sampled_01_h.T
    sampled_lb_h = [sampled_01_h[i]*(b[1] - b[0]) + b[0] for i, b in enumerate(bounds)]
    return np.array(sampled_lb).T,np.array(sampled_lb_h).T

bounds = [(-0.6, -0.4)] # Bounds on the position
bounds.append((-0.025, 0.025)) # Bounds on the velocity
bounds.append((0.4, 0.6)) # Bounds on the goal position
bounds.append((0.055, 0.075)) # Bounds on the max speed
# bounds.append((0.0005, 0.0017)) # Bounds on the power magnitude

from emukit.core import ContinuousParameter, ParameterSpace,InformationSourceParameter
#bound = ParameterSpace([ContinuousParameter('p', -0.6,-0.4)])
bound = ParameterSpace([ContinuousParameter('p', -0.6,-0.4), 
           ContinuousParameter('v', -0.025,0.025),
           ContinuousParameter('gp', 0.4,0.6), 
           ContinuousParameter('ms', 0.055,0.075),InformationSourceParameter(2)])

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

low_fidelity_cost = 75
high_fidelity_cost =1619


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
        return self.evalute(x), np.zeros(x.shape)

from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.model_wrappers import GPyMultiOutputWrapper
from GPy.models.gp_regression import GPRegression
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
import copy

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
            self.optimize_restarts = 5


        if 'XL' in kwargs:
            self.XL = kwargs['XL']
        else:
            self.XL = []



        if 'XH' in kwargs:
            self.XH = kwargs['XH']
        else:
            self.XH = []
     
    global XL, XH,YL, YH
    global x_array,y_array, f_index
    def initialize(self):
        global real_low_ce
        global real_high_ce
        global all_ce_high
        global all_ce_low
        global min_phi_obs       
        global low_exp_num,high_exp_num
        low_exp_num=0
        high_exp_num=0
        real_low_ce=0
        real_high_ce=0
        all_ce_low=0
        all_ce_high=0
        min_phi_obs=[]
        global X_ns
        global XL, XH,YL, YH  
        if len(self.XL) == 0:
            XL = sample_from(self.init_sample, self.bounds)[0]
            self.XL = XL
            low_exp_num=self.init_sample

        if len(self.XH) == 0:
            o=self.init_sample//3
            high_exp_num=self.init_sample//3
            XH = np.atleast_2d(np.random.permutation(XL)[:o])
            # XH=XL
            self.XH = XH
        # print(XH)
        global trajsL,trajsH
        global XL_ns, XH_ns,YL, YH
        trajsL = []
        trajsH = []
        # print("This is LF time")
        for x in self.XL:
            start_t=time.time()
            trajsL.append(self.system_under_test_L(x))
            # print(f"trajsL is {trajsL}")
        self.f_acqu=self.f_tree[0]
        YL = self.f_acqu.eval_robustness(trajsL)
        

        # print("This is HF time")
        for x in self.XH:
            start_t=time.time()
            trajsH.append(self.system_under_test_H(x))
            # print(f"trajsH is {trajsH}")
        self.f_acqu=self.f_tree[1]
        YH = self.f_acqu.eval_robustness(trajsH)
        ################------number of counter examples in initial sampling--------------------###########
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
                #  print(f"initial low fidelity ce :{f_xlow}")
        for x in self.XH:
               trajL=self.system_under_test_L(x)
               self.f_acqu=self.f_tree[0]
               f_x_low=self.f_acqu.eval_robustness(trajL)
               self.f_acqu=self.f_tree[1]
               traj_H=self.system_under_test_H(x)
               f_x_high=self.f_acqu.eval_robustness(traj_H)
               min_phi_obs.append(f_x_high)               
               if (f_x_high<0) and (f_x_low>0):
                 real_high_ce=1+real_high_ce
                 all_ce_high=all_ce_high+1
                #  print(f"initial high fidelity ce :{f_x_high}")
        
        global x_array,y_array
        # print(f"this is low-F: {XL}")
        # print(f"this is high-F: {XH}")
        x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
        
    global XL, XH,YL, YH
    global y_array
    global low_exp_num,high_exp_num
    global real_high_ce,real_low_ce

    def run_BO(self, iters_BO):
        for ib in range(iters_BO):
            global XL, XH,YL, YH
            global low_exp_num,high_exp_num
            global real_high_ce,real_low_ce
            global all_ce_high
            global all_ce_low
            # global min_phi_obs  
            print('BO iteration:', ib)
            global x_array,y_array
            kern_low = GPy.kern.RBF(4,ARD=True)
            # kern_low.lengthscale.constrain_bounded(0.01, 0.5)
            kern_err = GPy.kern.RBF(4,ARD=True)
            # kern_err.lengthscale.constrain_bounded(0.01, 0.5)
            
            multi_fidelity_kernel = LinearMultiFidelityKernel([kern_low, kern_err])
            gpy_model = GPyLinearMultiFidelityModel(x_array, y_array, multi_fidelity_kernel, 2,None)
            gpy_model.mixed_noise.Gaussian_noise.fix(0.001)
            gpy_model.mixed_noise.Gaussian_noise_1.fix(0.001)
            gpy_model.optimize()
            # GPmodel
            GPmodel = GPyMultiOutputWrapper(gpy_model, 2, 1, verbose_optimization=True)
            cost_acquisition = Cost([low_fidelity_cost, high_fidelity_cost])
            acquisition = MultiInformationSourceEntropySearch(GPmodel, bound) / cost_acquisition
            acquisition_optimizer=MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(bound), bound)
            new_x,val_acq=acquisition_optimizer.optimize(acquisition,None)
            # print(new_x)
            global low_exp_num,high_exp_num
            if new_x[0][-1]==0.:
               print("This is low-fidelity")
               global low_exp_num,high_exp_num
               x=new_x[0][0:4]
               low_exp_num =1 +low_exp_num 
               X_L=XL 
               XL=np.vstack((X_L, x))
               trajsL=self.system_under_test_L(x)
               self.f_acqu=self.f_tree[0]
               f_xl=self.f_acqu.eval_robustness(trajsL)
               min_phi_obs.append(f_xl)
               if f_xl<0:
                 all_ce_low=all_ce_low+1               
               self.f_acqu=self.f_tree[1]
               trajsH=self.system_under_test_H(x)
               f_test_ce=self.f_acqu.eval_robustness(trajsH)
               Y_L=YL
               if (f_xl<0) and (f_test_ce<0):
                  print("It's a valid counterexample")
                  real_low_ce=1+ real_low_ce
               YL=np.vstack((Y_L, f_xl))
               x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])

            elif new_x[0][-1]==1.:
               a=new_x[0][0:4]
               print("This is high-fidelity")
               X_H=XH
               XH=np.vstack((X_H, a))
               high_exp_num =1+high_exp_num
               trajsH=self.system_under_test_H(a)
               self.f_acqu=self.f_tree[1]
               f_xh=self.f_acqu.eval_robustness(trajsH)
               min_phi_obs.append(f_xh)               
               Y_H=YH
               if (f_xh<0):
                 real_high_ce=1+real_high_ce
                 all_ce_high=all_ce_high+1
               YH=np.vstack((Y_H, f_xh))
               
               x_array, y_array = convert_xy_lists_to_arrays([XL, XH], [YL, YH])
        
        global min_val
        global MF_c
        global sum_real_ce
        sum_real_ce=0
        sum_real_ce=real_high_ce+real_low_ce
        global all_ce

        all_ce=all_ce_high+all_ce_low
        # print(f" number of real conuter examples is: {sum_real_ce}")
        MF_c=161.9*(high_exp_num)+7.52*(low_exp_num)
        # print(f" the cost is {MF_c}")
        min_val = y_array.min()
nums=[]
min_phi=[]
MFBO_cost=[]
real_num_ce=[]
all_ce_2f=[]
############### specifications for lf
def pred1(trajLf):

    trajLf = trajLf[0]
    x_s = np.array(trajLf).T[0]
    up = min(-0.4 - x_s)
    low = min(x_s + 0.6)
    return min(up,low)

def pred2(trajLf):
    iters = trajLf[1]['iter_time']
    return -iters/250.

def pred3(trajLf):
    trajLf=trajLf[0]
    v_s = np.array(trajLf).T[1]
    return min(0.02 - np.abs(v_s))
##############################################Specifications for hf
def pred4(trajHf):
    trajLf = trajHf[0]
    x_s = np.array(trajHf).T[0]
    up = min(-0.4 - x_s)
    low = min(x_s + 0.6)
    return min(up,low)
   

def pred5(trajHf):
    iters = trajHf[1]['iter_time']
    return -iters/350.

def pred6(trajHf):
    trajHf=trajHf[0]
    v_s = np.array(trajHf).T[1]
    return min(0.02 - np.abs(v_s))

########## NOn_Smooth method ############
rand_num=[75, 29371, 2946276, 1238808, 129519, 968526, 4712957, 1495789, 4424084, 305169, 123095, 4362912, 618681, 426568, 4318216, 3420140, 4376872, 445558, 639664, 2573739, 1697528, 4280774, 1159611, 312704, 281844, 1575098, 3233622, 1542407, 4054422, 4742535, 1818570, 2746352, 478027, 4649798, 2681668, 1081513, 1835505, 506430, 4204609, 1163602, 455678, 3972889, 4271006, 3231785, 4502324, 1406983, 822040, 3947416, 1419252, 4258678, 4861650, 3266363, 4051878, 432617, 1811568, 3219384, 542721, 10876, 4738663, 1586398, 1019791, 1484715, 4257438, 3441514, 2796034, 1505731, 1454526, 1155004, 2013356, 2650683, 1890670, 4954160, 1120676, 1927071, 865123, 2112185, 1025842, 2000204, 3054922, 4333539, 4601199, 4350871, 3883109, 1262734, 4318961, 281688, 4570134, 2334354, 3741087, 3966315, 4220896, 2101102, 1945892, 1528275, 1639211, 1321534, 867633, 3741408, 635068, 2801483, 654136, 3578880, 2748637, 6383, 186152, 4940048, 287730, 32312, 4051798, 1454602, 2717920, 1849901, 3687303, 478993, 2104806, 4898772, 3339832, 2433012, 4783725, 990744, 4212376, 3417468, 4841428, 3191654, 1915990, 1356266, 2131290, 4864184, 2570743, 35843, 1793615, 4275349, 62181, 2744752, 1518368, 169270, 3947661, 805986, 3823919, 777249, 2324581, 2100703, 2203392, 3759242, 408554, 4157409, 3900738, 4477156, 372741, 1809129, 3133040, 1701520, 2578858, 2520038, 2064326, 1589454, 2499389, 4725330, 4615177, 1916336, 2269194, 4255552, 3409092, 3730567, 95397, 2415878, 3073522, 156900, 98648, 1552033, 4621042, 4134387, 3882716, 260826, 2875079, 4868879, 3561294, 3269287, 1373098, 3227621, 1117100, 4132397, 4598477, 1891712, 2209768, 1552776, 1277399, 2016085, 2004447, 2584097, 95383, 2371357, 3906078, 3708807, 586469, 4246894, 1621233, 1682636, 1602637, 4494482, 3411518, 3561744, 275471, 988747, 3291909, 2308068, 4520345, 3584080, 1755221, 3619548, 3435429, 1638136, 4980539, 1503112, 3325157, 4307667, 3006330, 54936, 3128916, 1898693, 2774785, 113184, 3606963, 4373709, 4920974, 4090211, 4365012, 1910438, 4630947, 1313338, 4966574, 2059262, 4902290, 3203337, 457692, 4105195, 1711790, 3472411, 3425340, 643980, 4724389, 252729, 3917747, 4436077, 2039837, 2874662, 1648134, 2004029, 4986903, 2353103, 3822432, 892997, 263966, 4707916, 656621, 4547779, 2033323, 4519390, 4908670, 3316318, 3311564, 4903589, 3603778, 4402637, 2682204, 647693, 2602352, 4770686, 4558878, 3361771, 1580594, 1764284, 2317998, 1351370, 1092947, 4785183, 4840855, 2555207, 985069, 1323258, 2075252, 4052424, 737071, 4462651, 447775, 4516944, 1080467, 2348243, 1447577, 2335854, 1368960, 3494435, 3084457, 4337770, 990633, 1929967, 1184840, 3671016, 2089345, 3134789,19052, 91277, 6326, 93037, 3623, 7725, 69997, 6384, 89300, 57887, 97079, 3287, 93421, 25131, 65974, 50835, 2275, 78204, 47062, 26457, 94943, 10844, 78089, 31019, 63046, 17183, 12590, 30057, 11331, 74241, 55150, 12888, 72359, 33166, 48312, 28541, 25566, 15614, 34640, 92971, 83399, 58763, 78293, 78386, 59004, 9408, 95736, 92938, 79486, 84964, 52908, 43499, 64924, 74408, 82151, 35734, 51045, 42664, 16269, 22259, 18356, 25365, 68338, 85655, 74658, 4810, 74654, 98075, 35758, 34210, 82474, 85605, 72556, 66353, 15110, 69895, 3423, 50450, 35100, 62887, 75855, 55942, 33420, 27027, 62291, 62226, 62243, 75947, 64370, 10562, 30644, 8249, 5262, 87950, 80498, 19437, 18308, 40157, 57718, 13500, 34618, 66247, 4636, 45783, 16059, 18309, 19990, 82558, 98720, 97577, 15293, 97035, 92721, 94056, 62991, 50067, 38671, 30981, 87322, 16028, 97794, 83658, 55555, 38311, 45479, 23542, 51296, 82169, 63647, 23877, 65029, 14756, 43287, 12596, 23643, 61889, 72645, 43873, 19104, 88339, 63188, 77169, 47267, 50006, 98805, 91937, 64005, 50069, 13873, 38501, 54728, 55245, 7613, 88672, 27899, 6175, 42230, 37147, 64929, 98092, 30616, 11187, 26738, 43080, 2244, 72804, 92054, 36207, 74980, 81772, 87340, 29443, 5805, 4439, 86843, 90815, 30404, 54855, 55260, 69937, 43236, 7939, 2219, 77061, 76590, 79365, 77538, 43809, 30102, 5989, 87117, 90424, 23338, 44005, 82122, 43739, 3036, 15217, 30286, 86206]

# parser = argparse.ArgumentParser(description='Takes and integer as random seed and runs the code')
# parser.add_argument('-r', metavar='N', type=int, help='Index to pick from the rand_num')

# args = parser.parse_args()
# print("Number of elements in the random seed list %d" % len(rand_num) )
# print("The index from random seed list : %d" % args.r)
# print("Value picked: %d" % rand_num[args.r])

# rand_num2=[rand_num[args.r]]

for r in rand_num[0:1]:
      np.random.seed(r)
      node0_ns = pred_node(f=pred1)
      node1_ns = pred_node(f=pred2)
      node2_ns = pred_node(f=pred3)
      node3_ns = min_node(children=[node0_ns, node2_ns])
      node4_lf = max_node(children=[node3_ns, node1_ns])


      node0 = pred_node(f=pred4)
      node1 = pred_node(f=pred5)
      node2 = pred_node(f=pred6)
      node3 = min_node(children=[node0_ns, node2_ns])
      node4_hf = max_node(children=[node3_ns, node1_ns])
      node=[node4_lf,node4_hf]

      TM_ns = test_module(bounds=bounds,suth=lambda x0: sutH(x0,max_steps=350),sutl=lambda x0: sutL(x0,max_steps=250), 
                          f_tree = node,init_sample =33, with_ns=True, optimize_restarts=1, exp_weight=5, normalizer=True)
      #cost_model=cost_func has been removed
      # exploration weight=2
      TM_ns.initialize()
      TM_ns.run_BO(66)
      MFBO_cost.append(MF_c)
      real_num_ce.append(sum_real_ce)
      min_phi.append(min_val)
      all_ce_2f.append(all_ce)      
      print(f"this all counterexamples: {all_ce_2f}")
      print(f" number of valid counterexamples is : {real_num_ce}")
      print(f"cost is {MFBO_cost}")
      print(f"min of phi after 50 BO iterations: {min_phi}")
      print(f" this is minvalue of optimization: {min_phi_obs}")