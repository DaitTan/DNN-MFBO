import numpy as np
import matplotlib.pyplot as plt
import random
import string
import os
import subprocess

from sklearn.preprocessing import StandardScaler

from hdf5storage import loadmat
from hdf5storage import savemat


class CurrinExp:
    def __init__(self, debug=False):
        self.dim = 2
        self.flevels = 2
        self.maximum = 13.798702307261388
        
        self.bounds = ((0.0,1.0), (0.0,1.0))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
   
        self.Flist = []
        self.Flist.append(self.eval_fed_L0)
        self.Flist.append(self.eval_fed_L1)
        
        
    def query(self, X, m):

        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)


        return ym

    def eval_fed_L0(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        args1 = np.array([x1+0.05, x2+0.05])
        args2 = np.array([x1+0.05, np.max(np.array([0, x2-0.05]))])
        args3 = np.array([x1-0.05, x2+0.05])
        args4 = np.array([x1-0.05, np.max(np.array([0, x2-0.05]))])
        
        f = 0.25 * (self.eval_fed_L1(args1) + self.eval_fed_L1(args2)) +\
            0.25 * (self.eval_fed_L1(args3) + self.eval_fed_L1(args4))
        
        return f    

    def eval_fed_L1(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        nom = 2300 * x1 * x1 * x1 + 1900 * x1 * x1 + 2092 * x1 + 60
        den = 100 * x1 * x1 * x1 + 500 * x1 * x1 + 4 * x1 + 20
        
        if x2 == 0:
            f = nom / den
        else:
            f = (1 - np.exp(-1/(2*x2))) * nom / den

        return f

    
class Branin:
    def __init__(self, debug=False):
        self.dim = 2
        self.flevels = 3
        self.maximum = -0.397887
        
        self.bounds = ((-5,10), (0,15))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]
            
        self.Flist = []
        self.Flist.append(self.eval_fed_L0)
        self.Flist.append(self.eval_fed_L1)
        self.Flist.append(self.eval_fed_L2)
        
    
    def query(self, X, m):
        # negate function to find maximum
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        for n in range(X.shape[0]):
            xn = X[n]
            ym[n] = self.Flist[m](xn)


        return -ym
    
    def eval_fed_L2(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        term1 = -1.275*np.square(x1)/np.square(np.pi) + 5*x1/np.pi + x2 - 6
        term2 = (10 - 5 / (4*np.pi))*np.cos(x1)
        
        f3 = np.square(term1) + term2 + 10
        
        return f3
    
    def eval_fed_L1(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        f3 = self.eval_fed_L2(xn-2)
        f2 = 10*np.sqrt(f3) + 2*(x1-0.5) - 3*(3*x2-1) - 1

        return f2

    def eval_fed_L0(self, xn):
        
        x1 = xn[0]
        x2 = xn[1]
        
        f2 = self.eval_fed_L1(1.2*(xn+2))
        f1 = f2 - 3*x2 + 1
        
        return f1

import numpy as np
from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot

from time import perf_counter
from collections import OrderedDict
from math import pi


# F16DataT = NDArray[np.float_]
# F16ResultT = ExtraResult[F16DataT, list]


F16_PARAM_MAP = OrderedDict({
    'air_speed': {
        'enabled': False,
        'default': 540
    },
    'angle_of_attack': {
        'enabled': False,
        'default': np.deg2rad(2.1215)
    },
    'angle_of_sideslip': {
        'enabled': False,
        'default': 0
    },
    'roll': {
        'enabled': True,
        'default': None,
        'range': (pi / 4) + np.array((-pi / 20, pi / 30)),
    },
    'pitch': {
        'enabled': True,
        'default': None,
        'range': (-pi / 2) * 0.8 + np.array((0, pi / 20)),
    },
    'yaw': {
        'enabled': True,
        'default': None,
        'range': (-pi / 4) + np.array((-pi / 8, pi / 8)),
    },
    'roll_rate': {
        'enabled': False,
        'default': 0
    },
    'pitch_rate': {
        'enabled': False,
        'default': 0
    },
    'yaw_rate': {
        'enabled': False,
        'default': 0
    },
    'northward_displacement': {
        'enabled': False,
        'default': 0
    },
    'eastward_displacement': {
        'enabled': False,
        'default': 0
    },
    'altitude': {
        'enabled': False,
        # 'default': 2338.4
        'default': 2335
    },
    'engine_power_lag': {
        'enabled': False,
        'default': 9
    }
})

class F16Model:
    def __init__(self, static_params_map, step_size, model, integrator) -> None:
        self.F16_PARAM_MAP = static_params_map
        self.step_size = step_size
        self.model = model
        self.integrator = integrator

    def get_static_params(self):
        static_params = []
        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                static_params.append(config['range'])
        return static_params


    def _compute_initial_conditions(self, X):
        conditions = []
        index = 0

        for param, config in self.F16_PARAM_MAP.items():
            if config['enabled']:
                conditions.append(X[index])
                index = index + 1
            else:
                conditions.append(config['default'])

        return conditions

    def simulate(
        self, inputs
    ):
        
        init_cond = self._compute_initial_conditions(inputs)
        
        step = 1 / self.step_size
        autopilot = GcasAutopilot(init_mode="roll", stdout=False, gain_str="old")

        start_time = perf_counter()
        result = run_f16_sim(init_cond, 15, autopilot, step, extended_states=True, model_str=self.model, integrator_str=self.integrator)
        end_time = perf_counter() - start_time

        trajectories = result["states"][:, 11].T.astype(np.float64)
        # print(trajectories)
        
        return np.min(trajectories)
        
    


def evaluate(model, integrator, freq, x):
    
    f16_model = F16Model(static_params_map=F16_PARAM_MAP, step_size=freq, integrator=integrator, model=model)
    return f16_model.simulate(x)
    
class Park1:
    def __init__(self, debug=False):
        self.dim = 3
        self.flevels = 2
        self.maximum = -np.inf
        
        self.bounds = (
        tuple((pi / 4) + np.array((-pi / 20, pi / 30))),
        tuple((-pi / 2) * 0.8 + np.array((0, pi / 20))),
        tuple((-pi / 4) + np.array((-pi / 8, pi / 8))))
        
        
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]

    def query(self, X, m):

        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        
        
        for n in range(X.shape[0]):
            xn = X[n]
            
            if m == 1:
                ym[n] = self.eval_fed_L0(xn)
            elif m == 0:
                ym[n] = self.eval_fed_L1(xn)
        return ym
    

    def eval_fed_L0(self, xn):
        
        f = evaluate("morelli", "rk45", 100, xn)
        
        return f    

    def eval_fed_L1(self, xn):
        
        f = evaluate("stevens", "euler", 35, xn)

        return f
    
class Levy:
    """ negative harmann3D, find maximum instead of global minimum """
    def __init__(self, debug=False):
        self.dim = 2
        self.flevels = 3
        self.maximum = 0

        self.bounds = ((-10,10),(-10,10))
        self.lb = np.array(self.bounds, ndmin=2)[:, 0]
        self.ub = np.array(self.bounds, ndmin=2)[:, 1]


    
    def query(self, X, m):
        # negate function to find maximum
        if X.ndim == 1:
            X = np.expand_dims(X, 0)
        
        N = X.shape[0]
        ym = np.zeros(N)
        
        
        for n in range(X.shape[0]):
            xn = X[n]
            
            if m == 2:
                ym[n] = -self.eval_high_fidel(xn)
            elif m == 1:
                ym[n] = -np.exp(0.1*np.sqrt(self.eval_high_fidel(xn))) - 0.1*np.sqrt(1 + np.square(self.eval_high_fidel(xn)))
            elif m == 0:
                ym[n] = -np.sqrt(1 + np.square(self.eval_high_fidel(xn)))

        return ym
    
    
    def eval_high_fidel(self, xn):
        outer = 0.0
        x1 = xn[0]
        x2 = xn[1]
        
        term1 = np.square(np.sin(3*np.pi*x1))
        term2 = np.square(x1-1) * (1 + np.square(np.sin(3*np.pi*x2)))
        term3 = np.square(x2-1) * (1 + np.square(np.sin(2*np.pi*x2)))
        
        f = term1 + term2 + term3
        
        return f
    


    
class SynMfData:
    def __init__(self, domain, Ntrain_list, Ntest_list, seed=None, perturb_scale=1e-2, perturb_thresh=1e-3):
        self.domain = domain

        Fdict = {
            'CurrinExp'  : CurrinExp(),
            'Branin'     : Branin(),
            'Park1'      : Park1(),
            'Levy'       : Levy(),
        }
        
        self.MfFn = Fdict[self.domain]

        self.Ntrain_list = Ntrain_list
        self.Ntest_list = Ntest_list
        self.dim = self.MfFn.dim
        self.Nfid = self.MfFn.flevels
        self.maximum = self.MfFn.maximum
        
        if seed == None:
            self.seed = np.random.randint(0,100000)
        else:
            self.seed = seed
        
        self.perturb_scale = perturb_scale
        self.perturb_thresh = perturb_thresh

        self.data = []
        
        self.Xscalers = []
        self.yscalers = []
        for m in range(self.Nfid):
            self.Xscalers.append(StandardScaler())
            self.yscalers.append(StandardScaler())

        for m in range(self.Nfid):
            Nm_train = self.Ntrain_list[m]
            Nm_test = self.Ntest_list[m]
            
            Dm = {}
            raw_Xall, yall = self.generate(Nm_train + Nm_test, m, self.seed)
    
            Dm['ytrain'] = yall[0:Nm_train]
            Dm['ytest'] = yall[Nm_train:Nm_train+Nm_test]
            
            Dm['raw_Xall'] = raw_Xall
            Dm['raw_Xtrain'] = raw_Xall[0:Nm_train,:]
            Dm['raw_Xtest'] = raw_Xall[Nm_train:Nm_train+Nm_test,:]
            
            self.Xscalers[m].fit(raw_Xall)

            
            Xall = self.Xscalers[m].transform(raw_Xall)

            Dm['Xtrain'] = Xall[0:Nm_train,:]
            Dm['Xtest'] = Xall[Nm_train:Nm_train+Nm_test,:]

            self.data.append(Dm)
            
        self.lb = np.squeeze(self.Xscalers[-1].transform(self.MfFn.lb.reshape([1,-1])))
        self.ub = np.squeeze(self.Xscalers[-1].transform(self.MfFn.ub.reshape([1,-1])))
            

            
    def query(self, X, m):
        
        rescale_X = self.Xscalers[m].inverse_transform(X)
        rescale_X = np.clip(np.squeeze(rescale_X), self.MfFn.lb, self.MfFn.ub).reshape([1,-1])
        
        ym = self.MfFn.query(rescale_X, m)
        return ym
            
            
    def generate(self, N, m, seed):
        
        state = np.random.get_state()
        X = None
        y = None
        try:
            np.random.seed(seed+m)

            noise = np.random.uniform(0,1,size=[N,self.dim])
            support = (self.MfFn.ub - self.MfFn.lb).reshape([1,-1])

            X = noise * support + self.MfFn.lb
            
            y = self.MfFn.query(X, m).reshape([-1,1])

        except:
            perm = np.arange(N)
        finally:
            np.random.set_state(state)

        return X, y

    
    def append(self, X, m):
        
        X = self.perturb(X, m)
        
        rescale_X = self.Xscalers[m].inverse_transform(X)
        rescale_X = np.clip(np.squeeze(rescale_X), self.MfFn.lb, self.MfFn.ub).reshape([1,-1])

        yq = self.MfFn.query(rescale_X, m)
        ystar = self.MfFn.query(rescale_X, self.Nfid - 1)


        raw_Xall = np.concatenate([rescale_X, self.data[m]['raw_Xall']], axis=0)
        self.Xscalers[m].fit(raw_Xall)
        self.data[m]['raw_Xall'] = self.Xscalers[m].transform(raw_Xall)
        
        
        raw_Xtrain = np.concatenate([rescale_X, self.data[m]['raw_Xtrain']], axis=0)
        Xtrain = self.Xscalers[m].transform(raw_Xtrain)
        perm = np.random.permutation(Xtrain.shape[0])
        
        Xtrain = Xtrain[perm]
        ytrain = self.MfFn.query(raw_Xtrain[perm], m)

        self.data[m]['Xtrain'] = Xtrain
        self.data[m]['ytrain'] = ytrain.reshape([-1,1])
        
        self.Ntrain_list[m] = Xtrain.shape[0]
        
        # updat the boundary
        self.lb = np.squeeze(self.Xscalers[-1].transform(self.MfFn.lb.reshape([1,-1])))
        self.ub = np.squeeze(self.Xscalers[-1].transform(self.MfFn.ub.reshape([1,-1])))
        
        return yq, ystar, X
    
    def perturb(self, X, m):
        # perturb X if necessary
        Xm = self.data[m]['Xtrain']
        dist = np.sqrt(np.sum(np.square(Xm - X), axis=1))
        
        if np.min(dist) < self.perturb_thresh:
            print('Perturb X!!!!!')
            bounds = self.ub - self.lb
            perturbation = bounds * self.perturb_scale * (np.random.uniform() - 0.5)
            
            X_perturb = X + perturbation.reshape([1,-1])
            return X_perturb
        
        return X
    