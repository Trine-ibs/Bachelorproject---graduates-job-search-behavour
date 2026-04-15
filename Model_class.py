import pandas as pd
import numpy as np
from types import SimpleNamespace


class ModelClass():

    def __init__(self, file_pre="HR_unobs_pre.csv", file_post="HR_unobs_post.csv"):

        self.file_pre = file_pre
        self.file_post = file_post

        self.settings()

        for name in self.namespaces:
            setattr(self, name, SimpleNamespace())

        self.setup()
        self.allocate()


    def settings(self):

        self.namespaces =['par', 'sol', 'data', 'counter']


    def setup(self):

        # load data
        data = self.data
        data.pre = SimpleNamespace()
        data.post = SimpleNamespace()

        df = pd.read_csv(self.file_pre)
        data.moments = df["HR_logit"]
        data.var = df["HR_var"]

        df_pre = pd.read_csv(self.file_pre)
        data.pre.moments = df_pre["HR_logit"]
        data.pre.var = df_pre["HR_var"]

        df_post = pd.read_csv(self.file_post)
        data.post.moments = df_post["HR_logit"]
        data.post.var = df_post["HR_var"]


        # initialize parameters
        par = self.par
        par.pre = SimpleNamespace()
        par.post = SimpleNamespace()

        # income levels
        par.w = 10000
        par.b1 = 4000
        par.b2 = 4000
        par.b3 = 1600
        
        par.pre.w = 10000
        par.pre.b1 = 4000
        par.pre.b2 = 4000
        par.pre.b3 = 1600

        par.post.w = 10000
        par.post.b1 = 3500
        par.post.b2 = 3500
        par.post.b3 = 1600

        # time periods
        par.T1 = 12
        par.T2 = 106
        par.T3 = len(data.pre.moments)

        par.pre.T1 = 12
        par.pre.T2 = 106
        par.pre.T3 = len(data.pre.moments)

        par.post.T1 = 12
        par.post.T2 = 54
        par.post.T3 = len(data.post.moments)

        # model parameters
        par.kappa = np.array([30.0, 50.0, 100.0])
        par.gamma = 1.0
        par.delta = 0.8
        par.mu = 1.5
        par.type_share1 = 0.2
        par.type_share2 = 0.2

        # initial guesses
        par.kappa0 = np.array([30.0, 50.0, 100.0])
        par.gamma0 = 1.0
        par.delta0 =  0.8
        par.mu0 = 1.1
        par.type_share1_0 = 0.4
        par.type_share2_0 = 0.3

        # transformed initial guesses
        par.kappa0_trans = np.log(par.kappa0)
        par.gamma0_trans = np.log(par.gamma0)
        par.delta0_trans = np.log(par.delta0/(1-par.delta0))
        par.mu0_trans = np.log(par.mu0)

        par.type_share1_trans = np.log(par.type_share1_0/(1-par.type_share1_0))
        par.type_share2_trans = np.log(par.type_share2_0/(1-par.type_share2_0))

        # options
        par.weight = True
        par.unobs_het = True

        par.moments_start = 0
        par.moments_end = None


    def allocate(self):
        sol = self.sol
        sol.pre = SimpleNamespace()
        sol.post = SimpleNamespace()

        counter = self.counter 
        counter.pre = SimpleNamespace()
        counter.post = SimpleNamespace()

        i = len(self.par.kappa)

        # allocate solution arrays
        sol.s = np.zeros((i, self.par.T3))
        sol.V_u = np.zeros((i, self.par.T3+1))
        sol.V_e = np.zeros((i, self.par.T3+1))
        sol.moments = np.zeros(self.par.T3)

        # pre-reform
        sol.pre.s = np.zeros((i, self.par.pre.T3))
        sol.pre.V_u = np.zeros((i, self.par.pre.T3+1))
        sol.pre.V_e = np.zeros((i, self.par.pre.T3+1))
        sol.pre.moments = np.zeros(self.par.pre.T3)

        # post-reform
        sol.post.s = np.zeros((i, self.par.post.T3))
        sol.post.V_u = np.zeros((i, self.par.post.T3+1))
        sol.post.V_e = np.zeros((i, self.par.post.T3+1))
        sol.post.moments = np.zeros(self.par.post.T3)

        # allocate counterfactuals
        counter.s = np.zeros((i, self.par.T3))
        counter.V_u = np.zeros((i, self.par.T3+1))
        counter.V_e = np.zeros((i, self.par.T3+1))
        counter.moments = np.zeros(self.par.T3)

        # pre-reform
        counter.pre.s = np.zeros((i, self.par.pre.T3))
        counter.pre.V_u = np.zeros((i, self.par.pre.T3+1))
        counter.pre.V_e = np.zeros((i, self.par.pre.T3+1))
        counter.pre.moments = np.zeros(self.par.pre.T3)
        
        # post-reform
        counter.post.s = np.zeros((i, self.par.post.T3))
        counter.post.V_u = np.zeros((i, self.par.post.T3+1))
        counter.post.V_e = np.zeros((i, self.par.post.T3+1))
        counter.post.moments = np.zeros(self.par.post.T3)

    def print_par(self):
        par = self.par
        results = {
            "Parameter": ["kappa1", "kappa2", "kappa3", "gamma", "delta", "mu", "type_share1", "type_share2", "type_share3"],
            "Estimated Value": [
                par.kappa[0], par.kappa[1], par.kappa[2],
                par.gamma, par.delta, par.mu,
                par.type_share1, par.type_share2,
                1 - par.type_share1 - par.type_share2
            ],
        }
        df = pd.DataFrame(results)
        return df.round(4)
        