from Model_funct import solve, solve_prepost
import numpy as np
from scipy.optimize import minimize

# bruges til model, hvor man kun estimerer på pre-reform eller post-reform data
def sim_hr(par, sol):
    VE, VU, s_star = solve(par,sol)

    if par.unobs_het == True:
        type_shares = np.array([par.type_share1, par.type_share2, 1-par.type_share1-par.type_share2])
        
    else:
        type_shares = np.array([1, 0, 0])

    for t in range(par.T3):
        if t == 0:
            sol.moments[t] = type_shares @ sol.s[:,t]
        else:
            type_shares = type_shares*(1-sol.s[:,t-1])
            type_shares = type_shares / np.sum(type_shares)
            
            sol.moments[t] = type_shares @ sol.s[:,t]
   
    return sol.moments


# bruges til model, hvor man estimerer på både pre-reform og post-reform data
def sim_hr_prepost(par, sol):
   
    solve_prepost(par,sol)

    if par.unobs_het == True:
        init_shares = np.array([par.type_share1, par.type_share2, 1-par.type_share1-par.type_share2])

    else:
        init_shares = np.array([1, 0, 0])

    # pre-reform moments
    type_shares = init_shares.copy()

    for t in range(par.pre.T3):
        if t == 0:
            sol.pre.moments[t] = type_shares @ sol.pre.s[:,t]
        else:
            type_shares = type_shares*(1-sol.pre.s[:,t-1])
            type_shares = type_shares / np.sum(type_shares)
            
            sol.pre.moments[t] = type_shares @ sol.pre.s[:,t]
    # post-reform moments
    type_shares = init_shares.copy()

    for t in range(par.post.T3):
        if t == 0:
            sol.post.moments[t] = type_shares @ sol.post.s[:,t]
        else:
            type_shares = type_shares*(1-sol.post.s[:,t-1])
            type_shares = type_shares / np.sum(type_shares)
            
            sol.post.moments[t] = type_shares @ sol.post.s[:,t]


def sum_sqr_diff(x, par, sol, data):

    if par.unobs_het == True:
        par.kappa = np.exp(np.array([x[0], x[1], x[2]]))
        par.gamma = np.exp(x[3])
        par.delta = 1/(1+np.exp(-x[4]))

        par.mu = np.exp(x[5])
        par.type_share1 = 1/(1+np.exp(-x[6]))
        par.type_share2 = (1-par.type_share1) * (1/(1+np.exp(-x[7])))

    else:
        par.kappa = np.array([np.exp(x[0]),0])
        par.gamma = np.exp(x[1])
        par.delta = 1/(1+np.exp(-x[2]))
        par.mu = np.exp(x[3])

    sim_hr(par, sol)

    # index for moments and variance
    idx = slice(par.moments_start, par.moments_end)
    moments = data.moments[idx]
    var = data.var[idx]
    moments_sim = sol.moments[idx]

    if par.weight == True:
        e = 1e-7
        ssmd = np.sum(((moments - moments_sim)**2) / (var+e))

    else: 
        ssmd = np.sum((moments - moments_sim)**2)
        
    return ssmd


def sum_sqr_diff_prepost(x, par, sol, data):

    if par.unobs_het == True:
        par.kappa = np.exp(np.array([x[0], x[1], x[2]]))
        par.gamma = np.exp(x[3])
        par.delta = 1/(1+np.exp(-x[4]))

        par.mu = np.exp(x[5])
        par.type_share1 = 1/(1+np.exp(-x[6]))
        par.type_share2 = (1-par.type_share1) * (1/(1+np.exp(-x[7])))

    else:
        par.kappa = np.array([np.exp(x[0]),0])
        par.gamma = np.exp(x[1])
        par.delta = 1/(1+np.exp(-x[2]))
        par.mu = np.exp(x[3])

    sim_hr_prepost(par, sol)

    # index for moments and variance
    idx = slice(par.moments_start, par.moments_end)
    
    # pre reform
    moments_pre = data.pre.moments[idx]
    var_pre = data.pre.var[idx]
    moments_sim_pre = sol.pre.moments[idx]

    # post reform
    moments_post = data.post.moments[idx]
    var_post = data.post.var[idx]
    moments_sim_post = sol.post.moments[idx]

    # combine pre and post moments and variance
    moments = np.concatenate((moments_pre, moments_post))
    var = np.concatenate((var_pre, var_post))
    moments_sim = np.concatenate((moments_sim_pre, moments_sim_post))

    if par.weight == True:
        e = 1e-7
        ssmd = np.sum(((moments - moments_sim)**2) / (var+e))

    else: 
        ssmd = np.sum((moments - moments_sim)**2)
        
    return ssmd



def smm_estimate(par, sol, data):

    if par.unobs_het == True:
        initial_x = [par.kappa0_trans[0], par.kappa0_trans[1], par.kappa0_trans[2], par.gamma0_trans, par.delta0_trans, par.mu0_trans, par.type_share1_trans, par.type_share2_trans]

        bounds = [
            (np.log(1e-3), None), #kappa1
            (np.log(1e-3), None), #kappa2
            (np.log(1e-3), None), #kappa3
            (np.log(1e-6), np.log(200)), #gamma
            (-10, 10), #delta
            (np.log(1e-3), None),
            (None, None),
            (None, None)
        ]

    else:
        initial_x = [par.kappa0_trans[0], par.gamma0_trans, par.delta0_trans, par.mu0_trans]

        bounds = [
            (np.log(1e-3), np.log(2000)), #kappa 
            (np.log(1e-6), np.log(200)), #gamma
            (-10, 10), #delta
            (np.log(1e-3), None), #mu
        ]

    res = minimize(
        sum_sqr_diff,
        x0 = initial_x,
        args=(par,sol,data),

        method= "L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1e-12,
            "gtol": 1e-6,
            "maxiter": 5000})

    return res.x, res.fun, res.success, res.message


def smm_estimate_prepost(par, sol, data):

    if par.unobs_het == True:
        initial_x = [par.kappa0_trans[0], par.kappa0_trans[1], par.kappa0_trans[2], par.gamma0_trans, par.delta0_trans, par.mu0_trans, par.type_share1_trans, par.type_share2_trans]

        bounds = [
            (np.log(1e-3), None), #kappa1
            (np.log(1e-3), None), #kappa2
            (np.log(1e-3), None), #kappa3
            (np.log(1e-6), np.log(200)), #gamma
            (-10, 10), #delta
            (np.log(1e-3), None),
            (None, None),
            (None, None)
        ]

    else:
        initial_x = [par.kappa0_trans[0], par.gamma0_trans, par.delta0_trans, par.mu0_trans]

        bounds = [
            (np.log(1e-3), np.log(2000)), #kappa 
            (np.log(1e-6), np.log(200)), #gamma
            (-10, 10), #delta
            (np.log(1e-3), None), #mu
        ]

    res = minimize(
        sum_sqr_diff_prepost,
        x0 = initial_x,
        args=(par,sol,data),

        method= "L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1e-12,
            "gtol": 1e-6,
            "maxiter": 5000})

    return res.x, res.fun, res.success, res.message