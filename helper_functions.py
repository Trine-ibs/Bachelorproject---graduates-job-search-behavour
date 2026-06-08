from dyn_model_estim import *
import numpy as np
from copy import deepcopy
from dyn_model_funct import solve_group
from types import SimpleNamespace


def simulate_to_length(model, group, T):
    """
    Simulate model for a specific group to length T,
    regardless of the data spell length.
    """
    par  = model.par
    gpar = deepcopy(getattr(model.par, group))
    gpar.T3 = T

    K    = par.n_types
    gsol = SimpleNamespace()
    gsol.s       = np.zeros((K, T))
    gsol.V_u     = np.zeros((K, T+1))
    gsol.V_e     = np.zeros((K, T+1))
    gsol.moments = np.zeros(T)

    solve_group(par, gpar, gsol)

    type_shares = gpar.type_shares.copy()
    for t in range(T):
        if t == 0:
            gsol.moments[t] = type_shares @ gsol.s[:, t]
        else:
            type_shares = type_shares * (1 - gsol.s[:, t-1])
            denom       = type_shares.sum()
            if denom <= 1e-12:
                gsol.moments[t:] = 0.0
                break
            type_shares     = type_shares / denom
            gsol.moments[t] = type_shares @ gsol.s[:, t]

    return gsol.moments


def aggregate_hazard(moments_list, shares, T=None):
    """ Compute aggregate hazard rate across groups weighted by population shares. """
    shares  = np.array(shares, dtype=float)
    shares /= shares.sum()

    T_min = T if T is not None else min(len(m) for m in moments_list)

    truncated = np.zeros((len(moments_list), T_min))
    for i, m in enumerate(moments_list):
        L = min(len(m), T_min)
        truncated[i, :L] = m[:L]

    group_weights = shares.copy()
    agg_moments   = np.zeros(T_min)

    for t in range(T_min):
        denom = group_weights.sum()
        if denom <= 1e-12:
            agg_moments[t:] = 0.0
            break
        w_norm         = group_weights / denom
        agg_moments[t] = w_norm @ truncated[:, t]
        group_weights  = group_weights * (1 - truncated[:, t])

    return agg_moments


def solve_counterfactual(individual_models, group,
                          w=None, b1=None, b2=None, b3=None,
                          T1=None, T2=None, T3=None):
    """ Solve the model for a counterfactual benefit schedule using parameters estimated from individual_models """
    model    = individual_models[group]
    gpar_est = getattr(model.par, group)
    par_cf   = deepcopy(model.par)

    # build counterfactual gpar starting from estimated pre-reform values
    gpar_cf     = SimpleNamespace()
    gpar_cf.w   = w  if w  is not None else gpar_est.w
    gpar_cf.b1  = b1 if b1 is not None else gpar_est.b1
    gpar_cf.b2  = b2 if b2 is not None else gpar_est.b2
    gpar_cf.b3  = b3 if b3 is not None else gpar_est.b3
    gpar_cf.T1  = T1 if T1 is not None else gpar_est.T1
    gpar_cf.T2  = T2 if T2 is not None else gpar_est.T2
    gpar_cf.T3  = T3 if T3 is not None else gpar_est.T3

    # structural parameters 
    gpar_cf.kappa       = gpar_est.kappa.copy()
    gpar_cf.type_shares = gpar_est.type_shares.copy()
    gpar_cf.beta        = gpar_est.beta
    gpar_cf.alpha       = gpar_est.alpha

    # allocate solution arrays
    K    = model.par.n_types
    T    = gpar_cf.T3
    gsol = SimpleNamespace()
    gsol.s       = np.zeros((K, T))
    gsol.V_u     = np.zeros((K, T+1))
    gsol.V_e     = np.zeros((K, T+1))
    gsol.moments = np.zeros(T)

    solve_group(par_cf, gpar_cf, gsol)

    # simulate hazard rate
    type_shares = gpar_cf.type_shares.copy()
    for t in range(gpar_cf.T3):
        if t == 0:
            gsol.moments[t] = type_shares @ gsol.s[:, t]
        else:
            type_shares = type_shares * (1 - gsol.s[:, t-1])
            denom       = np.sum(type_shares)
            if denom <= 1e-12:
                gsol.moments[t:] = 0.0
                break
            type_shares     = type_shares / denom
            gsol.moments[t] = type_shares @ gsol.s[:, t]

    return gsol, gpar_cf


def interpolate_early_weeks(HR_model, data_pre, data_post, ms, shares):
    """ Interpolate the first ms weeks of a model-implied hazard rate for aggeregated hazards. """
    shares = np.array(shares, dtype=float)
    shares /= shares.sum()

    # weighted average of t=0 hazard across pre and post empirical data
    t0_vals = []
    for i, (dpre, dpost) in enumerate(zip(data_pre, data_post)):
        t0_group = (np.array(dpre.moments)[0] + np.array(dpost.moments)[0]) / 2
        t0_vals.append(t0_group)
    t0 = np.average(t0_vals, weights=shares)

    # model value at ms
    t_ms = HR_model[ms]

    # linear interpolation from t=0 to t=ms
    interp = np.linspace(t0, t_ms, ms + 1)  # ms+1 points: t=0 ... t=ms

    # combine interpolated early weeks with model from ms onwards
    HR_combined = HR_model.copy()
    HR_combined[:ms] = interp[:-1] 

    return HR_combined


def interpolate_cf_early(sol_moments, data_post, ms):
    """ Interpolate first ms weeks of a counterfactual hazard rate for single group hazards. """
    t0    = np.array(data_post.moments)[0]   # empirical t=0
    t_ms  = sol_moments[ms]                   # counterfactual at ms

    interp           = np.linspace(t0, t_ms, ms + 1)
    HR_combined      = sol_moments.copy()
    HR_combined[:ms] = interp[:-1]

    return HR_combined


def survival_function(hazard):
    """ Compute survival function from hazard rate array. """
    S    = np.ones(len(hazard) + 1)
    for t in range(len(hazard)):
        S[t+1] = S[t] * (1 - hazard[t])
    return S
