from scipy import linalg

from dyn_model_funct import solve_group
import numpy as np
from scipy.optimize import minimize


# ── Simulation 

def sim_hr(par, sol, g):
    """Simulate hazard rate moments for a single group g."""
    gpar = getattr(par, g)
    gsol = getattr(sol, g)

    gsol.moments[:] = 0.0
    solve_group(par, gpar, gsol)

    type_shares = gpar.type_shares.copy()

    for t in range(gpar.T3):
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


def sim_hr_prepost(par, sol, demo_groups):
    """Simulate hazard rates for all groups in demo_groups."""
    for subgroups in demo_groups:
        for g in subgroups:
            sim_hr(par, sol, g)


# ── Utility type helpers 

def _has_alpha(util_type):
    return util_type in ("linear", "quadratic", "CRRA", "CRRA_shift", "CARA")


def _has_beta(util_type):
    return util_type in ("log_shift", "CRRA_shift", "sqrt_shift")


def _is_joint(par, demo_groups):
    """True when estimating multiple demographic groups jointly with beta_shared."""
    return getattr(par, "beta_shared", False) and len(demo_groups) > 1


# ── Parameter transformations 

def softmax_with_base(z):
    """Returns K shares summing to 1, with last type as base category."""
    if len(z) == 0:
        return np.array([1.0])
    logits     = np.append(z, 0.0)
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def build_ordered_kappas(raw):
    """
    Build ordered kappas from incremental log-space parameters.
    raw[0] = log(kappa1)
    raw[k] = log(kappa_{k+1} - kappa_k)  for k >= 1
    """
    kappas    = np.empty(len(raw))
    kappas[0] = np.exp(raw[0])
    for i in range(1, len(raw)):
        kappas[i] = kappas[i-1] + np.exp(raw[i])
    return kappas


def n_kappa_params(par):
    """Returns number of free kappa parameters."""
    if par.kappa3_fixed is not False:
        return par.n_types - 1
    return par.n_types


def unpack_kappas(x, idx, par):
    nk  = n_kappa_params(par)
    raw = np.array(x[idx:idx+nk])

    if par.kappa3_fixed is not False:
        kappas_free = build_ordered_kappas(raw)
        kappas      = np.append(kappas_free, par.kappa3_fixed)
    else:
        kappas = build_ordered_kappas(raw)

    return kappas, idx + nk


# ── Parameter vector packing / unpacking 

def unpack_params_prepost(x, par, demo_groups):
    """ 
    Unpack parameter vector x into par, following the order defined in build_initial_x_prepost.
    Transform parameters back to orginial scale.
    """


    K      = par.n_types
    idx    = 0
    joint  = _is_joint(par, demo_groups)

    # ── kappa, alpha and beta per demo group 
    for subgroups in demo_groups:
        kappa, idx = unpack_kappas(x, idx, par)

        if _has_alpha(par.util_type):
            alpha = np.exp(x[idx]); idx += 1

        if _has_beta(par.util_type) and not joint:
            min_b3 = min(getattr(par, g).b3 for g in subgroups)
            beta   = np.exp(x[idx]) - min_b3; idx += 1

        for g in subgroups:
            getattr(par, g).kappa = kappa
            if _has_alpha(par.util_type):
                getattr(par, g).alpha = alpha
            if _has_beta(par.util_type) and not joint:
                getattr(par, g).beta = beta

    # ── shared beta (joint beta mode only)
    if _has_beta(par.util_type) and joint:
        all_groups = [g for subgroups in demo_groups for g in subgroups]
        min_b3_all = min(getattr(par, g).b3 for g in all_groups)
        beta       = np.exp(x[idx]) - min_b3_all; idx += 1
        par.beta   = beta
        for subgroups in demo_groups:
            for g in subgroups:
                getattr(par, g).beta = beta

    # ── gamma and delta

    if not joint:
        par.gamma = np.exp(x[idx]); idx += 1
        if par.delta_fixed == False:
            par.delta = 1 / (1 + np.exp(-x[idx])); idx += 1

        # apply shared gamma/delta if multiple groups but not shared beta
        for subgroups in demo_groups:
            for g in subgroups:
                getattr(par, g).gamma = par.gamma
                getattr(par, g).delta = par.delta
    # if shared beta and multiple demo groups
    else:
        for subgroups in demo_groups:
            gamma = np.exp(x[idx]); idx += 1
            if par.delta_fixed == False:
                delta = 1 / (1 + np.exp(-x[idx])); idx += 1
            else:
                delta = par.delta
            for g in subgroups:
                gpar_g        = getattr(par, g)
                gpar_g.gamma  = gamma
                gpar_g.delta  = delta

            par.gamma = gamma
            par.delta = delta

    # ── shares per demo group 
    for subgroups in demo_groups:
        type_shares = softmax_with_base(np.array(x[idx:idx+(K-1)]))
        idx        += (K-1)
        for g in subgroups:
            getattr(par, g).type_shares = type_shares

    return par


def build_initial_x_prepost(par, demo_groups):
    """ Put parameters on transformed scale and build the x vector for optimisation input. """
    x0    = []
    joint = _is_joint(par, demo_groups)

    # ── kappa, alpha and beta
    for subgroups in demo_groups:
        gpar   = getattr(par, subgroups[0])
        min_b3 = min(getattr(par, g).b3 for g in subgroups)

        if par.kappa3_fixed is not False:
            x0.extend(gpar.kappa0_trans[:-1].tolist())
        else:
            x0.extend(gpar.kappa0_trans.tolist())

        if _has_alpha(par.util_type):
            x0.append(gpar.alpha0_trans)

        if _has_beta(par.util_type) and not joint:
            x0.append(np.log(min_b3 + gpar.beta0))

    # ── shared beta 
    if _has_beta(par.util_type) and joint:
        all_groups = [g for subgroups in demo_groups for g in subgroups]
        min_b3_all = min(getattr(par, g).b3 for g in all_groups)
        x0.append(np.log(min_b3_all + par.beta0))

    # ── gamma and delta 
    if not joint:
        x0.append(par.gamma0_trans)
        if par.delta_fixed == False:
            x0.append(par.delta0_trans)
    else:
        for subgroups in demo_groups:
            gpar = getattr(par, subgroups[0])
            x0.append(par.gamma0_trans)
            if par.delta_fixed == False:
                x0.append(par.delta0_trans)

    # ── shares per demo group 
    for subgroups in demo_groups:
        gpar = getattr(par, subgroups[0])
        x0.extend(gpar.type_shares0_trans.tolist())

    return x0


def build_bounds_prepost(par, demo_groups):
    """Build bounds for estimation."""
    K             = par.n_types
    n_kappa       = n_kappa_params(par)
    bounds        = []
    joint         = _is_joint(par, demo_groups)
    safety_margin = 1e-3  

    # ── kappa, alpha and beta
    for subgroups in demo_groups:
        for k in range(n_kappa):
            if par.kappa3_fixed is not False and k == n_kappa - 1:
                upper = np.log(par.kappa3_fixed - 1.0)
            else:
                upper = np.log(1e7)
            bounds.append((np.log(1e-7), upper))

        if par.util_type in ("linear", "CRRA", "CRRA_shift"):
            bounds.append((np.log(1e-3), np.log(20)))
        if par.util_type in ("quadratic", "CARA"):
            bounds.append((np.log(1e-8), np.log(1e-3)))

        if _has_beta(par.util_type) and not joint:
            min_b3 = min(getattr(par, g).b3 for g in subgroups)
            bounds.append((np.log(safety_margin), np.log(min_b3 + 1e6)))

    # ── shared beta 
    if _has_beta(par.util_type) and joint:
        all_groups = [g for subgroups in demo_groups for g in subgroups]
        min_b3_all = min(getattr(par, g).b3 for g in all_groups)
        bounds.append((np.log(safety_margin), np.log(min_b3_all + 1e6)))

    # ── gamma and delta 
    if not joint:
        bounds.append((np.log(0.1), np.log(20)))   # gamma
        if par.delta_fixed == False:
            bounds.append((-10, 10))               # delta
    else:
        for _ in demo_groups:
            bounds.append((np.log(0.1), np.log(20)))   # gamma per group
            if par.delta_fixed == False:
                bounds.append((-10, 10))               # delta per group

    # ── shares per demo group 
    for _ in demo_groups:
        bounds.extend([(-10, 10)] * (K - 1))

    return bounds


# ── Objective function 

def sum_sqr_diff_prepost(x, par, sol, data, demo_groups):

    """sum of squared differences objective function"""

    # unpack parameters and simulate moments for all groups
    par = unpack_params_prepost(x, par, demo_groups)
    sim_hr_prepost(par, sol, demo_groups)

    # keep moments in the specificed durations. Combine moments used for estimation
    idx_slice   = slice(par.moments_start, par.moments_end)
    all_moments, all_var, all_sim = [], [], []

    for g in [g for subgroups in demo_groups for g in subgroups]:
        gdata = getattr(data, g)
        gsol  = getattr(sol, g)
        all_moments.append(np.array(gdata.moments[idx_slice]))
        all_var.append(np.array(gdata.var[idx_slice]))
        all_sim.append(gsol.moments[idx_slice])

    moments     = np.concatenate(all_moments)
    var         = np.concatenate(all_var)
    moments_sim = np.concatenate(all_sim)

    if par.weight:
        ssmd = np.sum(((moments - moments_sim)**2) / (var))
    else:
        ssmd = np.sum((moments - moments_sim)**2)

    return ssmd


# ── Estimation 

def gmm_estimate_prepost(par, sol, data, demo_groups):
    """Estimate model by GMM using both pre and post data jointly."""
    initial_x = build_initial_x_prepost(par, demo_groups)
    bounds    = build_bounds_prepost(par, demo_groups)

    res = minimize(
        sum_sqr_diff_prepost,
        x0      = initial_x,
        args    = (par, sol, data, demo_groups),
        method  = "L-BFGS-B",
        bounds  = bounds,
        options = {"ftol": 1e-10, "gtol": 1e-6, "maxiter": 15000}
    )

    return res.x, res.fun, res.success, res.message

def par_SE(par, sol, data, demo_groups, theta):
    """ Compute standard errors of estimated parameter """
    n_params   = len(theta)
    I          = np.eye(n_params)
    all_groups = [g for subgroups in demo_groups for g in subgroups]
    idx_slice  = slice(par.moments_start, par.moments_end)

    # ── moment function 
    def get_moments(x):
        unpack_params_prepost(x, par, demo_groups)
        sim_hr_prepost(par, sol, demo_groups)
        return np.concatenate([
            getattr(sol, g).moments[idx_slice].copy() for g in all_groups
        ])

    # ── Jacobian G by finite differences 
    n_moments = len(get_moments(theta))
    G = np.zeros((n_moments, n_params))

    for j in range(n_params):
        eps_j       = 1e-5 * max(abs(theta[j]), 1.0)
        theta_plus  = theta + eps_j * I[j]
        theta_minus = theta - eps_j * I[j]
        G[:, j]     = (get_moments(theta_plus) - get_moments(theta_minus)) / (2 * eps_j)

    # reset parameters to estimated values
    unpack_params_prepost(theta, par, demo_groups)

    # ── build block-diagonal Omega and W 
    omega_blocks = []
    w_blocks     = []

    for g in all_groups:
        gdata   = getattr(data, g)

        # full covariance matrix for this group
        omega_g = np.array(gdata.omega[idx_slice, :][:, idx_slice])
        omega_blocks.append(omega_g)

        # W = diagonal matrix with inverse of diagonal elements of Omega
        diag_g  = np.diag(omega_g)
        w_g     = np.diag(1.0 / (diag_g))
        w_blocks.append(w_g)

    # full block-diagonal matrices
    Omega = linalg.block_diag(*omega_blocks)
    W     = linalg.block_diag(*w_blocks)

    # ── sandwich formula 
    GWG     = G.T @ W @ G
    GWG_inv = linalg.inv(GWG)
    middle  = G.T @ W @ Omega @ W @ G

    var_theta = GWG_inv @ middle @ GWG_inv
    se        = np.sqrt(np.diag(var_theta))

    return se, var_theta
    

