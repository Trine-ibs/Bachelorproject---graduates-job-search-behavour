import numpy as np
import math
from scipy.optimize import minimize_scalar


def consump_util(par, gpar, c):
    if par.util_type == "linear":
        return gpar.alpha * c
    if par.util_type == "quadratic":
        return c - gpar.alpha * c**2
    if par.util_type == "sqrt":
        return np.sqrt(max(c, 1e-10))
    if par.util_type == "sqrt_shift":
        return np.sqrt(max(c + gpar.beta, 1e-10))
    if par.util_type == "log":
        return np.log(max(c, 1e-6))
    if par.util_type == "log_shift":
        return np.log(max(c + gpar.beta, 1e-6))
    if par.util_type == "CRRA":
        c = max(c, 1e-10)
        if math.isclose(gpar.alpha, 1.0, rel_tol=1e-9):
            return np.log(c)
        else:
            return (c**(1 - gpar.alpha)) / (1 - gpar.alpha)
    if par.util_type == "CRRA_shift":
        c_shifted = max(c + gpar.beta, 1e-10)
        if math.isclose(gpar.alpha, 1.0, rel_tol=1e-9):
            return np.log(c_shifted)
        else:
            return (c_shifted**(1 - gpar.alpha)) / (1 - gpar.alpha)
    if par.util_type == "CARA":
        return -np.exp(-gpar.alpha * c)


def search_cost(s, kappa_i, gamma):
    return kappa_i * s**(1 + gamma) / (1 + gamma)


def s_policy(delta, gamma, future_val, kappa_i):
    if future_val <= 0:
        return 0.0
    return min(1.0, max(0.0, (delta / kappa_i * future_val) ** (1 / gamma)))


def value_employ_ss(par, gpar, w):
    delta = getattr(gpar, "delta", par.delta)
    return consump_util(par, gpar, w) / (1 - delta)


def value_unemploy_ss(par, gpar, V_E, kappa_i, c_ss):
    delta = getattr(gpar, "delta", par.delta)
    gamma = getattr(gpar, "gamma", par.gamma)

    def object_func(s):
        V_U = (
            consump_util(par, gpar, c_ss)
            - search_cost(s, kappa_i, gamma)
            + delta * s * V_E
        ) / (1 - delta * (1 - s))
        return -V_U

    result = minimize_scalar(object_func, bounds=(0, 1), method="bounded")
    return result.x, -result.fun


def solve_group(par, gpar, gsol):
    """Solve dynamic model with backwards induction
    """
    delta = getattr(gpar, "delta", par.delta)
    gamma = getattr(gpar, "gamma", par.gamma)

    for i in range(par.n_types):
        kappa_i        = gpar.kappa[i]
        V_e_val        = value_employ_ss(par, gpar, gpar.w)
        gsol.V_e[i, :] = V_e_val

        for t in range(gpar.T3, -1, -1):
            if t == gpar.T3:
                _, gsol.V_u[i, t] = value_unemploy_ss(par, gpar, V_e_val, kappa_i, gpar.b3)
            else:
                future_val   = gsol.V_e[i, t+1] - gsol.V_u[i, t+1]
                gsol.s[i, t] = s_policy(delta, gamma, future_val, kappa_i)

                if t < gpar.T1:
                    b = gpar.b1
                elif gpar.T1 <= t < gpar.T2:
                    b = gpar.b2
                else:
                    b = gpar.b3

                gsol.V_u[i, t] = (
                    consump_util(par, gpar, b)
                    - search_cost(gsol.s[i, t], kappa_i, gamma)
                    + delta * (
                        gsol.s[i, t] * gsol.V_e[i, t+1]
                        + (1 - gsol.s[i, t]) * gsol.V_u[i, t+1]
                    )
                )
