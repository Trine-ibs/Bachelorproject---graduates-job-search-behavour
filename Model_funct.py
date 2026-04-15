import numpy as np
import math
from scipy.optimize import minimize_scalar


#utility of consumption (CRRA function)
def consump_util(par, c):
    if math.isclose(par.mu, 1.0, rel_tol=1e-9):
        return np.log(c)
    else:
        return (c**(1 - par.mu)) / (1 - par.mu)

# search cost
def search_cost(s, par, i):
    return par.kappa[i] * s**(1+par.gamma) / (1+par.gamma)


# s* function
def s_policy(par, future_val, i):
    return min(1, max(0, ((par.delta/par.kappa[i]*future_val)**(1/par.gamma))))


#### Solve for steady state ###

# SS of employment
def value_employ_ss(par, w):
    V = consump_util(par, w) / (1-par.delta)
    return V


# SS of unemployment
def value_unemploy_ss(par, V_E, i):

    def object_func(s):

        c_ss = par.b3

        V_U = (consump_util(par, c_ss) - search_cost(s, par, i) + par.delta*s*V_E) / (1-par.delta*(1-s))
        return -V_U

    result = minimize_scalar(object_func, bounds=(0,1), method="bounded")

    optimal_s_ss = result.x
    V_u_ss = -result.fun
    return optimal_s_ss, V_u_ss


### Solve model ###

def solve(par, sol):

    T = par.T3

    if par.unobs_het == True:
        type_count = len(par.kappa)
    else:
        type_count = 1


    for i in range(type_count):

        w = par.w

        for t in range(T, -1, -1):
            if t == T:
                sol.V_e[i,t] = value_employ_ss(par, w)
                s_ss, sol.V_u[i,t] = value_unemploy_ss(par, sol.V_e[i,t], i)

            else:
                future_val = sol.V_e[i,t+1] - sol.V_u[i,t+1]
                sol.s[i,t] = s_policy(par, future_val, i)

                sol.V_e[i,t] = value_employ_ss(par, w)

                if t<par.T1:
                    b = par.b1
                elif par.T1 <= t < par.T2:
                    b = par.b2
                elif par.T2 <= t < par.T3:
                    b = par.b3

                sol.V_u[i,t] = consump_util(par, b) - search_cost(sol.s[i,t], par, i) + par.delta * (sol.s[i,t]*sol.V_e[i,t+1] + (1-sol.s[i,t])*sol.V_u[i,t+1])

    return sol.V_e, sol.V_u, sol.s


def solve_prepost(par, sol):

    T_pre = par.pre.T3
    T_post = par.post.T3

    if par.unobs_het == True:
        type_count = len(par.kappa)
    else:
        type_count = 1


    for i in range(type_count):

        # pre-reform
        pre_w = par.pre.w

        for t in range(T_pre, -1, -1):
            if t == T_pre:
                sol.pre.V_e[i,t] = value_employ_ss(par, pre_w)
                _, sol.pre.V_u[i,t] = value_unemploy_ss(par, sol.pre.V_e[i,t], i)

            else:
                future_val = sol.pre.V_e[i,t+1] - sol.pre.V_u[i,t+1]
                sol.pre.s[i,t] = s_policy(par, future_val, i)

                sol.pre.V_e[i,t] = value_employ_ss(par, pre_w)

                if t<par.pre.T1:
                    b = par.pre.b1
                elif par.pre.T1 <= t < par.pre.T2:
                    b = par.pre.b2
                elif par.pre.T2 <= t < par.pre.T3:
                    b = par.pre.b3

                sol.pre.V_u[i,t] = consump_util(par, b) - search_cost(sol.pre.s[i,t], par, i) + par.delta * (sol.pre.s[i,t]*sol.pre.V_e[i,t+1] + (1-sol.pre.s[i,t])*sol.pre.V_u[i,t+1])
        
        # post-reform
        post_w = par.post.w

        for t in range(T_post, -1, -1):
            if t == T_post:
                sol.post.V_e[i,t] = value_employ_ss(par, post_w)
                _, sol.post.V_u[i,t] = value_unemploy_ss(par, sol.post.V_e[i,t], i)

            else:
                future_val = sol.post.V_e[i,t+1] - sol.post.V_u[i,t+1]
                sol.post.s[i,t] = s_policy(par, future_val, i)

                sol.post.V_e[i,t] = value_employ_ss(par, post_w)

                if t<par.post.T1:
                    b = par.post.b1
                elif par.post.T1 <= t < par.post.T2:
                    b = par.post.b2
                elif par.post.T2 <= t < par.post.T3:
                    b = par.post.b3

                sol.post.V_u[i,t] = consump_util(par, b) - search_cost(sol.post.s[i,t], par, i) + par.delta * (sol.post.s[i,t]*sol.post.V_e[i,t+1] + (1-sol.post.s[i,t])*sol.post.V_u[i,t+1])
