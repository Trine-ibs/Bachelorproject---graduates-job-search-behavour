import pandas as pd
import numpy as np
from types import SimpleNamespace


class ModelClass():

    def __init__(self,
                 file_pre1="Data/HR_pre_nonpar_u30.xlsx",
                 file_pre2="Data/HR_pre_nonpar_o30.xlsx",
                 file_pre3="Data/HR_pre_par.xlsx",
                 file_post1="Data/HR_post_nonpar_u30.xlsx",
                 file_post2="Data/HR_post_nonpar_o30.xlsx",
                 file_post3="Data/HR_post_par.xlsx",
                 omega_pre1="Data/omega_pre_nonpar_u30.xlsx",
                 omega_pre2="Data/omega_pre_nonpar_o30.xlsx",
                 omega_pre3="Data/omega_pre_par.xlsx",
                 omega_post1="Data/omega_post_nonpar_u30.xlsx",
                 omega_post2="Data/omega_post_nonpar_o30.xlsx",
                 omega_post3="Data/omega_post_par.xlsx",

                 n_types=3,
                 util_type="log",
                 delta_fixed=False,
                 kappa3_fixed=False,
                 beta_shared=False,
                 groups=("post1", "post2")):


        self.file_pre1    = file_pre1
        self.file_pre2    = file_pre2
        self.file_pre3    = file_pre3
        self.file_post1   = file_post1
        self.file_post2   = file_post2
        self.file_post3   = file_post3
        self.omega_pre1   = omega_pre1
        self.omega_pre2   = omega_pre2
        self.omega_pre3   = omega_pre3
        self.omega_post1  = omega_post1
        self.omega_post2  = omega_post2
        self.omega_post3  = omega_post3
        self.n_types      = n_types
        self.groups       = groups
        self.util_type    = util_type
        self.delta_fixed  = delta_fixed
        self.kappa3_fixed = kappa3_fixed
        self.beta_shared  = beta_shared

        self.settings()
        for name in self.namespaces:
            setattr(self, name, SimpleNamespace())
        self.setup()
        self.allocate()


    def settings(self):
        self.namespaces = ['par', 'sol', 'data']


    def _group_par_defaults(self, gpar, w, b1, b2, b3, T1, T2, T3,
                             kappa0, type_shares0, alpha0, beta0):
        
        gpar.w  = w
        gpar.b1 = b1
        gpar.b2 = b2
        gpar.b3 = b3
        gpar.T1 = T1
        gpar.T2 = T2
        gpar.T3 = T3

        gpar.kappa        = kappa0.copy()
        gpar.kappa0       = kappa0.copy()
        gpar.type_shares  = type_shares0.copy()
        gpar.type_shares0 = type_shares0.copy()

        gpar.kappa0_trans       = self._kappa_to_trans(kappa0)
        gpar.type_shares0_trans = self._shares_to_trans(type_shares0)

        gpar.alpha  = alpha0
        gpar.alpha0 = alpha0
        gpar.beta   = beta0
        gpar.beta0  = beta0

        if self.util_type in ("linear", "quadratic", "CRRA", "CRRA_shift", "CARA"):
            gpar.alpha0_trans = np.log(gpar.alpha0)
        if self.util_type in ("log_shift", "CRRA_shift", "sqrt_shift"):
            gpar.beta0_trans  = np.log(gpar.beta0 + gpar.b3)


    def setup(self):
        data = self.data
        par  = self.par

        # ── load data 
        file_map = {
            "pre1":  self.file_pre1,  "pre2":  self.file_pre2,  "pre3":  self.file_pre3,
            "post1": self.file_post1, "post2": self.file_post2, "post3": self.file_post3,
        }
        omega_map = {
            "pre1":  self.omega_pre1,  "pre2":  self.omega_pre2,  "pre3":  self.omega_pre3,
            "post1": self.omega_post1, "post2": self.omega_post2, "post3": self.omega_post3,
        }

        for g in self.groups:
            gdata         = SimpleNamespace()
            df            = pd.read_excel(file_map[g])
            gdata.moments = df["HR_logit"]
            gdata.var     = df["HR_var"]
            gdata.upper   = df["Upper_CI"]
            gdata.lower   = df["Lower_CI"]
            gdata.omega = pd.read_excel(omega_map[g], index_col=0).to_numpy()
            setattr(data, g, gdata)

        

        # ── shared structural parameters 
        par.n_types = self.n_types
        par.gamma   = 1.0
        par.delta   = 0.90

        par.gamma0       = 1.0
        par.delta0       = 0.90
        par.gamma0_trans = np.log(par.gamma0)
        par.delta0_trans = np.log(par.delta0 / (1 - par.delta0))

        # ── shared beta 
        par.beta         = -300.0
        par.beta0        = -300.0
        par.beta_shared  = self.beta_shared

        # ── initial kappa and type_shares
        if par.n_types == 1:
            kappa0       = np.array([30.0])
            type_shares0 = np.array([1.0])
        elif par.n_types == 2:
            kappa0       = np.array([30.0, 50.0])
            type_shares0 = np.array([0.4, 0.6])
        elif par.n_types == 3:
            kappa0       = np.array([5.0, 9.0, 100.0])
            type_shares0 = np.array([0.4, 0.3, 0.3])
        elif par.n_types == 4:
            kappa0       = np.array([30.0, 60.0, 120.0, 2000.0])
            type_shares0 = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            raise ValueError("n_types must be 1, 2, 3, or 4")

        # ── initial alpha 
        par.alpha  = 2.0
        par.alpha0 = 2.0

        # ── group-specific income and time parameters 
        group_defaults = {
            "pre1":  dict(w=8020, b1=3522, b2=3522, b3=1922*(1-0.33), T1=0,  T2=106), # non-par, under 30
            "pre2":  dict(w=7867, b1=3522, b2=3522, b3=2982*(1-0.29), T1=0,  T2=106), # non-par, over 30
            "pre3":  dict(w=7732, b1=4039, b2=4039, b3=3595*(1-0.29), T1=0,  T2=106), # parents
            "post1": dict(w=8020, b1=3429, b2=2358, b3=1871*(1-0.42), T1=0,  T2=54),  # non-par, under 30
            "post2": dict(w=7867, b1=3429, b2=2978, b3=2903*(1-0.35), T1=0,  T2=54),  # non-par, over 30
            "post3": dict(w=7732, b1=3932, b2=3932, b3=3543*(1-0.32), T1=0,  T2=54),  # parents
        }

        for g in self.groups:
            gdata    = getattr(data, g)
            gpar     = SimpleNamespace()
            T3       = len(gdata.moments)
            defaults = group_defaults[g]
            self._group_par_defaults(
                gpar,
                w=defaults["w"],   b1=defaults["b1"],
                b2=defaults["b2"], b3=defaults["b3"],
                T1=defaults["T1"], T2=defaults["T2"], T3=T3,
                kappa0=kappa0, type_shares0=type_shares0,
                alpha0=par.alpha0, beta0=par.beta0)
            setattr(par, g, gpar)

        # ── estimation options 
        par.weight        = True
        par.moments_start = 6
        par.moments_end   = None
        par.util_type     = self.util_type
        par.delta_fixed   = self.delta_fixed
        par.kappa3_fixed  = self.kappa3_fixed


    def _kappa_to_trans(self, kappa0):
        K        = len(kappa0)
        trans    = np.empty(K)
        trans[0] = np.log(kappa0[0])
        for i in range(1, K):
            trans[i] = np.log(kappa0[i] - kappa0[i-1])
        return trans


    def _shares_to_trans(self, shares0):
        if len(shares0) > 1:
            return np.log(shares0[:-1] / shares0[-1])
        else:
            return np.array([])


    def allocate(self):
        sol = self.sol
        par = self.par

        for g in self.groups:
            gpar = getattr(par, g)
            T    = gpar.T3

            gsol         = SimpleNamespace()
            gsol.s       = np.zeros((par.n_types, T))
            gsol.V_u     = np.zeros((par.n_types, T+1))
            gsol.V_e     = np.zeros((par.n_types, T+1))
            gsol.moments = np.zeros(T)
            setattr(sol, g, gsol)


    def print_par_prepost(self, demo_groups, group_names=None):
        par = self.par

        if group_names is None:
            group_names = {subgroups[0]: subgroups[0] for subgroups in demo_groups}

        rows = []

        for k in range(par.n_types):
            row = {"Parameter": f"kappa{k+1}"}
            for subgroups in demo_groups:
                g = subgroups[0]
                row[group_names[g]] = getattr(par, g).kappa[k]
            rows.append(row)

        for k in range(par.n_types):
            row = {"Parameter": f"type_share{k+1}"}
            for subgroups in demo_groups:
                g = subgroups[0]
                row[group_names[g]] = getattr(par, g).type_shares[k]
            rows.append(row)

        if par.util_type in ("linear", "quadratic", "CRRA", "CRRA_shift", "CARA"):
            row = {"Parameter": "alpha"}
            for subgroups in demo_groups:
                g = subgroups[0]
                row[group_names[g]] = getattr(par, g).alpha
            rows.append(row)

        if par.util_type in ("log_shift", "CRRA_shift", "sqrt_shift"):
            if par.beta_shared:
                rows.append({"Parameter": "beta (shared)", **{group_names[subgroups[0]]: par.beta for subgroups in demo_groups}})
            else:
                row = {"Parameter": "beta"}
                for subgroups in demo_groups:
                    g = subgroups[0]
                    row[group_names[g]] = getattr(par, g).beta
                rows.append(row)

        df_group = pd.DataFrame(rows).set_index("Parameter").round(8)

        shared_params = [("gamma", par.gamma), ("delta", par.delta)]
        if par.util_type in ("log_shift", "CRRA_shift", "sqrt_shift") and par.beta_shared:
            shared_params.append(("beta", par.beta))
        df_shared = pd.DataFrame([
            {"Parameter": name, "Value": val}
            for name, val in shared_params
        ]).set_index("Parameter").round(6)

        return df_group, df_shared


    def update_initial_values(self, gamma0=None, delta0=None, kappa0=None,
                               type_shares0=None, alpha0=None, beta0=None):
        par = self.par

        if gamma0 is not None:
            par.gamma0       = gamma0
            par.gamma0_trans = np.log(gamma0)
            par.gamma        = gamma0

        if delta0 is not None:
            par.delta0       = delta0
            par.delta0_trans = np.log(delta0 / (1 - delta0))
            par.delta        = delta0

        if beta0 is not None and par.beta_shared:
            par.beta0 = beta0
            par.beta  = beta0

        for g in self.groups:
            gpar = getattr(par, g)

            if kappa0 is not None:
                gpar.kappa0       = np.array(kappa0)
                gpar.kappa0_trans = self._kappa_to_trans(gpar.kappa0)
                gpar.kappa        = gpar.kappa0.copy()

            if type_shares0 is not None:
                shares                  = np.array(type_shares0) / np.sum(type_shares0)
                gpar.type_shares0       = shares
                gpar.type_shares0_trans = self._shares_to_trans(shares)
                gpar.type_shares        = shares.copy()

            if alpha0 is not None:
                gpar.alpha0       = alpha0
                gpar.alpha0_trans = np.log(alpha0)
                gpar.alpha        = alpha0

            if beta0 is not None and not par.beta_shared:
                gpar.beta0       = beta0
                gpar.beta0_trans = np.log(beta0 + gpar.b3)
                gpar.beta        = beta0


    def update_group_initial_values(self, g, kappa0=None, type_shares0=None,
                                     alpha0=None, beta0=None, gamma0=None):
        par  = self.par
        gpar = getattr(par, g)

        if kappa0 is not None:
            gpar.kappa0       = np.array(kappa0)
            gpar.kappa0_trans = self._kappa_to_trans(gpar.kappa0)
            gpar.kappa        = gpar.kappa0.copy()

        if type_shares0 is not None:
            shares                  = np.array(type_shares0) / np.sum(type_shares0)
            gpar.type_shares0       = shares
            gpar.type_shares0_trans = self._shares_to_trans(shares)
            gpar.type_shares        = shares.copy()

        if alpha0 is not None:
            gpar.alpha0       = alpha0
            gpar.alpha0_trans = np.log(alpha0)
            gpar.alpha        = alpha0

        if beta0 is not None:
            if par.beta_shared:
                import warnings
                warnings.warn(f"beta_shared=True: beta0 set on group {g} ignored. "
                              "Use update_initial_values(beta0=...) instead.")
            else:
                gpar.beta0       = beta0
                gpar.beta0_trans = np.log(beta0 + gpar.b3)
                gpar.beta        = beta0

        if gamma0 is not None:
            gpar.gamma0       = gamma0
            gpar.gamma0_trans = np.log(gamma0)
            gpar.gamma        = gamma0
