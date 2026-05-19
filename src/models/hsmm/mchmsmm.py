from ..smm.fmsmm import FiniteMixutureScaleMixtureModel as FMSMM
import numpy as np
import numpy.linalg as LA
import pandas as pd
import string
from scipy.special import logsumexp
from scipy.special import gammaln
from . import _hmmc  # Added
import math
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from kneed import KneeLocator
import cupy as cp
from multiprocessing import Manager
import random

class HiddenMarkovScaleMixtureModel_elbow:
    """
    Hidden Markov-based Time-series Scale Mixture Model

    Parameters
    ----------
        n_state : int, default=2
            The number of states.

        method : {'bisect', 'newton', 'brentq'}, default='bisect'
            Method for optimization of nu.

        nu_fix : float, default=0
            Set fixed nu. 'nu_fix=0' means nu is optimized during EM.
            If nu_fix set to > 0, EM iterates with fixed nu.

        max_iter : int, default=5000
            Maximum number of iterations of EM algorithm.

        tol : float, default=1e-4
            Relative tolerance with regards to log-marginal likelihood to declare convergence

    """

    def __init__(self, n_states=2, method='bisect', nu_fix=None,
                 max_iter=5000, tol=1e-4, reg_flag=False,
                 verbose=False, params=string.ascii_letters, **kwargs):

        self.n_states = n_states  # Number of states

        self.log_marginal_likelihood_values = []  # History of log-likelihood
        self.history = []
        self.bic_dfs = {}
        self.method = method
        if nu_fix is None:
            self.nu_fix = np.zeros(self.n_states)
        else:
            self.nu_fix = nu_fix
        self.max_iter = max_iter
        self.tol = tol
        self.reg_flag = reg_flag
        self.verbose = verbose
        self._init_param_method = 'k-means'
        self.params = params

        self._init_manually(**kwargs)

    def _init_manually(self, components_per_state=None, component_range=None, pi=None, A=None, mu=None, psi=None, nu=None, pi_m=None):
        """Initialize parameters manually

        Parameters
        ----------
            pi : initial state

            A : transitions matrix

            mu : ndarray, shape=(n_states, n_dim)
                Mean parameters for each state

            psi : ndarray, shape=(n_states, n_dim, n_dim)
                Scale matrix for each state

            nu : ndarray, shape=(n_states), default=None
                Degrees of freedom for each state
        """
        if components_per_state is None:
            self.components_per_state = [None] * self.n_states
        else:
            self.components_per_state = components_per_state
        self.component_range = component_range

        self.pi = pi if (pi is not None) else np.random.dirichlet(
            alpha=np.ones(self.n_states))
        self.A = A if (A is not None) else np.random.dirichlet(
            alpha=np.ones(self.n_states), size=self.n_states)
        self.mu = mu
        self.psi = psi
        self.nu = nu
        self.pi_m = pi_m

    def fit(self, X, y=None, lengths=None):
        """
        Parameters
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        y : array-like of integers, shape (n_samples, ) 
            Label vector of the states for each sequences in X

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be n_samples.
        """
        self.n_samples, self.n_dim = X.shape
        self.supervised = True if (y is not None) else False

        if self.supervised:
            print('Supervised learning of all combination')
            n_states_ = np.unique(y).size
            _n_states_ = len(self.components_per_state)
            self._check_n_state(n_states_, _n_states_)
            if self.components_per_state is None or any(c is None for c in self.components_per_state):
                # train all fmsmms
                self.fmsmm_models = train_all_dynamic(X, y, n_states=self.n_states, method=self.method, tol=self.tol, nu_fix=self.nu_fix,component_range=self.component_range)
                for s in range(self.n_states):
                    bic_df, best_n = elbow(self.fmsmm_models,s)
                    self.bic_dfs[s] = bic_df
                    self.components_per_state[s] = best_n
            else:
                # train all fmsmms using fixed components for each state
                self.fmsmm_models = train_all_fixed(X, y, n_states=self.n_states, components_per_state=self.components_per_state,method=self.method, tol=self.tol, nu_fix=self.nu_fix)
            print('Learning done')
        
        self._allocate_parameters_memory()
        combination = [c - 1 for c in self.components_per_state]
        self.selected_fmsmms = [self.fmsmm_models[s][c] for s, c in enumerate(combination)]
        for state_index, fmsmm in enumerate(self.selected_fmsmms):
            n_mixtures = combination[state_index] + 1  
            for mixture_index in range(n_mixtures):
                self._get_params(fmsmm, state_index, mixture_index)

        for iter in range(self.max_iter):
            stats = self._initialize_sufficients_statistics()
            log_prob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                n_samples, _ = X[i:j].shape
                log_emission_probs = np.zeros((n_samples, self.n_states))

                for k in range(self.n_states):
                    log_emission_probs[:, k] = log_pdf(
                        X[i:j], self.pi_m[k], self.nu[k], self.mu[k], self.psi[k])

                log_alpha, frame_log_prob = self._forward(log_emission_probs)
                log_prob += frame_log_prob
                log_beta = self._backward(log_emission_probs)
                gamma, xi = self._e_step(
                    X[i:j], log_emission_probs, log_alpha, log_beta, frame_log_prob)

                self._accumulate_sufficient_statistics(
                    stats, gamma, xi)
                self.gamma[i:j] = gamma
            self._m_step(X, stats)

            self._report_likelihood(iter, log_prob)
            if self._is_converged(iter):
                print('Learning done')
                break

        return self


    def _get_params(self, model, state_index, component_index):
        self.mu[state_index][component_index] = model["mu"][component_index]
        self.psi[state_index][component_index] = model["psi"][component_index]
        self.nu[state_index][component_index] = model["nu"][component_index]
        self.pi_m[state_index][component_index] = model["pi"][component_index]


    def _forward(self, log_emission_probs):
        """
        Forward process
        """
        n_samples, _ = log_emission_probs.shape

        log_alpha = np.zeros((n_samples, self.n_states))
        _hmmc._forward(n_samples, self.n_states,
                       log_mask_zero(self.pi),
                       log_mask_zero(self.A),
                       log_emission_probs, log_alpha)

        frame_log_prob = logsumexp(log_alpha[-1])
        return log_alpha, frame_log_prob

    def _backward(self, log_emission_probs):
        """
        Backward process
        """
        n_samples, _ = log_emission_probs.shape

        log_beta = np.zeros((n_samples, self.n_states))
        log_beta[n_samples - 1] = np.ones(self.n_states)  # zeros -> ones

        _hmmc._backward(n_samples, self.n_states,
                        log_mask_zero(self.pi),
                        log_mask_zero(self.A),
                        log_emission_probs, log_beta)

        return log_beta

    def _e_step(self, X, log_emission_probs, log_alpha, log_beta, frame_log_prob):
        """E-step in EM iterations

        Parameters
        ----------
            X : ndarray or sparse matrix, shape=(n_samples, n_dim)
                Data for estimation

            log_emission_probs : ndarray, shape=(n_samples, n_state) 
                Logarithm of emission probability
                > log p(x_n|z_n) 

            log_alpha : ndarray, shape=(n_samples, n_state)
                Logarithm of alpha in forward-backward algorithm
                > log p(x_1, ..., x_n, z_n)

            log_beta : ndarray, shape=(n_samples, n_state)
                Logarithm of beta in forward-backward algorithm
                > log p(x_n+1, ..., x_N|z_n)

            frame_log_prob : ndarray, shape=(n_samples)
                Log-marginal likelihood
                > log p(X)

        """
        n_samples, _ = X.shape

        log_xi = np.zeros((n_samples - 1, self.n_states, self.n_states))

        gamma = self._compute_posteriors(log_alpha, log_beta)

        for k1 in range(self.n_states):
            for k2 in range(self.n_states):
                log_xi[:, k1, k2] = log_alpha[0:-1, k1] + \
                    log_mask_zero(self.A[k1, k2]) + \
                    log_emission_probs[1:, k2] + log_beta[1:, k2] - frame_log_prob

        xi = np.exp(log_xi)

        return gamma, xi

    def _compute_posteriors(self, log_alpha, log_beta):
        """
        """
        ln_gamma = log_alpha + log_beta
        log_normalize(ln_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(ln_gamma)

    def _initialize_sufficients_statistics(self):
        """
        """
        self.gamma = np.zeros(shape=(self.n_samples, self.n_states))
        self.omega = np.zeros(shape=(self.n_samples, self.n_states))
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_states),
                 'trans': np.zeros((self.n_states, self.n_states)),
                 'centered': None,
                 'gamma * omega': None}

        return stats

    def _accumulate_sufficient_statistics(self, stats, gamma, xi):
        """
        """
        stats['nobs'] += 1
        stats['start'] += gamma[0]

        xi_sum = np.sum(xi, axis=0)
        stats['trans'] += xi_sum


    def _m_step(self, X, stats):
        """M-step in EM iterations

        Parameters
        ----------
            X : ndarray or sparse matrix, shape=(n_samples, n_dim)
                Data for estimation
        """

        self.pi = stats['start'] / np.sum(stats['start'])
        self.A = stats['trans'] / \
            np.reshape(np.sum(stats['trans'], axis=1), (self.n_states, 1))

    def _allocate_parameters_memory(self):
        """Allocate memory for parameters"""
        self.mu = [None] * self.n_states
        self.psi = [None] * self.n_states
        self.nu = [None] * self.n_states
        self.pi_m = [None] * self.n_states

        for s in range(self.n_states):
            self.mu[s] = np.zeros((self.components_per_state[s], self.n_dim))
            self.psi[s] = np.zeros((self.components_per_state[s], self.n_dim, self.n_dim))
            self.nu[s] = np.zeros(self.components_per_state[s])
            self.pi_m[s] = np.zeros(self.components_per_state[s])

    def predict_proba(self, X):
        """
        """
        n_samples, _ = X.shape

        log_emission_probs = np.zeros((n_samples, self.n_states))
        for k in range(self.n_states):
            log_emission_probs[:, k] = log_pdf(X, self.pi_m[k],self.nu[k], self.mu[k], self.psi[k])

        log_alpha, frame_log_prob = self._forward(log_emission_probs)
        log_beta = self._backward(log_emission_probs)

        ln_gamma = log_alpha + log_beta - frame_log_prob
        gamma = np.exp(ln_gamma)

        return gamma

    def _check_n_state(self, n_states, _n_states_):
        """
        """
        if self.n_states != n_states or self.n_states != _n_states_:
            raise ValueError("n_states or _n_staes_ must be same length as self.n_state")

    def _report_likelihood(self, iter, log_prob):
        """
        """
        if self.verbose:
            delta = log_prob - self.history[-1] if self.history else np.nan
            message = "{iter:>10d} {log_prob:>16.4f} {delta:>+16.4f}".format(
                iter=iter+1, log_prob=log_prob, delta=delta)
            # print(message)

        self.history.append(log_prob)

    def _is_converged(self, iter):
        """Check convergence
        """
        return (iter == self.max_iter or
                (len(self.history) > 1 and
                 np.abs(self.history[-2] - self.history[-1]) < self.tol))


def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def log_normalize(a, axis=None):
    """
    Normalizes the input array so that ``sum(exp(a)) == 1``.
    Parameters
    ----------
    a : array
        Non-normalized input data.
    axis : int
        Dimension along which normalization is performed.
    Notes
    -----
    Modifies the input **inplace**.
    """
    if axis is not None and a.shape[axis] == 1:
        a[:] = 0
    else:
        with np.errstate(under="ignore"):
            a_lse = logsumexp(a, axis, keepdims=True)
        a -= a_lse


def pdf_smm(X, nu, mu, psi):
    """Calculate probability density function of scale mixture model

    Parameters
    ----------
        x:      input data
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix

    Returns
    -------
        prob : ndarray, shape=(n_samples)
            Probability from a scale mixutre model
            > p(x_n|z_n)

    """
    _, n_dim = X.shape

    diffs_ = X.T - mu.reshape(-1, 1)
    try:
        delta_ = np.sum(diffs_ * (np.dot(LA.inv(psi), diffs_)), axis=0)
    except np.linalg.LinAlgError:
        delta_ = np.sum(diffs_ * (np.dot(LA.inv(psi + 1e-7), diffs_)), axis=0)

    prob = math.gamma((nu + n_dim) / 2) / math.gamma(nu / 2) * \
        LA.det(psi)**(-1 / 2) / ((math.pi * nu)**(n_dim / 2)) * \
        (1 + delta_ / nu)**(-(nu + n_dim) / 2)

    return prob.T


def log_pdf_smm(X, nu, mu, psi):
    """Calculate log-probability density function of scale mixture model

    Parameters
    ----------
        x:      input data
        pi:     mixising coeffients
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
        pi_m:   mixising coeffients
    """
    _, n_dim = X.shape

    diffs_ = X.T - mu.reshape(-1, 1)
    #delta_ = np.sum(diffs_ * (LA.solve(psi, diffs_)), axis = 0)
    try:
        delta_ = np.sum(diffs_ * (np.dot(LA.inv(psi), diffs_)), axis=0)
    except np.linalg.LinAlgError:
        delta_ = np.sum(diffs_ * (np.dot(LA.inv(psi + 1e-7), diffs_)), axis=0)

    log_prob = gammaln((nu + n_dim) / 2.) - gammaln(nu / 2.) - \
        0.5 * np.log(LA.det(psi)) - 0.5 * n_dim * np.log(nu * np.pi) - \
        (nu + n_dim) / 2. * np.log(1 + delta_ / nu)

    return log_prob.T

def pdf(X, pi_m, nu, mu, psi):
    """Calculate probability density function of finite mixture of scale mixture model

    Parameters
    ----------
        x:      input data
        pi:     mixising coeffients
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
        pi_m:   mixising coeffients
    """
    n_component = pi_m.size

    p_X_ = 0.0
    for k in range(n_component):
        p_X_ += pi_m[k] * pdf_smm(X, nu[k], mu[k], psi[k])

    return p_X_.T

def log_pdf(X, pi_m, nu, mu, psi):
    """Calculate log-probability density function of finite mixture of scale mixture model

    Parameters
    ----------
        x:      input data
        pi:     mixising coeffients
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
    """

    ln_p_X_ = np.log(pdf(X, pi_m, nu, mu, psi))

    return ln_p_X_.T


def iter_from_X_lengths(X, lengths):
    """
    """
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield int(start[i]), int(end[i])

def init_gpu_queue(q):
    """
    """
    global gpu_queue
    gpu_queue = q

def train_task(args):
    X_cpu, y_cpu, s, m, method, tol, nu_fix, task_seed = args

    gpu_id = gpu_queue.get()
    try:
        with cp.cuda.Device(gpu_id):
            random.seed(task_seed)
            np.random.seed(task_seed)
            cp.random.seed(task_seed)

            X_state = cp.asarray(X_cpu[y_cpu == s])

            model = FMSMM(n_component=m,method=method,nu_fix=nu_fix[s],tol=tol)
            model.fit(X_state, seed=task_seed)

            bic_val = model.compute_bic(X_state)
            result_model_data = {
                "mu": cp.asnumpy(model.mu),
                "psi": cp.asnumpy(model.psi),
                "nu": cp.asnumpy(model.nu),
                "pi": cp.asnumpy(model.pi),
                "bic": float(cp.asnumpy(bic_val).item()),
                "task_seed": int(task_seed),
                "gpu_id": int(gpu_id),
            }

            del model, X_state
            cp.get_default_memory_pool().free_all_blocks()

    finally:
        gpu_queue.put(gpu_id)

    print(f"[Done] State={s+1}, mixtures={m} on GPU{gpu_id}, seed={task_seed}")
    return s, m - 1, result_model_data

def make_task_seed(base_seed: int, s: int, m: int) -> int:
    return (base_seed + s * 100000 + m * 1000) & 0xFFFFFFFF

def train_all_dynamic(X, y, n_states,method="bisect", tol=1e-4, nu_fix=None,component_range=3, gpus=None, seed=42):
    # Auto-detect GPUs if needed
    if gpus is None:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        gpus = list(range(gpu_count))
    else:
        gpus = list(gpus)

    X_cpu = cp.asnumpy(X) if isinstance(X, cp.ndarray) else np.asarray(X)
    y_cpu = cp.asnumpy(y) if isinstance(y, cp.ndarray) else np.asarray(y)

    if nu_fix is None:
        raise ValueError("nu_fix must be provided (e.g., list/array with length n_states).")

    tasks = []
    for s in range(n_states):
        for m in range(1, component_range + 1):
            task_seed = make_task_seed(seed, s, m)
            tasks.append((X_cpu, y_cpu, s, m, method, tol, nu_fix, task_seed))

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()

    gpu_queue = manager.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    fmsmm_models = {s: {} for s in range(n_states)}

    with ProcessPoolExecutor(max_workers=len(gpus),mp_context=ctx,initializer=init_gpu_queue,initargs=(gpu_queue,),) as exe:
        futures = [exe.submit(train_task, t) for t in tasks]
        for fut in as_completed(futures):
            s, mix_idx, model_data = fut.result()
            fmsmm_models[s][mix_idx] = model_data

    return fmsmm_models

def train_all_fixed(X, y, n_states, components_per_state,method='bisect', tol=1e-4, nu_fix=None, gpus=None, seed=42): 

    if gpus is None:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        gpus = list(range(gpu_count))
    else:
        gpus = list(gpus)
    if len(gpus) == 0:
        raise RuntimeError("No GPU available.")

    X_cpu = cp.asnumpy(X) if isinstance(X, cp.ndarray) else X
    y_cpu = cp.asnumpy(y) if isinstance(y, cp.ndarray) else y

    if len(components_per_state) != n_states:
        raise ValueError(f"len(components_per_state)={len(components_per_state)} must equal n_states={n_states}")

    tasks = []
    for s in range(n_states):
        m = int(components_per_state[s])
        if m < 1:
            raise ValueError(f"components_per_state[{s}] must be >= 1, got {m}")

        task_seed = make_task_seed(seed, s, m)
        tasks.append((X_cpu, y_cpu, s, m, method, tol, nu_fix, task_seed))

    ctx = mp.get_context('spawn')
    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    fmsmm_models = {s: {} for s in range(n_states)}
    max_workers = min(len(gpus), len(tasks))

    with ProcessPoolExecutor(max_workers=max_workers,mp_context=ctx,initializer=init_gpu_queue,initargs=(gpu_queue,)) as exe:
        futures = [exe.submit(train_task, t) for t in tasks]
        for fut in futures:
            s, mix_idx, model_pack = fut.result()
            fmsmm_models[s][mix_idx] = model_pack

    return fmsmm_models

def elbow(models, s):
    """
    elbow method
    """
    bic_records = []

    for m, model in models[s].items():
        bic_val = model['bic']
        bic_records.append((m + 1, bic_val))

    bic_df = pd.DataFrame(
        bic_records,
        columns=["n_component", "bic"]
    )

    bic_df.sort_values(by="n_component", inplace=True)
    bic_df.reset_index(drop=True, inplace=True)

    knee = KneeLocator(bic_df["n_component"],bic_df["bic"],curve="convex",direction="decreasing")

    if knee.knee is not None:
        best_n = int(knee.knee)
        print(f"[State {s}] Elbow detected at {best_n}")
    else:
        best_n = int(
            bic_df.loc[bic_df["bic"].idxmin(), "n_component"]
        )
        print(f"[State {s}] No elbow. Using min BIC: {best_n}")

    return bic_df, best_n