import cupy as cp
from cupyx.scipy.special import gammaln
from scipy.optimize import brentq, newton, bisect
from sklearn.cluster import KMeans
import numpy as np  
from scipy.special import digamma as sp_digamma
from cupyx.scipy.special import gamma as cp_gamma
from cupyx.scipy.linalg import solve_triangular
from cupyx.scipy.special import logsumexp

class FiniteMixutureScaleMixtureModel:
    """
    Finite Mixture of Scale Mixture Models
    Parameters
    ----------
        n_component : int, default=1
            The number of components.

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

    def __init__(self, n_component=1, method='bisect', nu_fix=0, max_iter=5000, tol=1e-4, verbose=False):

        self.n_component = n_component  # Number of components

        self.log_marginal_likelihood_values = [] # History of log-likelihood

        self.method = method
        self.nu_fix = nu_fix
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.seed = None 

    def fit(self, X, seed=None):
        self.seed = seed
        X = cp.asarray(X)
        self.n_samples, self.n_dim = X.shape

        self._allocate_parameters_memory(X)
        self.initialize_param(X,seed=self.seed)
        pre_lml = self.log_marginal_likelihood(X)

        for i in range(self.max_iter):
            # print(f"iter:{i}")
            self._e_step(X)
            self._cm_step1(X)
            if self.nu_fix == 0:
                self._cm_step2_optimized()

            lml = self.log_marginal_likelihood(X)
            self.log_marginal_likelihood_values.append(lml)

            if _is_converged(pre_lml, lml, self.tol):
                labels = cp.argmax(self.gamma, axis=1)
                return lml, labels

            pre_lml = lml

        labels = cp.argmax(self.gamma, axis=1)
        self.bic = self.compute_bic(X)
        return lml, labels

    def initialize_param(self, X, seed=None):
        if seed is not None:
            np.random.seed(seed)
            cp.random.seed(seed)
            cp_rng = cp.random.RandomState(seed)  
        else:
            cp_rng = cp.random

        # KMeans on CPU
        X_cpu = cp.asnumpy(X) if hasattr(X, "get") else np.asarray(X)

        kmeans_ = KMeans(
            n_clusters=self.n_component,
            max_iter=20,
            random_state=seed,   
            n_init=10
        ).fit(X_cpu)

        self.mu = cp.asarray(kmeans_.cluster_centers_)

        for k in range(self.n_component):
            idx_ = np.where(kmeans_.labels_ == k)[0]
            self.psi[k] = cp.diag(cp.var(X[idx_].T, axis=1, ddof=1))
            self.pi[k] = cp.asarray(np.size(idx_) / self.n_samples)
            if self.nu_fix == 0:
                self.nu[k] = cp.asarray((20.0 - 2.0) * cp_rng.rand() + 2.0)
            else:
                self.nu[k] = self.nu_fix

    def _e_step(self, X):
        """
        E_step
        """
        n_samples = X.shape[0]
        delta_ = self._delta
        log_resp = self._log_resp

        for k in range(self.n_component):
            diffs_ = X - self.mu[k]

            L = cp.linalg.cholesky(self.psi[k])
            log_det = 2.0 * cp.sum(cp.log(cp.diag(L)))
            y = solve_triangular(L, diffs_.T, lower=True)
            delta_[:, k] = cp.sum(y * y, axis=0)
            self.omega[:, k] = (self.nu[k] + self.n_dim) / (self.nu[k] + delta_[:, k])
            log_prob = log_pdf_smm(X, nu=self.nu[k], mu=self.mu[k], psi=self.psi[k],delta_=delta_[:, k],log_det_psi=log_det)
            
            log_resp[:, k] = cp.log(self.pi[k] + 1e-300) + log_prob

        log_denominator = logsumexp(log_resp, axis=1, keepdims=True)
        
        self.gamma = cp.exp(log_resp - log_denominator)



    def _cm_step1(self, X: cp.ndarray):
        """
        M_step1
        """
        n_samples, n_dim = X.shape
        k = self.n_component

        gamma = self.gamma        
        omega = self.omega        
        weights = gamma * omega   

        sum_g = cp.sum(gamma, axis=0)         
        sum_w = cp.sum(weights, axis=0)       

        self.pi = sum_g / n_samples                  
        self.mu = (weights.T @ X) / sum_w[:, None]   

        for k in range(self.n_component):
            w_k = weights[:, k]               
            diffs_k = X - self.mu[k]          
            
            self.psi[k] = (diffs_k.T * w_k) @ diffs_k / sum_g[k]

            self.psi[k] = 0.5 * (self.psi[k] + self.psi[k].T)
            eps = 1e-6 * max(cp.trace(self.psi[k]) / n_dim, 1.0)
            self.psi[k].flat[:: n_dim + 1] += eps

            self.psi[k] = _check_psi(self.psi[k])

    def _cm_step2_optimized(self):
        """
        CM-step2 
        """
        D = int(self.n_dim)

        n_k_gpu = cp.sum(self.gamma, axis=0)  
        term_gpu = cp.sum(self.gamma * (cp.log(self.omega) - self.omega), axis=0)  

        n_k = cp.asnumpy(n_k_gpu)     
        term = cp.asnumpy(term_gpu)   
        nu_new = np.empty(self.n_component, dtype=np.float64)

        # CPU root-finding
        for k in range(self.n_component):
            n_comp = float(n_k[k])

            if n_comp <= 0.0:
                nu_new[k] = 200.0
                continue

            t_k = float(term[k])

            def f(v):
                return (-sp_digamma(0.5 * v)
                        + np.log(0.5 * v)
                        + 1.0 + (t_k / n_comp)
                        + sp_digamma(0.5 * (v + D))
                        - np.log(0.5 * (v + D)))

            try:
                if self.method in {'bi', 'bisect', 'bisection'}:
                    res = bisect(f, 0.1, 200.0, xtol=1e-3)
                elif self.method in {'new', 'newton'}:
                    res = newton(f, x0=2.0, tol=1e-3, maxiter=50)
                    if res > 200.0:
                        res = 200.0
                elif self.method in {'brent', 'brentq'}:
                    res = brentq(f, 0.1, 200.0, xtol=1e-3)
                else:
                    raise ValueError("method must be bisect/newton/brentq")
            except (ValueError):
                res = 200.0
            if res < 0.1:
                res = 0.1
            if res > 200.0:
                res = 200.0
            nu_new[k] = res

        self.nu = cp.asarray(nu_new, dtype=self.nu.dtype)



    def _allocate_parameters_memory(self,X):
        """
        allocate memory
        """
        self.nu = cp.zeros(self.n_component)
        self.pi = cp.zeros(self.n_component)
        self.psi = cp.zeros((self.n_component, self.n_dim, self.n_dim))
        self.gamma = cp.zeros((self.n_samples, self.n_component))
        self.omega = cp.zeros((self.n_samples, self.n_component))
        self._log_resp = cp.empty((self.n_samples, self.n_component), dtype=X.dtype)
        self._delta    = cp.empty((self.n_samples, self.n_component), dtype=X.dtype)


    def log_marginal_likelihood(self, X):

        n, _ = X.shape
        K = self.n_component

        log_comp = cp.zeros((n, K))
        for k in range(K):
            log_comp[:, k] = cp.log(self.pi[k] + 1e-300) + \
                            log_pdf_smm(X, self.nu[k], self.mu[k], self.psi[k])
        return logsumexp(log_comp, axis=1).sum()

    
    def get_history_of_loglikelihood(self):
        """Get history of log-likelihood"""
        return self.log_marginal_likelihood_values
    
    def compute_bic(self, X_train):
        """
        Compute BIC for FMSMM model
        """
        if not isinstance(X_train, cp.ndarray):
        # 先尝试由 NumPy 转换，否则直接用 asarray 包装
            try:
                X_train = cp.asarray(X_train,dtype=cp.float32)
            except Exception:
                raise TypeError("X_train must be array-like (NumPy or CuPy), got %r" % type(X_train))
            
        n_components = self.n_component
        n_dim = self.n_dim
        n_samples = X_train.shape[0]

        # Determine parameter count (k)
        num_mu = n_components * n_dim
        num_psi = n_components * (n_dim * (n_dim + 1)) // 2
        num_nu = 0 if self.nu_fix > 0 else n_components
        num_pi = n_components - 1
        k = int(num_mu + num_psi + num_nu + num_pi)
        log_likelihood = self.log_marginal_likelihood(X_train)
        bic_value = cp.log(n_samples) * k - 2 * log_likelihood
        return bic_value
    
    def compute_icl(self, X_train):
        """
        Compute ICL (Integrated Complete-data Likelihood) for FMSMM model.
        Formula: ICL = BIC - 2 * sum_{i,k} (gamma_{ik} * ln(gamma_{ik}))
        
        Parameters
        ----------
        X_train : array-like (n_samples, n_dim)
            Data for calculating ICL.
            
        Returns
        -------
        icl_value : float
            The ICL value. Lower is better.
        """
        # 1. 确保输入是 cupy 数组
        if not isinstance(X_train, cp.ndarray):
            try:
                X_train = cp.asarray(X_train)
            except Exception:
                raise TypeError("X_train must be array-like (NumPy or CuPy), got %r" % type(X_train))
        
        bic_value = self.compute_bic(X_train)
        eps = 1e-300
        entropy_term = cp.sum(self.gamma * cp.log(self.gamma + eps))
        icl_value = bic_value - 2.0 * entropy_term
        
        return icl_value

def _check_psi(psi):
    n_dim, _ = psi.shape

    sign, _ = cp.linalg.slogdet(psi)
    if sign <= 0:
        print("psi is not PD; add small diagonal jitter.")
        return psi + cp.eye(n_dim, dtype=psi.dtype) * 1e-5
    return psi



def _is_converged(pre_J, J, tol):
    """Check convergence
    """
    if cp.abs(pre_J - J) < tol:
        return True
    else:
        return False

def pdf_smm(X, nu, mu, psi):
    """Calculate probability density function of scale mixture model

    Parameters
    ----------
        x:      input data
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
    """
    _, n_dim = X.shape

    diffs_ = X.T - mu.reshape(-1, 1)
    delta_ = cp.sum(diffs_ * cp.linalg.solve(psi, diffs_), axis=0)

    p_x_ = cp_gamma((nu + n_dim)/2)/cp_gamma(nu/2) * \
            cp.linalg.det(psi)**(-1/2)/((cp.pi*nu)**(n_dim/2)) * \
            (1 + delta_/nu)**(-(nu + n_dim)/2)

    return p_x_.T

def log_pdf_smm(X, nu, mu, psi,delta_=None,log_det_psi=None):
    """Calculate log-probability density function of scale mixture model
    Parameters
    ----------
        x:      input data
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
    Returns
    -------
        log_prob : ndarray, shape=(n_samples)
            Log-probability from a scale mixutre model
            > log p(x_n|z_n)
    """
    _, n_dim = X.shape

    if delta_ is None:
        diffs_ = X.T - mu.reshape(-1, 1)
        delta_ = cp.sum(diffs_ * cp.linalg.solve(psi, diffs_), axis=0)
    if log_det_psi is None:
        sign, log_det_psi = cp.linalg.slogdet(psi)  # fallback
    
    log_prob = gammaln((nu + n_dim) / 2.) - gammaln(nu / 2.) - \
                0.5 * log_det_psi - 0.5 * n_dim * cp.log(nu * cp.pi) - \
                0.5*(nu+n_dim) * cp.log1p(delta_/nu)

    return log_prob.T


def pdf(X, pi, nu, mu, psi):
    """Calculate log-probability density function of finite mixture of scale mixture model
    Parameters:
        x:      input data
        pi:     mixising coeffients
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
    """
    n_component = pi.size

    p_X_ = 0.0
    for k in range(n_component):
        p_X_ += pi[k] * pdf_smm(X, nu[k], mu[k], psi[k])

    return p_X_.T

def log_pdf(X, pi, nu, mu, psi):
    """Calculate log-probability density function of finite mixture of scale mixture model
    Parameters:
        x:      input data
        pi:     mixising coeffients
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
    """

    ln_p_X_ = cp.log(pdf(X, pi, nu, mu, psi))

    return ln_p_X_.T
