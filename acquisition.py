import numpy as onp
from jax import numpy as jnp
from jax import jit, vjp, random
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
#from jax.scipy.optimize import minimize
from functools import partial
from pyDOE import lhs
from tqdm import trange


###################################
###### Optimization Pipeline ######
###################################


def minimize_lbfgs(objective, x0, bnds = None, verbose = False, maxfun = 15000):
    if verbose:
        def callback_fn(params):
            print("Loss: {}".format(objective(params)[0]))
    else:
        callback_fn = None
        
    result = minimize(objective, x0, jac=True,
                      method='L-BFGS-B', bounds = bnds,
                      callback=callback_fn, options = {'maxfun':maxfun})
    return result.x, result.fun

class MCAcquisition:
    def __init__(self, posterior, bounds, *args, 
                acq_fn = 'US',
                norm = lambda x : jnp.linalg.norm(x, axis=-1),
                output_weights=lambda x: jnp.ones(x.shape[0]),
                sig=1):  
        self.posterior = posterior
        self.bounds = bounds            # domain bounds
        self.args = args                # arguments required by different acquisition functions
        self.acq_fn = acq_fn            # a string indicating the chosen acquisition function
        self.norm = norm
        self.weights = output_weights   # a callable function returning the likelihood weighted weights
        self.sig = sig

    def evaluate(self, x):
        # Inputs are (q x d), use vmap to vectorize across a batch
        # samples[...,0]  corresponds to the objective function
        # samples[...,1:] corresponds to the constraints
        # samples[...,i] are (q x ensemble_size x queries x values)
        q = x.shape[0]
        # Common acquisition functions
        if self.acq_fn == 'US': # Uncertainty Sampling
            samples = self.posterior(x) # shape (q,num_samples,m,1)
            mu = jnp.mean(samples, axis=1, keepdims=True)    # shape (q,1,m,1)
            uncertainty = self.norm(samples-mu) # shape (q, num_samples, m)
            reparam = jnp.sqrt(0.5*jnp.pi) * uncertainty # shape (q, num_samples, m)
            US = jnp.mean(jnp.max(reparam, axis=0)) # scalar
            return -US
        elif self.acq_fn == 'GEE': # stands for Gaussian Empirical Entropy, a novel experimental acquisition function
            samples = self.posterior(x) # shape (q,num_samples,m)
            weights = self.weights(x).reshape(q,1)
            Ns = samples.shape[1] # ensamble size
            dists = self.norm(samples.reshape((q,1,Ns,-1)) - samples.reshape((q,Ns,1,-1)))  # shape (q,Ns,N_s)?
            #entropy = -logsumexp(dists/(-4*self.sig), axis=1) # shape (q,N_s) # old version
            entropy = -logsumexp(-(dists/(2 * self.sig))**2, axis=1, b=jnp.ones_like(dists)/Ns) # shape (q,N_s)
            reparam = weights * entropy
            US = jnp.mean(jnp.max(reparam, axis=0)) # Make sure taking max here makes sense for BALD
            return -US
        elif self.acq_fn == 'EM': # naive Expectation Maximization
            samples = self.posterior(x) # shape (q,num_samples,m,1)
            return -samples.mean()
        
        # That's all for now..
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def acq_value_and_grad(self, inputs):
        primals, f_vjp = vjp(self.evaluate, inputs)
        grads = f_vjp(jnp.ones_like(primals))[0]
        return primals, grads

    def next_best_point(self, q = 1, num_restarts = 10, maxfun=15000, required_init_pts=None, seed=None):
        if seed is not None:
            onp.random.seed(seed)
        lb, ub = self.bounds   
        dim = lb.shape[0]
        if self.acq_fn == 'random': #inactive policies
            #x_new = onp.random.uniform(low=lb, high=ub, size=(q,dim)) # random uniform
            x_new = onp.random.normal(size=(q,dim)) # random normal
            x_new = jnp.array(x_new.flatten())
        elif self.acq_fn == 'random_unif': #inactive policies
            x_new = onp.random.uniform(low=lb, high=ub, size=(q,dim)) # random uniform
            x_new = jnp.array(x_new.flatten())
        else:
            # Define objective that returns float64 NumPy arrays
            def objective(x):
                x = x.reshape(q, dim)
                value, grads = self.acq_value_and_grad(x)
                out = (onp.array(value, dtype=onp.float64), 
                    onp.array(grads.flatten(), dtype=onp.float64))
                return out
            # Optimize with random restarts
            loc, acq = [], []
            if required_init_pts is None:
                #key = random.PRNGKey(seed)
                #x0 = random.uniform(key, shape=(num_restarts, q, dim), minval=lb, maxval=ub)
                init = lb + (ub-lb)*lhs(dim, q*num_restarts)
                x0 = init.reshape(num_restarts, q, dim)
            else:
                num_new = num_restarts-len(required_init_pts)
                init = lb + (ub-lb)*lhs(dim, q*num_new)
                x0 = init.reshape(num_new, q, dim)
                x0 = jnp.concatenate([required_init_pts, x0], axis=0)
            dom_bounds = tuple(map(tuple, onp.tile(onp.vstack((lb, ub)).T,(q,1))))
            for i in trange(num_restarts):
                pos, val = minimize_lbfgs(objective, x0[i,...].flatten(), bnds = dom_bounds, maxfun=maxfun)
                loc.append(pos)
                acq.append(val)
            
            # remove potential NaNs
            loc = jnp.vstack(loc)
            acq = jnp.vstack(acq)
            valid_idx = jnp.logical_not(jnp.isnan(acq.flatten()))
            acq, loc = acq[valid_idx], loc[valid_idx]

            # select best point
            idx_best = jnp.argmin(acq)
            x_new = loc[idx_best,:]

        return x_new