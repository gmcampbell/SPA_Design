import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad


import optax
import torch.utils.data as data

from functools import partial
import itertools
from tqdm import trange
import matplotlib.pyplot as plt

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

##############################################
########## Model Class for Training ##########
##############################################

class Operator:
    def __init__(self, arch, batch, optimizer=None, key=random.PRNGKey(43),
                 normalize_data=True, has_weights=False,
                 huber_delta=1.) -> None:
        # Define model
        self.arch = arch
        self.key = key
        self.has_weights = has_weights
        self.huber_delta = huber_delta

        # Initialize parameters
        if self.has_weights:
            inputs, output, weights = batch
        else:
            inputs, output = batch
        us, ys = inputs
        self.params = self.arch.init(self.key, us, ys)

        # Tabulate function for checking network architecture
        self.tabulate = lambda : self.arch.tabulate(self.key, us, ys, console_kwargs={'width':110})
        
        # Vectorized functions
        #self.apply = vmap(self.arch.apply, in_axes=(None,0,0))
        self.normalize_data = normalize_data
        if normalize_data:
            mu_u, sig_u = us[:,:3].mean(0), us[:,:3].std(0) # stats for first three entries
            rings = jnp.concatenate(jnp.split(us[:,3:], 2, axis=-1)) # extract ring coefs
            rings = rings[jnp.logical_not(jnp.isnan(rings).any(1))] # remove nan rings
            mu_rings = rings.mean(0) # mean of ring coefs
            mu_rings = jnp.where(jnp.isnan(mu_rings), 0., mu_rings) # potentially replace nan with 0
            mu_u = jnp.concatenate([mu_u, mu_rings, mu_rings])
            sig_rings = rings.std(0) # std of ring coefs
            sig_rings = jnp.where(jnp.isnan(sig_rings), 1., sig_rings)  # potentially replace nan with 1
            sig_u = jnp.concatenate([sig_u, sig_rings, sig_rings])
            self.norm_stats = (mu_u, sig_u)
            self.apply = lambda params, u, y : self.arch.apply(params, (u-mu_u)/sig_u, y)
        else:
            self.apply = self.arch.apply
        # jit apply function for faster calls
        self.apply = jit(self.apply)

        # Optimizer
        if optimizer is None:
            lr = optax.exponential_decay(5e-3, transition_steps=5_000, decay_rate=0.1, end_value=1e-4)
            self.optimizer = optax.adam(learning_rate=lr)
        else:
            self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self.params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.grad_norm_log = []

    def l2_loss(self, params, u, y, s, w):
        outputs = self.apply(params, u, y)
        if self.huber_delta is None:
            # use L2 loss
            error = (outputs-s[None,:])
            error = error**2
        else:
            # use huber loss with specified delta
            error = optax.huber_loss(outputs, s[None,:], delta=self.huber_delta)
        if w is not None:
            # multiply by training weights
            error = error * w
        return error.mean()
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        if self.has_weights:
            inputs, targets, weights = batch
        else:
            inputs, targets = batch
            weights=None
        u, y = inputs
        s = targets
        return self.l2_loss(params, u, y, s, weights).mean() # scalar
    

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, params, opt_state, batch):
        grads = grad(self.loss)(params, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, grads

    # Optimize parameters in a loop
    def train(self, dataset, nIter = 10_000):
        data = iter(dataset)
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            batch = next(data)
            self.params, self.opt_state, grads = self.step(self.params, self.opt_state, batch)
            # Logger
            if it % 100 == 0:
                l = self.loss(self.params, batch)
                grad_norm = jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grads))
                self.loss_log.append(l)
                self.grad_norm_log.append(grad_norm)
                pbar.set_postfix({'loss': l, 'grad_norm': jnp.mean(jnp.array(grad_norm))})

    def plot_training_log(self):
        plt.figure(figsize=(16, 4))

        # Ploting loss
        plt.subplot(121)
        plt.plot(100*jnp.arange(len(self.loss_log)), self.loss_log)
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Through Training')

        # Plotting gradient norms
        plt.subplot(122)
        plt.plot(100*jnp.arange(len(self.loss_log)), [jnp.mean(jnp.array(g)) for g in self.grad_norm_log])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm Through Training')

        plt.show()

    def plot_predictions(self, batch, return_predictions=False, use_uq=False, return_RMSE = False):
        # Create reconstructions
        if self.has_weights:
            (u, y), s_true, weights = batch
        else:
            (u, y), s_true = batch
        s_pred_uq = self.apply(self.params, u, y)
        s_pred = s_pred_uq.mean(0)

        error_fn = lambda target, output: jnp.linalg.norm(target-output,2)/jnp.linalg.norm(target,2)
        error = vmap(error_fn, in_axes=(0,0))(s_true, s_pred)
        RMSE = jnp.sqrt(jnp.mean(error**2))
        #print('Relative L2 error: {:.2e}'.format(jnp.mean(error)))

        plt.figure(figsize=(16, 4))

        # Ploting examples of reconstructions
        plt.subplot(131)
        plt.scatter(y, s_pred, s= 0.005)
        plt.xlabel('$y$')
        plt.ylabel('$s$')
        plt.title('Example Reconstructions')

        plt.subplot(132)
        plt.scatter(y, s_true, s= 0.005)
        plt.xlabel('$y$')
        plt.ylabel('$s$')
        plt.title('True Values')

        # plotting histogram of errors
        plt.subplot(133)
        error = s_pred-s_true
        plt.hist(error.flatten(), bins=50)
        plt.title(f'Histogram of errors (RMSE is {jnp.sqrt((error**2).mean()):.2f})\n(median absolute error is {jnp.median(abs(error)):.2f})')
        plt.show()

        if return_predictions:
            if use_uq:
                return s_pred_uq
            else:
                return s_pred

        if return_RMSE:
            return RMSE




#################################
########## Data Loader ##########
#################################

# Dataset loader
class BatchedDataset(data.Dataset):

  def __init__(self, raw_data, key, batch_size=None, has_weights=False):
    super().__init__()
    self.inputs = raw_data[0]
    self.targets = raw_data[1]
    self.has_weights = has_weights
    if self.has_weights:
        self.weights = raw_data[2] # optional training weights
    else:
        self.weights = None
    self.size = len(self.inputs[0])
    self.key = key
    if batch_size is None: # Will use full batch
      self.batch_size = self.size
    else:
      self.batch_size = batch_size
    
  def __len__(self):
    return self.size
  
  def __getitem__(self, idx):
    self.key, subkey = random.split(self.key)
    return self.__select_batch(subkey)

  @partial(jit, static_argnums=(0,))
  def __select_batch(self, key):
    idx = random.choice(key, self.size, (self.batch_size,), replace=False)
    batch_inputs = (self.inputs[0][idx], self.inputs[1][idx])
    batch_targets = self.targets[idx]
    if self.weights is None: # return only inputs and targets
        return batch_inputs, batch_targets
    else: # return inputs, targets and training weights
        batch_weights = self.weights[idx]
        return batch_inputs, batch_targets, batch_weights