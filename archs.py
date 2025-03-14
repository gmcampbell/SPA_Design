import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad

from flax import linen as nn

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

identity = lambda x : x

###################################
########## Architectures ##########
###################################

# All implementes using Flax library
# architectures ares instances of nn.Module class


class MLP(nn.Module):
    layers: Sequence[int]
    activation: Callable=nn.gelu
    output_activation : Callable=lambda x : x

    @nn.compact
    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.Dense(l)(x)
            x = self.activation(x)
        x = nn.Dense(self.layers[-1], use_bias=True)(x)
        x = self.output_activation(x) # potentially different activation on last layer
        return x
    
class Ensemble(nn.Module):
    arch : nn.Module
    ensemble_size : int
        
    @nn.compact
    def __call__(self, x):
        ensemble = nn.vmap(lambda mdl, x : mdl(x),
                           in_axes=0, out_axes=0,
                           variable_axes={'params': 0},
                           split_rngs={'params': True})
        x_tilde = jnp.broadcast_to(x, (self.ensemble_size, *x.shape)) # (ensamble_size, batch_size, input_dim)
        outputs = ensemble(self.arch, x_tilde) # (ensamble_size, batch_size, output_dim)
        return outputs

class OperatorEnsemble(nn.Module):
    arch : nn.Module
    ensemble_size : int
        
    @nn.compact
    def __call__(self, u, y):
        ensemble = nn.vmap(lambda mdl, u, y : mdl(u, y),
                           in_axes=0, out_axes=0,
                           variable_axes={'params': 0},
                           split_rngs={'params': True})
        u_tilde = jnp.broadcast_to(u, (self.ensemble_size, *u.shape)) # (ensamble_size, batch_size, input_dim)
        y_tilde = jnp.broadcast_to(y, (self.ensemble_size, *y.shape)) # (ensamble_size, batch_size, input_dim)
        outputs = ensemble(self.arch, u_tilde, y_tilde) # (ensamble_size, batch_size, output_dim)
        return outputs
    
class RPNEnsemble(nn.Module):
    arch : nn.Module
    ensemble_size : int
    beta : float=1.
        
    @nn.compact
    def __call__(self, u, y):
        # compute all outputs using a single forward pass
        outputs = OperatorEnsemble(self.arch, 2*self.ensemble_size)(u,y) # shape (2*ensemble_size, ...)

        # split outputs into prior and trainable components
        # the stop_gradient operation makes it so that prior parameters are not changed during training
        train_outputs = outputs[:self.ensemble_size] # shape (ensemble_size, ...)
        prior_outputs = jax.lax.stop_gradient(outputs[self.ensemble_size:]) # shape (ensemble_size, ...)

        # combines outputs and returns prediction
        return train_outputs + self.beta*prior_outputs

class RingEncoding(nn.Module):
    embed_dim : int # should be at least 3
    pre_processor : nn.Module
    activation : Callable=nn.tanh
    na_embeding_init : Callable=nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, r, w):
        na_embeding = self.param('na_embeding',
                                 self.na_embeding_init, # Initialization function
                                 (self.embed_dim,))
        ''' # the first two commands of this function are logically equivalent to this commented block
        if (r is None) or (w is None):
            v = na_embeding
        else:
            v = jnp.hstack((r[:,None], w[:,None]))
            v = nn.Dense(self.embed_dim, use_bias=False)(v)
        '''
        # checks if either radius of width is nan
        no_ring = jnp.logical_or(jnp.isnan(r), jnp.isnan(w))[:,None]
        v = jnp.hstack((r[:,None], w[:,None]))

        # this step is required to avoid nan in gradient computations
        v = jnp.where(no_ring,
                      jnp.zeros_like(v),
                      v)
        # maps v to embeded dimension
        v = jnp.where(no_ring,
                      na_embeding[None,:], # if nan, set to fixed vector
                      nn.Dense(self.embed_dim, use_bias=False)(v)) # if not, project into embed_dim
        
        v = self.activation(v)
        v = self.pre_processor(v)
        return v
        

    
class Actuator(nn.Module):
    ring_encoder : nn.Module
    latent_encoder : nn.Module
    polynomial_degree : int=2 # change this to increase/decrease the degree of polynomial approximation in p
    output_activation : Callable=lambda x : x
    max_num_rings : int=2


    @nn.compact
    def __call__(self, u, y):
        h, t, w0, r1, w1, r2, w2 = u.T # unpack parameters

        # encode rings into a single vector
        v1 = self.ring_encoder(r1, w1)
        v2 = self.ring_encoder(r2, w2)
        v = v1+v2

        # compute one-hot-encoding of number of rings
        no_ring_1 = jnp.logical_or(jnp.isnan(r1), jnp.isnan(w1))
        no_ring_2 = jnp.logical_or(jnp.isnan(r1), jnp.isnan(w1))
        num_rings = self.max_num_rings-no_ring_1.astype(int)-no_ring_2.astype(int) # integer representation of number of rings
        num_rings = nn.one_hot(num_rings, num_classes=self.max_num_rings+1) # convert into one-hot encoding

        # compute polynomial coefficients
        u_new = jnp.hstack((v, h[:,None], t[:,None], w0[:,None]))
        u_new = jnp.hstack((u_new, (u_new.shape[-1]/2)*num_rings))
        latent = self.latent_encoder(u_new)
        coefs = nn.Dense(self.polynomial_degree+1)(latent) # (batch_size, polynomial_degree+1)

        basis = (y**jnp.arange(self.polynomial_degree+1)) # (batch_size, polynomial_degree+1)
        fs = jnp.sum(coefs * basis, axis=-1, keepdims=True) # (batch_size, 1)
        return self.output_activation(fs) # (batch_size, 1)


    
class MonotonicActuator(nn.Module):
    ring_encoder : nn.Module
    latent_encoder : nn.Module
    polynomial_degree : int=2 # change this to increase/decrease the degree of polynomial approximation in p
    max_num_rings : int=2
    output_activation : Callable=nn.relu # set to nn.relu to enforce exact force > 0


    @nn.compact
    def __call__(self, u, y):
        h, t, w0, r1, w1, r2, w2 = u.T # unpack parameters

        # encode rings into a single vector
        v1 = self.ring_encoder(r1, w1)
        v2 = self.ring_encoder(r2, w2)
        v = v1+v2

        # compute one-hot-encoding of number of rings
        no_ring_1 = jnp.logical_or(jnp.isnan(r1), jnp.isnan(w1))
        no_ring_2 = jnp.logical_or(jnp.isnan(r1), jnp.isnan(w1))
        num_rings = self.max_num_rings-no_ring_1.astype(int)-no_ring_2.astype(int) # integer representation of number of rings
        num_rings = nn.one_hot(num_rings, num_classes=self.max_num_rings+1) # convert into one-hot encoding

        # compute coefficients
        u_new = jnp.hstack((v, h[:,None], t[:,None], w0[:,None]))
        u_new = jnp.hstack((u_new, (u_new.shape[-1]/2)*num_rings))
        latent = self.latent_encoder(u_new)
        coefs = nn.Dense(self.polynomial_degree+1)(latent) # (batch_size, polynomial_degree+2)
        # extract coefficients for polynomial (all positive to enforce monotonicity, except for constant term)
        constant_term = coefs[...,:1] # (batch_size, 1)
        poly_coefs = nn.leaky_relu(coefs[...,1:]) # (batch_size, polynomial_degree)


        basis = ((y/1_000)**(1+jnp.arange(self.polynomial_degree))) # (batch_size, polynomial_degree)
        fs = jnp.sum(poly_coefs * basis, axis=-1, keepdims=True) # (batch_size, 1)
        fs = fs + constant_term # (batch_size, 1)
        return self.output_activation(fs) # (batch_size, 1)
    

class Actuator_NonPolynomial(nn.Module):
    ring_encoder : nn.Module
    latent_encoder : nn.Module
    output_activation : Callable=lambda x : x
    max_num_rings : int=2


    @nn.compact
    def __call__(self, u, y):
        h, t, w0, r1, w1, r2, w2 = u.T # unpack parameters

        # encode rings into a single vector
        v1 = self.ring_encoder(r1, w1)
        v2 = self.ring_encoder(r2, w2)
        v = v1+v2

        # compute one-hot-encoding of number of rings
        no_ring_1 = jnp.logical_or(jnp.isnan(r1), jnp.isnan(w1))
        no_ring_2 = jnp.logical_or(jnp.isnan(r1), jnp.isnan(w1))
        num_rings = self.max_num_rings-no_ring_1.astype(int)-no_ring_2.astype(int) # integer representation of number of rings
        num_rings = nn.one_hot(num_rings, num_classes=self.max_num_rings+1) # convert into one-hot encoding

        # compute forces
        u_new = jnp.hstack((v, h[:,None], t[:,None], w0[:,None], y))
        u_new = jnp.hstack((u_new, (u_new.shape[-1]/2)*num_rings))
        latent = self.latent_encoder(u_new)
        fs = nn.Dense(1)(latent) # (batch_size, 1)

        return self.output_activation(fs) # (batch_size, 1)

