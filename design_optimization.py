import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad


import optax
import torch.utils.data as data
from flax import linen as nn


################################
###### Default Parameters ######
################################


# Parameters for material
_radius = 70. # radius of the entire membrane
_min_spacing = 3. # minimum spacing between rings and other rings/boundaries
_min_width = 5. # minimum width of each ring

_tmin = 2. # minimum membrane thickness
_tmax = 3. # maximum membrane thickness
_contactmin = 25.4 # minimum contact disk radius
_contactmax = 38.1  # maximum contact disk radius

# parameters for pressure and height
_hmin = 0 # minimum height
_hmax = 70 # maximum height (in mm I think)
_pmin = 0 # minimum pressure
_pmax = 10_000 # maximum pressure (in PSI I think?) # TODO: check units
_fmin = 0 # minimum force (not really used as of now)
_fmax = 150 # maximum force - used for normalizations


##############################
###### Useful functions ######
##############################

# computes radii and widths of rings
# effectively, this function transforms numbers between 0 and 1 into the proper ring parameters
def get_rw(w0, ring_params, num_rings, radius=_radius, min_spacing=_min_spacing, min_width=_min_width, max_rings=2):
    ring_params = ring_params.sort() # sorts entries from smallest to largest
    available_space = radius - w0 - min_spacing*(num_rings+1) - 2*min_width*num_rings
    #assert (available_space>0), "There is no space available for rings!"
    radii = [w0 + (i+1)*min_spacing + (2*i+1)*min_width +  available_space*(ring_params[2*i+1]+ring_params[2*i])/2 for i in range(num_rings)]
    widths = [min_width + available_space*(ring_params[2*i+1]-ring_params[2*i])/2 for i in range(num_rings)]
    radii = jnp.array(radii + (max_rings-num_rings)*[jnp.nan])
    widths = jnp.array(widths + (max_rings-num_rings)*[jnp.nan])
    return jnp.ravel(jnp.vstack([radii, widths]), order='F')

# 'optimization variables to physical variables'
def opt_var_to_phys_var(membrane_coefs, num_rings, max_rings=2):
    material_params = membrane_coefs[:2]
    ring_params = jnp.concatenate([membrane_coefs[2:], jnp.array(2*(max_rings-num_rings)*[jnp.nan])])
    ring_params = get_rw(material_params[1], ring_params, num_rings)

    membrane = jnp.concatenate([material_params, ring_params])
    return membrane

def print_recommendations(u_new):
    q = u_new.shape[0]
    num_rings = 2 - jnp.isnan(u_new).sum() // (2*q)
    for j in range(q):
        print(f'\nMembrane:')
        print(f'thickness: {u_new[j][0] : .2f}')
        print(f'contact radius: {u_new[j][1] : .2f}')
        for i in range(num_rings):
            print(f'Ring {i+1}) radius: {u_new[j][i*2 + 2] : .2f}; width: {u_new[j][i*2 + 3] : .2f}')






##################################################
###### Posterior Functions For Optimization ######
##################################################

# obtain membrane closest to target trajectory
def get_trajectory_posterior_fn(
        model, num_rings, target_Fs, target_Ps, target_Hs,
        k_force, k_pressure, k_height,
        hmin = _hmin, hmax=_hmax, pmin=_pmin, pmax=_pmax, f_max=_fmax, max_rings=2,
        tmin=_tmin, tmax=_tmax, contactmin=_contactmin, contactmax=_contactmax):
    # make sure all targets have the same length
    assert (len(target_Fs) == len(target_Ps) == len(target_Hs)), "the following vectors should all have the same lenght: target_Fs, target_Ps, target_Hs"
    num_targets = len(target_Fs)

    # computing optimization bounds
    lb = jnp.array([tmin, contactmin] + 2*num_rings*[0.] + num_targets*[hmin] + num_targets*[pmin])
    ub = jnp.array([tmax, contactmax] + 2*num_rings*[1.] + num_targets*[hmax] + num_targets*[pmax])

    # function to compute posterior function
    def posterior(coefs, return_individual_vals=False, return_predictions_only=False):
        # extract relevant values
        membrane_coefs = coefs[:2+2*num_rings] # vector with up to 6 entries: t, w0, r1, w1, r2, w2
        hs = coefs[-2*num_targets:-num_targets] # heights; shape (num_targets,)
        ps = coefs[-num_targets:] # pressures; shape (num_targets,)

        # convert optimization varaible into physical coefficients of membrane
        membrane = opt_var_to_phys_var(membrane_coefs, num_rings, max_rings) # shape (6,)

        #preparing us
        us = [jnp.concatenate([jnp.array([h]), membrane])[None,:] for h in hs] # list of num_hs vectors of shape (1, 7) each
        us = jnp.concatenate(us) # shape (num_targets, 7)

        #preparing ys
        ys = jnp.array(ps)[:,None] # shape (num_targets,1)

        # making predictions
        # us should be shape (num_targets, 7); ys should be shape (num_targets, 1)
        samples = model.apply(model.params, us, ys) # shape (num_ensembles, num_targets, 1) # predicted forces
        fs = samples.mean((0,-1)) # shape (num_targets,)

        # compute normalized errors squared
        force_error = ((fs - target_Fs)/f_max)**2 # shape (num_targets,)
        pressure_error = ((ps - target_Ps)/pmax)**2 # shape (num_targets,)
        height_error = ((hs - target_Hs)/hmax)**2 # shape (num_targets,)
        
        # summarize into single values
        force_error = force_error.mean() # scalar
        pressure_error = pressure_error.mean() # scalar
        height_error = height_error.mean() # scalar


        if return_predictions_only:
            # ignore other stuff and return only predictions
            # used mostly for debugging/diagnostics
            return fs, ps, hs
        else:
            if return_individual_vals:
                return force_error, pressure_error, height_error
            else:
                # combine errors using desired weighting
                total_score = -k_force*force_error - k_pressure*pressure_error - k_height*height_error # scalar
                return total_score # scalar
    # returns posterior function and bounds for optimization
    return posterior, (lb, ub)



# obtain membrane with largest hight for given force & pressure
def get_height_max_posterior_fn(model, num_rings, target_Fs, target_Ps,
                     k_force, k_pressure, k_height,
                     hmin = _hmin, hmax=_hmax, pmin=_pmin, pmax=_pmax, f_max=60, max_rings=2,
                     tmin=_tmin, tmax=_tmax, contactmin=_contactmin, contactmax=_contactmax):
    # make sure all targets have the same length
    assert len(target_Fs) == len(target_Ps), "the following vectors should all have the same lenght: target_Fs, target_Ps"
    num_targets = len(target_Fs)

    # computing optimization bounds
    lb = jnp.array([tmin, contactmin] + 2*num_rings*[0.] + num_targets*[hmin] + num_targets*[pmin])
    ub = jnp.array([tmax, contactmax] + 2*num_rings*[1.] + num_targets*[hmax] + num_targets*[pmax])

    # function to compute posterior function
    def posterior(coefs, return_individual_vals=False, return_predictions_only=False):
        # extract relevant values
        membrane_coefs = coefs[:2+2*num_rings] # vector with up to 6 entries: t, w0, r1, w1, r2, w2
        hs = coefs[-2*num_targets:-num_targets] # heights; shape (num_targets,)
        ps = coefs[-num_targets:] # pressures; shape (num_targets,)

        # convert optimization varaible into physical coefficients of membrane
        membrane = opt_var_to_phys_var(membrane_coefs, num_rings, max_rings) # shape (6,)

        #preparing us
        us = [jnp.concatenate([jnp.array([h]), membrane])[None,:] for h in hs] # list of num_hs vectors of shape (1, 7) each
        us = jnp.concatenate(us) # shape (num_targets, 7)

        #preparing ys
        ys = jnp.array(ps)[:,None] # shape (num_targets,1)

        # making predictions
        # us should be shape (num_targets, 7); ys should be shape (num_targets, 1)
        samples = model.apply(model.params, us, ys) # shape (num_ensembles, num_targets, 1) # predicted forces
        fs = samples.mean((0,-1)) # shape (num_targets,)

        # compute normalized errors squared
        force_error = ((fs - target_Fs)/f_max)**2 # shape (num_targets,)
        pressure_error = ((ps - target_Ps)/pmax)**2 # shape (num_targets,)
        height_factor = (hs/hmax) # shape (num_targets,)

        # summarize into single values
        force_error = force_error.mean() # scalar
        pressure_error = pressure_error.mean() # scalar
        # scale = jnp.log(len(height_factor)) # scalar for scaling logsumexp (a smooth way of computing minimum)
        scale = hmax/15 # scalar for scaling logsumexp (a smooth way of computing minimum) - GMC 2/15
        # height_factor = -nn.activation.logsumexp(-height_factor*scale)/scale # scalar
        # to use mean instead of worst height, comment two lines above, and uncomment the one below
        height_factor = height_factor.mean() # scalar


        if return_predictions_only:
            # ignore other stuff and return only predictions
            # used mostly for debugging/diagnostics
            return fs, ps, hs
        else:
            if return_individual_vals:
                return force_error, pressure_error, height_factor
            else:
                # combine errors using desired weighting
                # we want to minimize force and pressure errors, while maximizing heights
                total_score = -k_force*force_error - k_pressure*pressure_error + k_height*height_factor # scalar
                return total_score # scalar
    # returns posterior function and bounds for optimization
    return posterior, (lb, ub)