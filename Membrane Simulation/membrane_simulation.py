import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad, value_and_grad

import matplotlib.pyplot as plt
from tqdm import trange
from functools import partial
import copy

import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, Kvaerno5
import optimistix as optx

# material constants - elastic solved from uniaxial tensile testing.
_MU_elastic = 31_700 
_MU_stiff = 1e6
_JM_elastic = 39.6
_JM_stiff = 25

# Gent hypereastlic model - strain energy density 
def W_fn(lamb_r, lamb_theta, constants):
    # unpack constants
    mu, Jm, h = constants
    
    # compute argument of log
    log_arg = 1 - (lamb_r**2 + lamb_theta**2 + 1/(lamb_r**2 * lamb_theta**2) - 3)/Jm
    
    return -mu*Jm*jnp.log(log_arg)/2

# Strain energy density gradients
def W_grads(lamb_r, lamb_theta, constants):
    # funciton for computin gradient of W_fn
    def _grads(lamb_r, lamb_theta):
        W, (W_r, W_theta) = value_and_grad(W_fn, argnums=(0,1))(lamb_r, lamb_theta, constants=constants)
        return W_r, (W, W_theta)
    
    # now get other derivatives
    (W_r, (W, W_theta)), (W_rr, W_r_theta) =  value_and_grad(_grads, argnums=(0,1), has_aux=True)(lamb_r, lamb_theta)
    return W, W_r, W_theta, W_rr, W_r_theta

# this determines the evolution of the ODE system
def vector_field(r, y, args):
    # unpack y
    lamb_r, lamb_theta, beta = y
    # unpack constants
    mat_constants, p = args
    mu, Jm, h, = mat_constants

    # compute information about W (and derivatives)
    W, W_r, W_theta, W_rr, W_r_theta = W_grads(lamb_r, lamb_theta, mat_constants)
    
    d_lamb_r = (W_theta-lamb_r*W_r_theta)*jnp.sin(beta)/(r*W_rr) - (W_r - lamb_theta*W_r_theta)/(r*W_rr)
    d_lamb_theta = (lamb_r*jnp.sin(beta)-lamb_theta)/r
    d_beta = W_theta*jnp.cos(beta)/(r*W_r) - p * lamb_r*lamb_theta/(h*W_r)

    d_y = d_lamb_r, d_lamb_theta, d_beta
    return jnp.array(d_y)



# setup for evolving DOE using explicit method
_ode_term = ODETerm(vector_field)
_ode_solver = Tsit5()
_ode_dr0 = 0.000075


def solve_ode(y0, p, r_init, r_end, mat_constants):
    # solve ode
    args = mat_constants, p
    saveat = SaveAt(ts=jnp.linspace(r_init, r_end, 1000))
    sol = diffeqsolve(_ode_term, _ode_solver, r_init, r_end, _ode_dr0, y0, args=args, saveat=saveat)
    return sol

# setup for finind next initial conditions
# must solve for x to find root of this
def error_for_boundary(x, args):
    W_r_goal, lamb_theta, mat_constants = args
    W, W_r, W_theta, W_rr, W_r_theta = W_grads(x, lamb_theta, mat_constants) # only care about W_r
    return (W_r_goal - W_r)

_Dogleg_solver = optx.BestSoFarRootFinder(optx.Dogleg(rtol=1e-2, atol=1e2))
_bisect_solver = optx.BestSoFarRootFinder(optx.Bisection(rtol=1e-2, atol=1e2))

def find_next_y0(yf, mat_constants, next_mat_constants, solver=_Dogleg_solver):
    W, W_r, W_theta, W_rr, W_r_theta = W_grads(yf[0], yf[1], mat_constants) # only care about W_r
    args_root = W_r, yf[1], next_mat_constants
    first_guess = 1.
    root_sol = optx.root_find(error_for_boundary, solver, first_guess, args=args_root,
                            options=dict(lower=.95, upper=1.2),
                            max_steps=512, throw=False).value
    # package initial conditions for next piece
    y0 = jnp.array([root_sol, yf[1], yf[2]])
    return y0

def simulate_ringed_membrane(x, F, p, h, change_material, material_type):
    '''
    Args:
        x (float): initial value for lamb_r
        F (float): force being applied to the membrane
        p (float): pressure being applied to the membrane
        h (float): thickness of membrane
        change_material (tuple of floats): points where membrane changes material
        material_type (tuple of string): sequence of material types
    '''
    # make sure we have a material type for each piece
    assert len(change_material) == (1+len(material_type)), 'change_material and material_type have incompatible lengths'
    # set material constants for first piece
    if material_type[0] == 'elast':
        mat_constants = (_MU_elastic, _JM_elastic, h)
    elif material_type[0] == 'stiff':
        mat_constants = (_MU_stiff, _JM_stiff, h)
    else:
        raise NotImplementedError(f'Could not recognize material {material_type[i]}')
    # compute initial conditions
    R0 = change_material[0] # get contact area
    W, W_r, W_theta, W_rr, W_r_theta = W_grads(x, 1., mat_constants)
    beta_0 = jnp.arccos( (-F+jnp.pi * p * R0**2) / (2*jnp.pi*h*R0*W_r) )
    y0 = jnp.array([x, 1., beta_0])

    full_sols = []
    for i in range(len(material_type)):
        # select material constants
        if material_type[i] == 'elast':
            mat_constants = (_MU_elastic, _JM_elastic, h)
        elif material_type[i] == 'stiff':
            mat_constants = (_MU_stiff, _JM_stiff, h)
        else:
            raise NotImplementedError(f'Could not recognize material {material_type[i]}')
        # set piece start and end points
        r_init = change_material[i] # where piece starts
        r_end = change_material[i+1] # where piece ends

        # solve ode
        sol = solve_ode(y0, p, r_init, r_end, mat_constants)
        full_sols.append(copy.deepcopy(sol))

        if i != (len(material_type)-1): # if not on final piece, compute next y0
            # select material constants for next piece
            if material_type[i+1] == 'elast':
                next_mat_constants = (_MU_elastic, _JM_elastic, h)
                root_solver = _Dogleg_solver
            elif material_type[i+1] == 'stiff':
                next_mat_constants = (_MU_stiff, _JM_stiff, h)
                root_solver = _Dogleg_solver
            else:
                raise NotImplementedError(f'Could not recognize material {material_type[i+1]}')
            # compute next initial conditions
            yf = sol.ys[-1,:] # final from last piece
            y0 = find_next_y0(yf, mat_constants, next_mat_constants, solver=root_solver)
    return full_sols


def shoot_x(x, F, p, h, change_material, material_type):
    full_sols = simulate_ringed_membrane(x, F, p, h, change_material, material_type)
    return full_sols[-1].ys[-1,1] - 1


# function to compute height of a given membrane
def simulate_height(x, F, p, h, change_material, material_type):
    full_sols = simulate_ringed_membrane(x, F, p, h, change_material, material_type)
    
    start = 0
    i=0
    for sol in full_sols:
        r_init = change_material[i]
        r_end = change_material[i+1]
        z = (r_end-r_init)*jnp.cumsum(sol.ys[:,0]*jnp.cos(sol.ys[:,2]))/1000
        i = i+1
        start = start + z[-1]
    return start


def obtain_x(h, change_material, material_type,
            F_min=0, F_max=20, num_Fs=100,
            p_min=50, p_max=8_000, num_ps=100,
            init_guess=5, dx_init=0.01, atol=1e-4, max_iter=500,
            verbose=True):
    '''
    Args:
        h (float): thickness of membrane
        change_material (tuple of floats): points where membrane changes material
        material_type (tuple of string): sequence of material types
        F_min (float): minimum force being applied to the membrane
        F_max (float): maximum force being applied to the membrane
        num_Fs (int): resolution of force grid
        p_min (float): minimum pressure being applied to the membrane
        p_max (float): maximum pressure being applied to the membrane
        num_ps (int): resolution of pressure grid
        init_guess (float): initial guess for shooting x
        dx_init (float): how much to decrease value of x until reaching negative value
        atol (float): absolute tolerance for considering a value of x acceptable
        max_iter (int): maximum number of iterations for shooting x
        verbose (bool): wether to print some informative messages
    '''
    # setup force/pressure combinations
    Fs = jnp.linspace(F_min, F_max, num_Fs) # shape (num_fs,)
    ps = jnp.linspace(p_min, p_max, num_ps) # shape (num_ps,)
    FF, PP = jnp.meshgrid(Fs, ps) # FF and PP are both shape (num_fs, num_ps)
    FF, PP = FF.flatten(), PP.flatten() # FF and PP are now both shape (num_fs*num_ps,)

    # initialize values
    dx = jnp.ones_like(FF)*dx_init
    current_x = jnp.ones_like(dx) * init_guess

    # compute values
    best_x = jnp.ones_like(dx) * init_guess # best x value so far
    best_val =vmap(shoot_x, in_axes=(0, 0, 0, None, None, None))(best_x, FF, PP, h, change_material, material_type) # best value so far
    lowest_positive_guess = jnp.ones_like(dx) * jnp.nan # guess with lowest positive value so far
    lowest_positive_val = jnp.ones_like(dx) * jnp.nan
    #lowest_positive_val = jnp.where(lowest_positive_val>0, lowest_positive_val, jnp.nan) # set to nan if value is actually negative
    highest_negative_guess = jnp.ones_like(dx) * jnp.nan # guess with highest positive value so far
    highest_negative_val = jnp.ones_like(dx) * jnp.nan

    for i in trange(max_iter):
        # determine next guesses
        no_negative_guess = jnp.isnan(highest_negative_guess)
        no_positive_guess = jnp.isnan(lowest_positive_val)
        missing_guess  = jnp.logical_or(no_negative_guess, no_positive_guess)
        current_x = jnp.where(missing_guess,
                        current_x - dx, # if missing guess, decrease guess by dx
                        (lowest_positive_guess + highest_negative_guess)/2) # both present, use bisection
        
        # compute value for current x
        current_val = vmap(shoot_x, in_axes=(0, 0, 0, None, None, None))(current_x, FF, PP, h, change_material, material_type)



        # checks if lowest positive value/guess needs to be updated
        # this will happen either when a first positive value is found OR when the new value is positive and smaller than current lpv
        current_val_is_positive = current_val > 0 # true when values are positive
        # true when val is positive AND there is no positive val stored
        first_positive_val = jnp.logical_and(current_val_is_positive,
                                             no_positive_guess)
        # true when val is positive AND lower than existing lpv
        current_val_is_lpv = jnp.logical_and(current_val_is_positive, 
                                             current_val < lowest_positive_val)
        # true if we found the first positive value, or new lpv
        update_lpv = jnp.logical_or(first_positive_val, current_val_is_lpv)
        # update values when needed
        lowest_positive_guess = jnp.where(update_lpv,
                                          current_x,
                                          lowest_positive_guess)
        lowest_positive_val = jnp.where(update_lpv,
                                        current_val,
                                        lowest_positive_val)

        # checks if highest negative value/guess needs to be updated
        current_val_is_negative =  current_val < 0
        # checks if there is a positive guess, but not negative
        pos_but_no_neg = jnp.logical_and(jnp.logical_not(no_positive_guess),
                                         no_negative_guess)
        # true when val is negative AND there is a positive but no negative val stored
        first_good_negative_val = jnp.logical_and(current_val_is_negative,
                                                  pos_but_no_neg)
        # true if current val is negative AND higher than existing hnv
        current_val_is_hnv = jnp.logical_and(current_val_is_negative,
                                             current_val > highest_negative_val)
         # true if we found the first good negative value, or new hnv
        update_hnv = jnp.logical_or(first_good_negative_val, current_val_is_hnv)
        # update values when needed
        highest_negative_guess = jnp.where(update_hnv,
                                           current_x,
                                           highest_negative_guess)
        highest_negative_val = jnp.where(update_hnv,
                                         current_val,
                                         highest_negative_val)
        
        # check if dx should be decreased
        # this will happen only when new val is nan AND there is a positive but not negative val stored
        current_val_is_nan = jnp.isnan(current_val) # true when values are nan
        decrease_dx = jnp.logical_and(current_val_is_nan,
                                      pos_but_no_neg)
        # decreases dx by half if needed
        dx = jnp.where(decrease_dx, dx/2, dx)
               

        # update best val if needed
        new_bx = abs(current_val) < abs(best_val)
        best_x = jnp.where(new_bx, current_x, best_x)
        best_val = jnp.where(new_bx, current_val, best_val)


        if (abs(best_val) < atol).all() and verbose:
            print(f'Found all sufficiently good values after {i+1} iterations.')
            break
        if (i == max_iter-1) and verbose:
            print(f'Loop finished after {max_iter} iterations, but tolerance has not been reached yet for some points.')
        
    # loop for computing x has ended. Now filtering some values if needed
    good_vals = abs(best_val) < atol
    FF = FF[good_vals]
    PP = PP[good_vals]
    best_x = best_x[good_vals]
    if verbose:
        print(f'Overall, {good_vals.sum()} pressure/force pairs were successfull.')
        if good_vals.sum()<len(good_vals):
            print(f'Could not find roots for the remaining {num_Fs*num_ps - good_vals.sum()}.')
    return best_x, FF, PP


# Primary simulation function for Force-pressure-height plane generation
def obtain_qoi_plane(h, change_material, material_type,
            F_min=0, F_max=20, num_Fs=100,
            p_min=50, p_max=8_000, num_ps=100,
            init_guess=5, dx_init=0.01, atol=1e-4, max_iter=500,
            verbose=True, filter_negative=True, min_height=1e-3,
            return_best_x=False):
    '''
    Args:
        h (float): thickness of membrane
        change_material (tuple of floats): points where membrane changes material
        material_type (tuple of string): sequence of material types
        F_min (float): minimum force being applied to the membrane
        F_max (float): maximum force being applied to the membrane
        num_Fs (int): resolution of force grid
        p_min (float): minimum pressure being applied to the membrane
        p_max (float): maximum pressure being applied to the membrane
        num_ps (int): resolution of pressure grid
        init_guess (float): initial guess for shooting x
        dx_init (float): how much to decrease value of x until reaching negative value
        atol (float): absolute tolerance for considering a value of x acceptable
        max_iter (int): maximum number of iterations for shooting x
        verbose (bool): wether to print some informative messages
        filter_negative (bool): indicates whether or not to exclude force/pressure pairs that result in a negative height prediction
        min_height (float): lift height [m] below which we will filter (see filter_negative)
        reutrn_best_x (bool): tells the function to return the best values found for the starting strain x obtained by the shooting method
    '''
    
    best_x, FF, PP = obtain_x(h, change_material, material_type,
            F_min=F_min, F_max=F_max, num_Fs=num_Fs,
            p_min=p_min, p_max=p_max, num_ps=num_ps,
            init_guess=init_guess, dx_init=dx_init, atol=atol, max_iter=max_iter,
            verbose=verbose)
    
    if verbose:
        print('\nComputing heights for successfull pressure/force pairs...')
    heights = vmap(simulate_height, in_axes=(0, 0, 0, None, None, None))(best_x, FF, PP, h, change_material, material_type)

    if filter_negative:
        good_heights = heights > min_height # keep heights larger than threshold
        if verbose:
            print(f'Found {good_heights.sum()} heights larger than 1mm. Other {len(heights)-good_heights.sum()} were less than that.')
        FF = FF[good_heights]
        PP = PP[good_heights]
        heights = heights[good_heights]
        best_x = best_x[good_heights]

    if return_best_x:
        return FF, PP, heights, best_x
    else:
        return FF, PP, heights
    


# for computing membrane shapes
def compute_membrane_shape(x, F, p, h, change_material, material_type):
    full_sols = simulate_ringed_membrane(x, F, p, h, change_material, material_type)
    i=0
    start = 0

    dists = []
    zs = []
    for sol in full_sols:
        r_init = change_material[i]
        r_end = change_material[i+1]
        #z = start + jnp.cumsum(sol.ys[:,0]*jnp.sin(sol.ys[:,2]))*(r_end-r_init)/1000
        z = start + jnp.cumsum(sol.ys[:,0]*jnp.cos(sol.ys[:,2]))*(r_end-r_init)/1000
        dist = (sol.ys[:,1]*sol.ts)
        zs.append(z.copy())
        dists.append(dist.copy())
        i = i+1
        start = z[-1]
    #return jnp.concat(dists), jnp.concat(zs)
    return dists, zs

# visualization function - membrane
def plot_membrane_shape(x, F, p, h, change_material, material_type):
    dists, zs = compute_membrane_shape(x, F, p, h, change_material, material_type)
    colors = {'elast' : 'red', 'stiff' : 'blue'}
    labels = {'elast' : 'flexible', 'stiff' : 'stiff'}
    for j in range(len(zs)):
        plt.plot(dists[j], -zs[j] + zs[-1][-1], c=colors[material_type[j]], label=labels[material_type[j]])
        labels[material_type[j]] = None
    plt.title(f"Membrane Shape\nForce is {F:.2f}, pressure is {p:.2f}")
    plt.legend()
    plt.show()

# visualization function - ODE values
def plot_ode_sol(x, F, p, h, change_material, material_type):
    # solve ODE
    full_sol = simulate_ringed_membrane(x, F, p, h, change_material, material_type)

    # make plots for solutions
    plt.figure(figsize=(15,4))
    colors = {'elast' : 'red', 'stiff' : 'blue'}

    plt.subplot(131)
    plt.title('Lambda r')
    plt.xlabel('r')
    labels = {'elast' : 'flexible', 'stiff' : 'stiff'}
    for j in range(len(full_sol)):
        sol = full_sol[j]
        plt.plot(sol.ts, sol.ys[:,0], c=colors[material_type[j]], label=labels[material_type[j]])
        labels[material_type[j]] = None
    plt.legend()

    plt.subplot(132)
    labels = {'elast' : 'flexible', 'stiff' : 'stiff'}
    plt.title('Lambda theta')
    plt.xlabel('r')
    for j in range(len(full_sol)):
        sol = full_sol[j]
        plt.plot(sol.ts, sol.ys[:,1], c=colors[material_type[j]], label=labels[material_type[j]])
        labels[material_type[j]] = None
    plt.legend()

    plt.subplot(133)
    labels = {'elast' : 'flexible', 'stiff' : 'stiff'}
    plt.title('Beta')
    plt.xlabel('r')
    for j in range(len(full_sol)):
        sol = full_sol[j]
        plt.plot(sol.ts, sol.ys[:,2], c=colors[material_type[j]], label=labels[material_type[j]])
        labels[material_type[j]] = None
    plt.legend()
    
    plt.suptitle(f"Force is {F:.2f}, pressure is {p:.2f}")
    plt.legend()
    plt.show()


    # make plot for W grads
    # set figure
    plt.figure(figsize=(25,5))
    labels = {'elast' : 'flexible', 'stiff' : 'stiff'}

    for j in range(len(material_type)):
        # select solution
        sol = full_sol[j]
        # select material constants
        if material_type[j] == 'elast':
            mat_constants = (_MU_elastic, _JM_elastic, h)

        elif material_type[j] == 'stiff':
            mat_constants = (_MU_stiff, _JM_stiff, h)
        else:
            raise NotImplementedError(f'Could not recognize material {material_type[j]}')
        # compute W grads for this piece
        W, W_r, W_theta, W_rr, W_r_theta = vmap(W_grads, in_axes=(0,0,None,None))(sol.ys[:,0], sol.ys[:,1], mat_constants)

        # plot results
        plt.subplot(151)
        plt.plot(sol.ts, W, c=colors[material_type[j]], label=labels[material_type[j]])
        plt.subplot(152)
        plt.plot(sol.ts, W_r, c=colors[material_type[j]], label=labels[material_type[j]])
        plt.subplot(153)
        plt.plot(sol.ts, W_theta, c=colors[material_type[j]], label=labels[material_type[j]])
        plt.subplot(154)
        plt.plot(sol.ts, W_rr, c=colors[material_type[j]], label=labels[material_type[j]])
        plt.subplot(155)
        plt.plot(sol.ts, W_r_theta, c=colors[material_type[j]], label=labels[material_type[j]])

        labels[material_type[j]] = None


    # set legends and titles
    plt.subplot(151)
    plt.title('W')
    plt.legend()
    plt.subplot(152)
    plt.title('W_r')
    plt.legend()
    plt.subplot(153)
    plt.title('W_theta')
    plt.legend()
    plt.subplot(154)
    plt.title('W_rr')
    plt.legend()
    plt.subplot(155)
    plt.title('W_r_theta')
    plt.legend()
    
    plt.suptitle(f"Force is {F:.2f}, pressure is {p:.2f}")
    plt.show()