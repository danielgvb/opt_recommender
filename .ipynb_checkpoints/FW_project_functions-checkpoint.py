import pandas as pd
import numpy as np

# REMARK: The main data structured being used are SciPy's sparse matrices
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds, norm

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import os
import pickle

# This functions will help transforming sparse matrices (sp)
# to serial objects (s) and back to use them as keys of dictionaries
sp2s = lambda M_sp: pickle.dumps(M_sp)
s2sp = lambda M_s:  pickle.loads(M_s)


# --------------------------------------------------- Main functions

def cost_func_sp(U_sp, X_sp, mask):
    """
    U_sp: Data matrix to be completed.
    X_sp: Variable matrix to be optimized.
    mask: Boolean mask indicating which entries
          of the U_sp matrix are known. Equivalent
          to the set of known indices J.
    """
    return (U_sp - X_sp).multiply(mask).power(2).sum()

def gradient_func_sp(U_sp, X_sp, mask):
    """
    U_sp: Data matrix to be completed.
    X_sp: Variable matrix to be optimized.
    mask: Boolean mask indicating which entries
          of the U_sp matrix are known. Equivalent
          to the set of known indices J.
    """
    return 2 * (X_sp - U_sp).multiply(mask)

def linear_oracle_sp(grad, delta):
    """
    This function produces the closest possible matrix
    in the feasible set to the exact gradient of the function
    at that iteration's X_sp point.
    
    grad: Sparse matrix containing the exact gradient.
    delta: Scalar parameter of the problem.
    """
    
    # SciPy's function svds computes the only the first component
    # of the SVD decomposition of -grad. It also leverages the fact
    # that the input is a native sparse matrix to efficiently
    # obtain the result.
    u, _, vt = svds(-grad, k=1)
    s = delta * csr_matrix(u) @ csr_matrix(vt)
    return s


def frank_wolfe_sp(U_sp, delta, mask, max_iter=100, tol=1e-6, disable=True):
    
    """
    Simple Frank-Wolfe algorithm
    
    U_sp: Sparse matrix containing the matrix to be completed.
    delta: Scalar parameter of the problem.
    mask: Boolean mask indicating which entries of the U_sp
          matrix are known. Equivalent to the set of known
          indices J.
    """

    l = []
    start = pd.Timestamp.now()

    # The all-zeros matrix is created as the initial solution as
    # it lies within the feasible set and it is costless to generate
    # as a sparse matrix.
    X_sp = csr_matrix(U_sp.shape)

    for k in tqdm(range(max_iter), disable=disable):

        grad = gradient_func_sp(U_sp, X_sp, mask)  # Compute the gradient
        s_t = linear_oracle_sp(grad, delta)        # Get the LMO
        d_FW = s_t - X_sp                          # Obtain the FW direction (only to compute the gap)
        g_FW = -(grad.multiply(d_FW)).sum()        # Calculate the gap (only to compare convergence with
                                                   #                    the other methods)

        l.append((pd.Timestamp.now() - start,      # Collect the results
                  cost_func_sp(U_sp, X_sp, mask),
                  g_FW))

        gamma = 2 / (k + 2)                        # Compute the proposed step-size
        X_new = (1 - gamma) * X_sp + gamma * s_t   # Make a in the s_t direction

        # Check for convergence
        if norm(X_new - X_sp, 'fro') < tol:
            break

        X_sp = X_new

    # Prepare a dataframe containing the algorithm's evolution
    costs = pd.DataFrame(l, columns=['time_ms', 'cost', 'g_fw'])
    costs['time_ms'] = costs.time_ms.dt.total_seconds()
    costs.delta = delta
    
    return X_sp, costs


def away_steps_frank_wolfe_sp(U_sp, delta, mask, max_iter=100, tol=1e-6, disable=True):
    """
    Away Step Frank-Wolfe algorithm
    
    U_sp: Sparse matrix containing the matrix to be completed.
    delta: Scalar parameter of the problem.
    mask: Boolean mask indicating which entries of the U_sp
          matrix are known. Equivalent to the set of known
          indices J.
    """

    l = []
    start = pd.Timestamp.now()

    # The all-zeros matrix is created as the initial solution as
    # it lies within the feasible set and it is costless to generate
    # as a sparse matrix.
    X_sp = csr_matrix(U_sp.shape)

    # The set S containing the directions to keep track of
    # is implemented as a dictionary. This allows to store
    # the alpha value alongside each vector v.
    # Additionally, since sparse matrices are not hashable,
    # they cannot be used as keys of the dictionary. Hence,
    # these objectes are first serialized (sp2s: sparse to
    # serial) as a solution to this technical issue.
    S = {sp2s(X_sp): 1.0}

    for _ in tqdm(range(max_iter), disable=disable):

        grad = gradient_func_sp(U_sp, X_sp, mask)  # Compute the gradient
        s_t = linear_oracle_sp(grad, delta)        # Get the LMO
        d_FW = s_t - X_sp                          # Obtain the FW direction (only to compute the gap)

        # Compute the step-away direction my maximizing the inner product of the
        # gradient against the stored vectors in S.
        v_t_key = max(S.keys(), key=lambda v_s: s2sp(v_s).multiply(grad).sum())
        v_t = s2sp(v_t_key)   # The best vector is deserialized (s2sp: serial to sparse)
        d_A = X_sp - v_t      # Compute the step-away direction

        # Convergence criterion
        if (g_FW := -grad.multiply(d_FW).sum()) <= tol:
            break

        # Condition to step in the FW direction
        elif g_FW >= -grad.multiply(d_A).sum():
            step, d_t, gamma_max = "FW", d_FW, 1

        # Condition to step in the away direction
        else:
            alpha = S[v_t_key]
            step, d_t, gamma_max = "A", d_A, alpha / (1 - alpha)

        l.append((pd.Timestamp.now() - start,      # Collect the results
                  cost_func_sp(U_sp, X_sp, mask),
                  g_FW))

        # Compute the stepsize according to the solution of a linesearch
        # in the chosen direction.
        gamma = np.clip(
                    (U_sp - X_sp).multiply(mask).multiply(d_t).sum() / d_t.multiply(mask).power(2).sum(),
                0, gamma_max)

        X_sp = X_sp + gamma*d_t  # Perform a step

        # Update the weigths.
        # Only v vectors with strictly positive alpha values
        # are updated and kept track of.
        if step == "FW":   # Frank-Wolfe update
            s_t_key = sp2s(s_t)

            if gamma == 1:   # If gamma==1, reset the tracking set S
                S = { s_t_key : S.get(s_t_key,0) }

            # Update rule for all the elements of the tracking set
            S = {v:new_alpha for v,alpha in S.items()
                 if (new_alpha := (1-gamma)*alpha) > 0}

            # Sum gamma to the new atom's alpha, only keep it if the
            # result is positive
            if (new_alpha_s_t := (S.get(s_t_key,0) + gamma)) > 0:
                S[s_t_key] = new_alpha_s_t

        elif step=="A":   # Away step update
            
            # If gamma equals gamma_max, remove the v_t vector
            # from the tracking set
            if gamma == gamma_max:
                del S[v_t_key]

            # Update rule for all the elements of the tracking set
            S = {v:new_alpha for v,alpha in S.items()
                 if (new_alpha := (1+gamma)*alpha) > 0}

            # Subtract gamma to the v_t vector's alpha, only keep
            # it if the result is positive
            if (new_alpha_v_t := (S.get(v_t_key,0) - gamma)) > 0:
                S[s_t_key] = new_alpha_v_t
                
    # Prepare a dataframe containing the algorithm's evolution
    costs = pd.DataFrame(l,columns=['time_ms','cost','g_fw'])
    costs['time_ms'] = costs.time_ms.dt.total_seconds()
    costs.delta = delta
    
    return X_sp, costs


def pairwise_frank_wolfe_sp(U_sp, delta, mask, max_iter=100, tol=1e-6, disable=True):
    """
    Pairwise Frank-Wolfe algorithm
    
    U_sp: Sparse matrix containing the matrix to be completed.
    delta: Scalar parameter of the problem.
    mask: Boolean mask indicating which entries of the U_sp
          matrix are known. Equivalent to the set of known
          indices J.
    """

    l = []
    start = pd.Timestamp.now()

    # The all-zeros matrix is created as the initial solution as
    # it lies within the feasible set and it is costless to generate
    # as a sparse matrix.
    X_sp = csr_matrix(U_sp.shape)

    # The set S containing the directions to keep track of
    # is implemented as a dictionary. This allows to store
    # the alpha value alongside each vector v.
    # Additionally, since sparse matrices are not hashable,
    # they cannot be used as keys of the dictionary. Hence,
    # these objectes are first serialized (sp2s: sparse to
    # serial) as a solution to this technical issue.
    S = {sp2s(X_sp): 1.0}

    for _ in tqdm(range(max_iter), disable=disable):

        grad = gradient_func_sp(U_sp, X_sp, mask)  # Compute the gradient
        s_t = linear_oracle_sp(grad, delta)        # Get the LMO
        d_FW = s_t - X_sp                          # Obtain the FW direction (only to compute the gap)

        # Compute the step-away direction my maximizing the inner product of the
        # gradient against the stored vectors in S.
        v_t_key = max(S.keys(), key=lambda v_s: s2sp(v_s).multiply(grad).sum())
        v_t = s2sp(v_t_key)   # The best vector is deserialized (s2sp: serial to sparse)

        # Convergence criterion
        if (g_FW := -grad.multiply(d_FW).sum()) <= tol:
            break

        l.append((pd.Timestamp.now() - start,      # Collect the results
                  cost_func_sp(U_sp, X_sp, mask),
                  g_FW))
        
        gamma_max = S[v_t_key] # Maximum possible step size
        d_t = s_t - v_t        # Direction of the pairwise algorithm
        
        # Compute the stepsize according to the solution of a linesearch
        # in the computed direction.
        gamma = np.clip(
                    (U_sp - X_sp).multiply(mask).multiply(d_t).sum() / d_t.multiply(mask).power(2).sum(),
                0, gamma_max)

        X_sp = X_sp + gamma*d_t  # Perform a step

        # If the gamma is strictly positive, procede updating the alphas.
        # Else, there is no point on spending time with this calculation.
        if gamma > 0:
            s_t_key = sp2s(s_t)
            S[v_t_key] = S.get(v_t_key,0) - gamma
            S[s_t_key] = S.get(s_t_key,0) + gamma
            
            if S[v_t_key] <= 0: del S[v_t_key]   # If v_t's alpha becomes less
                                                 # than zero, remove it from S

    # Prepare a dataframe containing the algorithm's evolution
    costs = pd.DataFrame(l,columns=['time_ms','cost','g_fw'])
    costs['time_ms'] = costs.time_ms.dt.total_seconds()
    costs.delta = delta
    
    return X_sp, costs




# --------------------------------------------------- Auxiliar plotting functions

# Plot the results of a single run of a FW optimization process
def plot_summary(title, costs, dataset, figsize=(12, 5), savefig=True):
    MIT = costs.time_ms.diff().mean()
    TT = costs.iloc[-1,0]
    delta = costs.delta

    fig, axs = plt.subplots(1,1+('g_fw'in costs), figsize=figsize)
    if not isinstance(axs,np.ndarray): axs = np.array([axs])

    fig.suptitle(f"{title}\n" +
                 f"MIT = {MIT:,.2f} s/it\n" +
                 f"TT = {TT:,.2f} s\n" +
                 f"$\delta$ = {delta:,.2f}")

    axs[0].plot(costs.index, costs.cost, color=costs.color)
    axs[0].scatter(costs.index, costs.cost, c='red', marker='o', s=8)

    axs[0].set_title('Cost function vs iteration')
    axs[0].set_xlabel('Iteration no.')
    axs[0].set_ylabel('Cost')
    axs[0].set_yscale("log")

    axs[0].grid(True, which="both", ls="--", color='gray', alpha=0.25)
    axs[0].grid(True, which="minor", ls="--", color='gray', alpha=0.2, axis='y')
    axs[0].grid(True, which="major", ls="--", color='gray', alpha=0.35, axis='x')
    
    axs[1].plot(costs.index, costs.g_fw, color=costs.color)
    axs[1].scatter(costs.index, costs.g_fw, c='red', marker='o', s=8)

    axs[1].set_title('Gap vs iteration')
    axs[1].set_xlabel('Iteration no.')
    axs[1].set_ylabel('Gap')
    axs[1].set_yscale("log")

    axs[1].grid(True, which="both", ls="--", color='gray', alpha=0.25)
    axs[1].grid(True, which="minor", ls="--", color='gray', alpha=0.2, axis='y')
    axs[1].grid(True, which="major", ls="--", color='gray', alpha=0.35, axis='x')

    fig.tight_layout()
    
    if savefig:
        path = os.path.join('plots',
                            f"{dataset}__{title.replace(' ','_').lower()}__summary_plot.png")
        fig.savefig(path)
    fig.show()
    
    
    
    
# Plot the results of multiple runs of FW
# optimization process in order to compare
# their behaviour

def plot_comparisons(results, kwargs, dataset, figsize=(14, 6), savefig=True, loc="lower right"):

    fig, axs = plt.subplots(1,2,figsize=figsize)

    fig.suptitle(f"FW variants comparison\n" +
                 f"$\delta$ = {kwargs['delta']}\n" +
                 f"$shape(U)$ = {kwargs['U_sp'].shape}\n" +
                 f"Non-null entries: {kwargs['U_sp'].count_nonzero() / (kwargs['U_sp'].shape[0]*kwargs['U_sp'].shape[1]):.2%}")

    for name,dct in results.items():
        axs[0].plot(dct['costs'].index, dct['costs'].cost,
                    color=dct['costs'].color, linestyle=dct['costs'].linestyle, label=name)

    axs[0].set_title('Cost function vs iteration')
    axs[0].set_xlabel('Iteration no.')
    axs[0].set_ylabel('Objective function')
    axs[0].set_yscale("log")

    axs[0].grid(True, which="both", ls="--", color='gray', alpha=0.25)
    axs[0].grid(True, which="minor", ls="--", color='gray', alpha=0.2, axis='y')
    axs[0].grid(True, which="major", ls="--", color='gray', alpha=0.35, axis='x')


    for name,dct in results.items():
        axs[1].plot(dct['costs'].index, dct['costs'].g_fw,
                    color=dct['costs'].color, linestyle=dct['costs'].linestyle, label=name)

    axs[1].set_title('Gap vs iteration')
    axs[1].set_xlabel('Iteration no.')
    axs[1].set_ylabel('Gap')
    axs[1].set_yscale("log")

    axs[1].grid(True, which="both", ls="--", color='gray', alpha=0.25)
    axs[1].grid(True, which="minor", ls="--", color='gray', alpha=0.2, axis='y')
    axs[1].grid(True, which="major", ls="--", color='gray', alpha=0.35, axis='x')

    axs[0].legend(loc=loc)
    
    fig.tight_layout()
    
    if savefig:
        path = os.path.join('plots',
                            f"{dataset}__comparison_plot.png")
        fig.savefig(path)
    fig.show()
    
    


    
# Run several tests of a given FW variant by varying
# the \delta parameter each time with the goal of
# studying how the soution is affected as this
# parameter does.

# The FW function version used is passed as a
# parameter.

# A single plot is generated and plotted in the end.

def delta_change_plot(name, fw_function, delta_min, delta_max, kwargs, dataset, N=50, savefig=True):
    
    DELTA = np.logspace(delta_min,delta_max,N)
    X_list = []
    costs_list = []
    
    lowest_cost = []
    X_sizes = []
    
    for delta in tqdm(DELTA):
        print(f'New delta: {delta:,.2f}')

        kwargs2 = kwargs.copy()
        kwargs2['delta'] = delta

        X, costs = fw_function(**kwargs2)
        X_list.append(X)
        costs_list.append(costs)

        lowest_cost.append(costs.cost.iloc[-1])
        X_sizes.append(norm(X,'fro'))

    fig, ax1 = plt.subplots()

    fig.suptitle('Effect of parameter $\delta$\n' +
                 f"{name}\n" +
                 f"$shape(U)$ = {kwargs['U_sp'].shape}")

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.plot(DELTA, lowest_cost, 'b-', label='Lowest cost obtained')
    ax1.set_xlabel('DELTA')
    ax1.set_ylabel('Lowest Cost', color='b')

    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.plot(DELTA, X_sizes, 'r-', label=r'$||X||$')
    ax2.set_ylabel(r'$||X||$', color='r')

    fig.tight_layout()
    
    if savefig:
        path = os.path.join('plots',
                            f"{dataset}__{name.replace(' ','_').lower()}__delta_change.png")
        fig.savefig(path)
    plt.show()
    
    
    
# Same function as before, but here an axis object is passed to
# the function. The idea is to produce a subplot of a larger figure,
# which is assumed to be created outside the function.

def delta_change_subplot(name, fw_function, delta_min, delta_max, kwargs, ax, N=50):
    
    DELTA = np.logspace(delta_min,delta_max,N)
    X_list = []
    costs_list = []
    for delta in tqdm(DELTA):

        kwargs2 = kwargs.copy()
        kwargs2['delta'] = delta

        X, costs = fw_function(**kwargs2)
        X_list.append(X)
        costs_list.append(costs)

    lowest_cost = [df.cost.iloc[-1] for df in costs_list]
    X_sizes = [norm(X,'fro') for X in X_list]
    
    ax.cla()

    ax.set_title(name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(DELTA, lowest_cost, 'b-', label='Lowest cost obtained')
    ax.set_xlabel('$\delta$')
    ax.set_ylabel('Lowest Cost', color='b')

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.plot(DELTA, X_sizes, 'r-', label=r'$||X||$')
    ax2.set_ylabel(r'$||X||$', color='r')