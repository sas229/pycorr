import numpy as np
from .subset import*

def neighbours(i, j, subsets, solved, preconditioned, roi, iterations, zncc, p_precond, tol, max_diffnorm, max_iterations):
    n_vec = np.asarray(([-1,-1],[-1,0], [-1, 1],[0,-1],[0,1],[1,-1],[1,0],[1,1]), dtype=int)
    for k in range(n_vec.shape[0]):
        # If within the region of interest and if not previously solved.
        if i >= 0 and i < roi.shape[0]-1 and j >= 0 and j < roi.shape[1]-1 and roi[i+n_vec[k,0],j+n_vec[k,1]] == True and solved[i+n_vec[k,0],j+n_vec[k,1]] == 0:
            # First assume nearest-neighbour preconditioning helps.
            subsets[i+n_vec[k,0],j+n_vec[k,1]].solve(max_diffnorm, max_iterations, p_precond)
            # Check if correlation less than tolerance, and if so update queue.
            if subsets[i+n_vec[k,0],j+n_vec[k,1]].zncc > tol:
                # Extract output data and add to queue.
                solved[i+n_vec[k,0],j+n_vec[k,1]] = 1 # 1 indicates that these points are added to the queue.
                preconditioned[i+n_vec[k,0],j+n_vec[k,1]] = True
                iterations[i+n_vec[k,0],j+n_vec[k,1]] = subsets[i+n_vec[k,0],j+n_vec[k,1]].iterations
                zncc[i+n_vec[k,0],j+n_vec[k,1]] = subsets[i+n_vec[k,0],j+n_vec[k,1]].zncc
            else:
                # Otherwise, extrapolate preconditioning and re-solve.
                x = subsets[i,j].coord[0]
                y = subsets[i,j].coord[1]
                p = subsets[i,j].p
                x_c = subsets[i+n_vec[k,0],j+n_vec[k,1]].coord[0]
                y_c = subsets[i+n_vec[k,0],j+n_vec[k,1]].coord[1]
                p_precond[0] = p[0] + p[2]*(x_c-x) + p[3]*(y_c-y);
                p_precond[1] = p[1] + p[4]*(x_c-x) + p[5]*(y_c-y);
                subsets[i+n_vec[k,0],j+n_vec[k,1]].solve(max_diffnorm, max_iterations, p_precond)
                if subsets[i+n_vec[k,0],j+n_vec[k,1]].zncc > tol:
                    # Extract output data and add to queue.
                    solved[i+n_vec[k,0],j+n_vec[k,1]] = 1 # 1 indicates that these points are added to the queue.
                    preconditioned[i+n_vec[k,0],j+n_vec[k,1]] = False
                    iterations[i+n_vec[k,0],j+n_vec[k,1]] = subsets[i+n_vec[k,0],j+n_vec[k,1]].iterations
                    zncc[i+n_vec[k,0],j+n_vec[k,1]] = subsets[i+n_vec[k,0],j+n_vec[k,1]].zncc
                else:
                    # Otherwise resort to using standard initial guess method
                    subsets[i+n_vec[k,0],j+n_vec[k,1]].solve(max_diffnorm, max_iterations)
                    if subsets[i+n_vec[k,0],j+n_vec[k,1]].zncc > tol:
                        # Extract output data and add to queue.
                        solved[i+n_vec[k,0],j+n_vec[k,1]] = 1 # 1 indicates that these points are added to the queue.
                        preconditioned[i+n_vec[k,0],j+n_vec[k,1]] = False
                        iterations[i+n_vec[k,0],j+n_vec[k,1]] = subsets[i+n_vec[k,0],j+n_vec[k,1]].iterations
                        zncc[i+n_vec[k,0],j+n_vec[k,1]] = subsets[i+n_vec[k,0],j+n_vec[k,1]].zncc
                    else:
                        print('Cannot solve this subset...')
    return solved, preconditioned, roi, iterations, zncc



def reliability_guided(x_m, y_m, ref, tar, template, max_diffnorm, max_iterations, p, tol):
    roi = np.ones(x_m.shape, dtype=bool)
    preconditioned = np.zeros(x_m.shape, dtype=bool)
    solved = np.zeros(x_m.shape, dtype=int)
    seed_index = np.asarray((3,3), dtype=int)
    iterations = np.zeros(x_m.shape, dtype=int)
    zncc = np.zeros(x_m.shape, dtype=np.float64)
    SSSIG = np.zeros(x_m.shape, dtype=np.float64)
    stddev_intensity = np.zeros(x_m.shape, dtype=np.float64)
    
    # Create subsets.
    subsets = np.empty((x_m.shape), dtype=object)
    for i in range(x_m.shape[0]):
        for j in range(x_m.shape[1]):
            x_s = x_m[i,j]
            y_s = y_m[i,j]
            subsets[i,j] = Subset(x_s, y_s, ref, tar, template)
    
    # Solve seed subset.

    subsets[seed_index[0],seed_index[1]].solve(max_diffnorm, max_iterations, p)
    solved[seed_index[0],seed_index[1]] = -1 # -1 indicates that the subset has been solved but is not currently in the queue.
    iterations[seed_index[0],seed_index[1]] = subsets[seed_index[0],seed_index[1]].iterations
    preconditioned[seed_index[0],seed_index[1]] = False
    zncc[seed_index[0],seed_index[1]] = subsets[seed_index[0],seed_index[1]].zncc
    
    # Solve neighbours around seed.
    p_precond = subsets[seed_index[0],seed_index[1]].p
    i = seed_index[0]
    j = seed_index[1]
    solved, preconditioned, roi, iterations, zncc = neighbours(i, j, subsets, solved, preconditioned, roi, iterations, zncc, p_precond, tol, max_diffnorm, max_iterations)
    
    # Work through sorted queue until empty, adding neighbours recursively.
    while np.max(solved) > -1:
        # Identify next subset to solve neighbours at.
        queue = solved*zncc
        # print(len(np.where(queue>0)[0]))
        i, j = np.unravel_index(queue.argmax(), queue.shape)
        p_precond = subsets[i,j].p
        
        # Remove current subset from queue.
        solved[i,j] = -1
        
        # Solve neighbours for current subset.
        solved, preconditioned, roi, iterations, zncc = neighbours(i, j, subsets, solved, preconditioned, roi, iterations, zncc, p_precond, tol, max_diffnorm, max_iterations)
    
    x = np.vectorize(lambda subset: subset.coord[0])(subsets)
    y = np.vectorize(lambda subset: subset.coord[1])(subsets)
    u = np.vectorize(lambda subset: subset.u)(subsets)
    v = np.vectorize(lambda subset: subset.v)(subsets)
    SSSIG_hist = np.vectorize(lambda subset: subset.SSSIG)(subsets).flatten()
    sigma_intensity_hist = np.vectorize(lambda subset: subset.sigma_intensity)(subsets).flatten()
    zncc_hist = np.vectorize(lambda subset: subset.zncc)(subsets).flatten()
    iterations_hist = np.vectorize(lambda subset: subset.iterations)(subsets).flatten()
    return x, y, u, v, SSSIG_hist, sigma_intensity_hist, zncc_hist, iterations_hist