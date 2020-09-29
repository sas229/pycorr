import numpy as np

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
