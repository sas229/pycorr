import pycorr as pc
import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib as mpl
import time
# import multiprocessing as mp

def neighbours(i, j, subsets, solved, preconditioned, roi, iterations, zncc, p_precond, tol):
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

def create_circular_subset(radius):
    """Method to create a subset template."""
    # Create template for extracting circular subset information by checking if pixels are within the subset radius.
    xx, yy = np.meshgrid(np.arange(-radius, radius+1, 1), np.arange(-radius, radius+1, 1))
    dist = np.sqrt(xx**2 + yy**2)
    x_s, y_s = np.where(dist<=radius)
    
    # Create template coordinates matrix.
    n_px = x_s.shape[0]
    subset_coords = np.empty((n_px,2), order='F')
    subset_coords[:,0] = (x_s - radius).astype(float)
    subset_coords[:,1] = (y_s - radius).astype(float)
    return subset_coords

if __name__=='__main__':
    
    datapath = u'/usr/local/pycorr-lib/data/'
    filename_ref = 'IMG_0200'
    filename_tar = 'IMG_0205'
    ext = '.jpg'
    border = 20
    
    # Solver settings.
    max_diffnorm = 1e-5
    max_iterations = 50
    r = 50
    
    template = create_circular_subset(r)
    
    ref = pc.Image(datapath + filename_ref + ext, border)
    tar = pc.Image(datapath + filename_tar + ext, border)
    
    # #%%
    # x_s = 1000.32132
    # y_s = 1000.54352
    # p0 = np.zeros(12)
    # subset = pc.Subset(x_s, y_s, ref, tar, template)
    # subset.solve(max_diffnorm, max_iterations, p0)
    
    # %%
    start_time = time.time()
    
    # Mesh.
    # x = np.arange(100, 950, 50)
    # y = np.arange(100, 950, 50)
    x = np.arange(200, 2400, 20)
    y = np.arange(750, 1750, 20)
    x_m, y_m = np.meshgrid(x, y)
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
            subsets[i,j] = pc.Subset(x_s, y_s, ref, tar, template)
    
    # Solve seed subset.
    tol = 0.75
    p = np.zeros(12)
    subsets[seed_index[0],seed_index[1]].solve(max_diffnorm, max_iterations, p)
    solved[seed_index[0],seed_index[1]] = -1 # -1 indicates that the subset has been solved but is not currently in the queue.
    iterations[seed_index[0],seed_index[1]] = subsets[seed_index[0],seed_index[1]].iterations
    preconditioned[seed_index[0],seed_index[1]] = False
    zncc[seed_index[0],seed_index[1]] = subsets[seed_index[0],seed_index[1]].zncc
    
    # Solve neighbours around seed.
    p_precond = subsets[seed_index[0],seed_index[1]].p
    i = seed_index[0]
    j = seed_index[1]
    solved, preconditioned, roi, iterations, zncc = neighbours(i, j, subsets, solved, preconditioned, roi, iterations, zncc, p_precond, tol)
    
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
        solved, preconditioned, roi, iterations, zncc = neighbours(i, j, subsets, solved, preconditioned, roi, iterations, zncc, p_precond, tol)
        
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # Plot output.
    x = np.vectorize(lambda subset: subset.coord[0])(subsets).flatten()
    y = np.vectorize(lambda subset: subset.coord[1])(subsets).flatten()
    u = np.vectorize(lambda subset: subset.u)(subsets).flatten()
    v = np.vectorize(lambda subset: subset.v)(subsets).flatten()
    SSSIG_hist = np.vectorize(lambda subset: subset.SSSIG)(subsets).flatten()
    sigma_intensity_hist = np.vectorize(lambda subset: subset.sigma_intensity)(subsets).flatten()
    zncc_hist = np.vectorize(lambda subset: subset.zncc)(subsets).flatten()
    iterations_hist = np.vectorize(lambda subset: subset.iterations)(subsets).flatten()
    
    print(np.mean(zncc_hist))
    print(np.std(zncc_hist))
    
    # f, ax = plt.subplots(1)
    # plt.gca().invert_yaxis()
    # colour_array = np.sqrt(v**2 + u**2)
    # plt.imshow(ref.image_gs, cmap='gray')
    # quiver = ax.quiver(x,y,u,-v, colour_array, scale=0.05, scale_units='xy')
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # f.colorbar(quiver)
    # quiver.set_clim(0, np.ceil(np.max(colour_array)))
    # plt.tight_layout()
    
    # f, ax = plt.subplots(1)
    # plt.gca().invert_yaxis()
    # colour_array = u
    # plt.imshow(ref.image_gs, cmap='gray')
    # quiver = ax.quiver(x,y,u,-v, colour_array, scale=0.05, scale_units='xy')
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # f.colorbar(quiver)
    # sym_lim = np.ceil(np.max(np.abs(u)))
    # quiver.set_clim(-sym_lim, sym_lim)
    # plt.tight_layout()
    
    # f, ax = plt.subplots(1)
    # plt.gca().invert_yaxis()
    # colour_array = v
    # plt.imshow(ref.image_gs, cmap='gray')
    # quiver = ax.quiver(x,y,u,-v, colour_array, scale=0.05, scale_units='xy')
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # f.colorbar(quiver)
    # sym_lim = np.ceil(np.max(np.abs(v)))
    # quiver.set_clim(-sym_lim, sym_lim)
    # plt.tight_layout()
    
    # f, ax = plt.subplots(1)
    # ax.hist(SSSIG_hist, 100)
    # plt.tight_layout()
    
    # f, ax = plt.subplots(1)
    # ax.hist(sigma_intensity_hist, 100)
    # plt.tight_layout()
    
    # f, ax = plt.subplots(1)
    # mesh = ax.pcolor(iterations)
    # plt.gca().invert_yaxis()
    # mesh.set_clim(0, max_iterations)
    # f.colorbar(mesh)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # plt.tight_layout()
    
    # f, ax = plt.subplots(1)
    # cmap_reversed = mpl.cm.get_cmap('viridis_r')
    # mesh = ax.pcolor(zncc, cmap=cmap_reversed)
    # mesh.set_clim(tol, 1)
    # plt.gca().invert_yaxis()
    # f.colorbar(mesh)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # plt.tight_layout()
    
    del subsets
    del ref
    del tar


