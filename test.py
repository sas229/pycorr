import pycorr as pc
import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib as mpl
import time
# import multiprocessing as mp

if __name__=='__main__':
    
    datapath = u'data/'
    filename_ref = 'IMG_0200'
    filename_tar = 'IMG_0205'
    ext = '.jpg'
    border = 20
    
    # Solver settings.
    max_diffnorm = 1e-5
    max_iterations = 50
    r = 50
    tol = 0.75
    p = np.zeros(12)
    
    template = pc.create_circular_subset(r)
    
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

    x, y, u, v, SSSIG_hist, sigma_intensity_hist, zncc_hist, iterations_hist = pc.reliability_guided(x_m, y_m, ref, tar, template, max_diffnorm, max_iterations, p, tol)
        
    print("--- %s seconds ---" % (time.time() - start_time))
    
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
    
    del ref
    del tar


