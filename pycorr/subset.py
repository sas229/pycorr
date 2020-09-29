import cv2
import numpy as np
import pycorr_extensions as cpp

def create_circular_subset(radius):
    """Method to create a circular subset template."""
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

class Subset:
    """Subset class for pycorr.
    
    Parameters
    ----------
    x : float
        Horizontal subset coordinate.
    y : float
        Vertical subset coordinate.
    f_img : pycorr.Image
        Reference image of pycorr.Image class, instantiated by :mod:`~image.Image`.
    g_img : pycorr.Image
        Target image of pycorr.Image class, instantiated by :mod:`~image.Image`.
    template_coords: int
        Subset template coordinates.
    
    
    Attributes
    ----------
    coord : `numpy.ndarray` (x, y)
        1D array of the coordinates of the subset in reference image of type `float`.
    f_img : pycorr.Image
        Reference image of pycorr.Image class, instantiated by :mod:`~image.Image`.
    g_img : pycorr.Image
        Target image of pycorr.Image class, instantiated by :mod:`~image.Image`.
    template_coords: `numpy.ndarray` (Nx, 2)
        2D array of subset template coordinates of type `int`.
    init_guess_size : int
        Size of subset used to define the initial guess, approximated by private method :meth:`~_get_initial_guess_size`. 
    f_coords : `numpy.ndarray` (Nx, 2)
        2D array of reference subset coordinates of type `float`, computed by private method :meth:`~_get_f_coords`.
    f : `numpy.ndarray` (Nx, 1)
        1D array of reference intensity values for reference subset of type `float`, computed by private method :meth:`~_get_f`.
    f_m : float
        Mean intensity of the subset in the reference image, computed by private method :meth:`~_get_f_m`.
    Delta_f : float
        Square root of the sum of the square of the variance from the mean of the reference subset intensities, computed by private method :meth:`~_get_Delta_f`.
    SSSIG : float
        Sum of the square of the reference subset intensity gradients, computed by private method :meth:`~_get_SSSIG`.
    sigma_intensity : float
        Standard deviaition of the reference subset intensities, computed by private method :meth:`~_get_sigma_intensity`.
    g_coords : `numpy.ndarray` (Nx, 2)
        2D array of target subset coordinates of type `float`, computed by private method :meth:`~_get_g_coords`.
    g : `numpy.ndarray` (Nx, 1)
        1D array of target intensity values for reference subset of type `float`, computed by private method :meth:`~_get_g`.
    g_m : float
        Mean intensity of the subset in the target image, computed by private method :meth:`~_get_g_m`.
    Delta_g : float
        Square root of the sum of the square of the variance from the mean of the target subset intensities, computed by private method :meth:`~_get_Delta_g`.
    derivatives : `numpy.ndarray` (Nx, Ny)
        2D array of derivatives for reference subset of type `float`, computed by private method :meth:`~_get_derivatives`.
    hessian : `numpy.ndarray` (Nx, Ny)
        2D array of second derivatives for reference subset of type `float`, computed by private method :meth:`~_get_hessian`.
    Delta_p : `numpy.ndarray` (Nx, 1)
        1D array of the increments in the warp function parameters of type `float`, computed by private method :meth:`~_get_Delta_p`.
    p_new : `numpy.ndarray` (Nx, 1)
        1D array of new warp function parameters of type `float`, computed by private method :meth:`~_get_p_new`.
    norm : float
        Custom norm of the increment in the warp function parameters after Gao et al. (2015), computed by private method :meth:`~_get_norm`.
    znssd : float
        Zero-normalised sum of squared differences coefficient, computed by private method :meth:`~_get_correlation`.
    zncc : float
        Zero-normalised cross-correlation coefficient, computed by private method :meth:`~_get_correlation`.
    """
    
    def __init__(self, x, y, f_img, g_img, template_coords):
        """Initialisation of pycorr subset object."""
        # Calculate method independent reference subset quantities.
        self.coord = np.array([x, y])
        self.f_img = f_img
        self.g_img = g_img
        self.template_coords = template_coords
        self._get_initial_guess_size()
        self._get_f_coords()
        self._get_f()
        self._get_f_m()
        self._get_Delta_f()
        self._get_SSSIG()
        self._get_sigma_intensity()
            
    def solve(self, max_norm=1e-5, max_iterations=50, p_0=np.zeros(6)):
        r"""Method to solve for the subset displacements using the ICGN method.
        
        Parameters
        ----------
        max_norm : float, optional
            Exit criterion for norm of increment in warp function. Defaults to value of :math:`1 \cdot 10^{-5}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations. Defaults to value of 50.
        p_0 : ndarray, optional
            1D array of warp function parameters with `float` type. 
        
        
        .. note::
            * If all members of the warp function parameter array are zero, then an initial guess at the subset displacement is performed by :meth:`~_get_initial_guess`.
            * Otherwise, if any members of the warp function parameter array are non-zero, the array is used to precondition the ICGN computation directly. 
            * If not specified, the solver defaults to a first order warp function. 
            * If an array length of 12 is specified a second order warp function is assumed.
         
        .. seealso::
            :meth:`~_get_initial_guess_size`
            :meth:`~_get_initial_guess`
        """
        
        # Determine computation mode (defaults to first order ICGN).
        if p_0.shape[0] == 6:
            self.mode = 1
        elif p_0.shape[0] == 12:
            self.mode = 2
            
        # Store solver settings.
        self.max_norm = max_norm
        self.max_iterations = max_iterations
        
        # Calculate reference subset quantities.
        self._get_derivatives()
        self._get_hessian()
        
        # Compute initial guess if warp vector initialised with zeros, otherwise precondition.
        if np.sum(p_0**2) == 0: 
            self._get_initial_guess()
            self.p = self.p_init
        else:
            self.p = p_0
        
        # Perform ICGN iterations.
        self.iterations = 0
        self.norm = 1
        self.u = 0.0
        self.v = 0.0
        while self.norm > self.max_norm and self.iterations < self.max_iterations:
            self._get_g_coords()
            self._get_g()
            self._get_g_m()
            self._get_Delta_g()
            self._get_Delta_p()
            self._get_p_new()
            self._get_norm()
            self.iterations += 1
            self.p = self.p_new
            
        # Compute correlation and store output.
        self._get_correlation()
        self.u = self.p[0]
        self.v = self.p[1]
        return
    
    def _get_initial_guess_size(self):
        r"""Private method to estimate the size of square subset to use in the initial guess.
        
        **Implementation**:
        
        The initial guess subset size is a square of side length :math:`s` such that:
            
        .. math::
            
           s = \sqrt{n}
           
        where :math:`n` is the set of pixels that comprise the subset template.
        
        .. seealso::
            :meth:`~_get_initial_guess`
        """
        self.initial_guess_size = np.round(np.sqrt(np.shape(self.template_coords)[0]), 1)
        return
    
    def _get_initial_guess(self):
        r"""Private method to compute an initial guess of the subset displacement using OpenCV function :py:meth:`cv2.matchTemplate` and the Normalised Cross-Correlation (NCC) criteria.
        
        **Implementation**:
            
        .. math::
            
            C_{NCC (x_{g}, y_{g})} = \frac{\sum_{\left(x, y \right) \in n} \left( f_{\left(x, y\right)} \cdot g_{\left(x_{g}+x, y_{g}+y \right)} \right)}{\sqrt{\sum_{(x, y) \in n} f_{\left(x, y\right)}^{2} \cdot \sum_{(x, y) \in n} g_{\left(x_{g}+x, y_{g}+y\right)}^{2}}}
        
        where :math:`(x_{g}, y_{g})` are the coordinates of a point in the target image :math:`g`, the coordinates :math:`(x, y)` denote a point within the subset of the reference image :math:`f`  relative to its centre :math:`(x_{0}, y_{0})`, and :math:`n` is the set of coordinates that comprise the subset.
        The difference between the reference subset coordinates :math:`(x_{0}, y_{0})` and the coordinates :math:`(x_{g}, y_{g})` that exhibit the maximum value of :math:`C_{NCC}` denotes the initial guess of the warp function parameters :math:`u` and :math:`v`.
        
        .. note::
            * The size of the subset used in the initial guess is automatically approximated as a square of equal area to the subset to be used in the ICGN computations using private method meth:`~_get_initial_guess_size`.
         
        .. seealso::
            :meth:`~_get_initial_guess_size`
        """
        # Extract square subset for initial guess.
        x = self.coord[0]
        y = self.coord[1]
        x_min = (np.round(x, 0)-self.initial_guess_size/2).astype(int)
        x_max = (np.round(x, 0)+self.initial_guess_size/2).astype(int)
        y_min = (np.round(y, 0)-self.initial_guess_size/2).astype(int)
        y_max = (np.round(y, 0)+self.initial_guess_size/2).astype(int)
        subset = self.f_img.image_gs.astype(np.float32)[y_min:y_max, x_min:x_max]
        
        # Apply template matching technique.
        res = cv2.matchTemplate(self.g_img.image_gs.astype(np.float32), subset, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Create initialised warp vector with affine displacements preconditioned.
        if self.mode == 1:
            self.p_init = np.zeros(6)
        elif self.mode == 2:
            self.p_init = np.zeros(12)
        self.p_init[0] = (max_loc[0] + self.initial_guess_size/2) - x
        self.p_init[1] = (max_loc[1] + self.initial_guess_size/2) - y
        return
    
    def _get_f_coords(self):
        r"""Private method to calculate the reference subset coordinates from the subset template coordinates via the C++ extension :py:meth:`pycorr.cpp.get_f_coords`.
        The reference coordinates :math:`(x, y)` are calculated by adding the subset template coordinate, :math:`(x_{t}, y_{t})`, to the subset coordinates, :math:`(x_{0}, y_{0})`:
        
        .. math::
            x = x_{0} + x_{t} \\
            y = y_{0} + y_{t}
        """
        self.f_coords = cpp.get_f_coords(self.coord, self.template_coords)
        return
    
    def _get_f(self):
        r"""Private method to calculate the subset intensities in the reference image via the C++ extension :py:meth:`pycorr.cpp.get_f`.
        The image intensity at each coordinate of the reference subset, :math:`f_{(x, y)}`, is estimated using bi-quintic B-spline image intensity interpolation.
        First, the sub-pixel component of the position of each point in the subset is computed as follows from the current coordinates:
            
        .. math::
            
            \delta x = x - \left\lfloor x \right\rfloor \\
            \delta y = y - \left\lfloor y \right\rfloor
            
        where :math:`\left\lfloor x \right\rfloor` and :math:`\left\lfloor y \right\rfloor` are the floor of the coordinates of each point within the reference subset.
        The interpolated pixel intensity at the current sub-pixel coordinate, :math:`f_{(x, y)}`, is then calculated by performing the following operation:
            
        .. math::
            
            f_{(x, y)} = \begin{bmatrix} 1 & \delta y & \delta y^2 & \delta y^3 & \delta y^4 & \delta y^5 \end{bmatrix} \cdot
            \mathbf{QK} \cdot \mathbf{C}_{f \left(\left\lfloor x \right\rfloor-2:\left\lfloor x \right\rfloor+3, \left\lfloor y \right\rfloor-2:\left\lfloor y \right\rfloor+3\right)} \cdot \mathbf{QK^T} \cdot
            \begin{bmatrix} 1 \\ \delta x \\ \delta x^2 \\ \delta x^3 \\ \delta x^4 \\ \delta x^5 \end{bmatrix}
            
        where :math:`\mathbf{QK} \cdot \mathbf{C}_{f} \cdot \mathbf{QK^T}` was precomputed for the reference image :math:`f` by :py:meth:`pycorr.Image._get_QK_C_QKT`.
        """
        self.f = cpp.get_intensity(self.f_coords, self.f_img.QK_C_QKT)
        return
    
    def _get_f_m(self):
        r"""Private method to calculate the mean reference subset grayscale intensity via the C++ extension :py:meth:`pycorr.cpp.get_f_m`.
        
        The quantity :math:`f_{m}` is the mean reference subset pixel intensity:
            
        .. math::
            
            f_{m} = \sum_{(x, y) \in n} \frac{f_{(x, y)}}{n} 
        
        where :math:`f_{(x, y)}` is the image intensity at a point :math:`(x, y)` within the subset,
        and :math:`n` is the set of coordinates that comprise the subset.
        """
        self.f_m = cpp.get_f_m(self.f)
        return
    
    def _get_Delta_f(self):
        r"""Private method to calculate the square root of the sum of the square of the variance from the mean of the reference subset intensities via the C++ extension :py:meth:`pycorr.cpp.get_Delta_f`.
        
        The quantity :math:`\Delta f` is calculated as follows:
            
        .. math:: 
            
            \Delta f = \sqrt{ \sum_{(x, y) \in n} \left( f_{(x, y)} - f_{m} \right)^2}
            
        where :math:`f_{(x, y)}` is the image intensity at a point :math:`(x, y)` within the subset,
        :math:`f_{m}` is the mean subset intensity, and :math:`n` is the set of coordinates that comprise the subset.
        """
        self.Delta_f = cpp.get_Delta_f(self.f, self.f_m)
        return
    
    def _get_SSSIG(self):
        r"""Private method to calculate the Sum of Squared Subset Intensity Gradients (:math:`SSSIG`) after Pan et al. (2008) for the reference subset via a C++ extension.
        The horizontal and vertical image intensity gradients :math:`\nabla f_{x}` and :math:`\nabla f_{y}`
        are calculated using bi-quintic B-spline image intensity interpolation as follows:
        
        .. math::
            
            \nabla f_{x} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \end{bmatrix} \cdot
            \mathbf{QK} \cdot \mathbf{C}_{f \left(\left\lfloor x \right\rfloor-2:\left\lfloor x \right\rfloor+3, \left\lfloor y \right\rfloor-2:\left\lfloor y \right\rfloor+3\right)} \cdot \mathbf{QK^T} \cdot
            \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix} \\
                
            \nabla f_{y} = \begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix} \cdot
            \mathbf{QK} \cdot \mathbf{C}_{f \left(\left\lfloor x \right\rfloor-2:\left\lfloor x \right\rfloor+3, \left\lfloor y \right\rfloor-2:\left\lfloor y \right\rfloor+3\right)} \cdot \mathbf{QK^T} \cdot
            \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
        
        where :math:`\left\lfloor x \right\rfloor` and :math:`\left\lfloor y \right\rfloor` are the floor of the coordinates of each point within the reference subset, and :math:`\mathbf{QK} \cdot \mathbf{C}_{f} \cdot \mathbf{QK^T}` was precomputed for the reference image :math:`f` by :py:meth:`pycorr.Image._get_QK_C_QKT`.
        
        In this implementation an estimate of :math:`SSSIG` is computed by averaging the horizontal and vertical image intensity gradients:
            
        .. math::
            
            SSSIG \approx \sum_{(x, y) \in n} \frac{1}{2}\left[\left(\nabla f_{x}\right)^{2}+\left(\nabla f_{y}\right)^{2}\right]
        
        where :math:`n` is the set of coordinates that comprise the subset.
        
        .. note::
            
            Values of :math:`SSSIG > 1 \cdot 10^5` are indicative of sufficient subset size and contrast according to Stanier et al. (2016).
        """
        self.SSSIG = cpp.get_SSSIG(self.f_coords, self.f_img.QK_C_QKT)
        return
    
    def _get_sigma_intensity(self):
        r"""Private method to calculate the standard deviation of the reference subset intensities via a C++ extension.
        The standard deviation of the subset pixel intensities, :math:`\sigma_{s}`, after Stanier and White (2013), is calculated as follows:
            
        .. math::
            
            \sigma_{s}=\sqrt{\sum_{(x,y) \in n} \frac{1}{n} \left(f_{(x,y)}-f_{m}\right)^{2}}
        
        where :math:`f_{m}` is the mean subset pixel intensity calculated by :py:meth:`pycorr.Subset._get_f_m` and :math:`n` is the set of coordinates that comprise the subset.
        
        
        .. note::
            
            Values of :math:`\sigma_{s} > 15` are indicative of optimal seeding according to Stanier et al. (2016).
        """
        self.sigma_intensity = cpp.get_sigma_intensity(self.f, self.f_m)
        return
    
    def _get_g_coords(self):
        r"""Private method to calculate the target subset coordinates as a function of the reference coordinates and the warp function via the C++ extension :py:meth:`pycorr.cpp.get_g_coords`."""
        self.g_coords= cpp.get_g_coords(self.coord, self.p, self.f_coords, self.mode)
        return
    
    def _get_g(self):
        r"""Private method to calculate the subset intensities in the target image via the C++ extension :py:meth:`pycorr.cpp.get_g`.
        The image intensity at each coordinate of the reference subset, :math:`g_{(x, y)}`, is estimated using bi-quintic B-spline image intensity interpolation.
        First, the sub-pixel component of the position of each point in the subset is computed as follows from the current coordinates:
            
        .. math::
            
            \delta x = x - \left\lfloor x \right\rfloor \\
            \delta y = y - \left\lfloor y \right\rfloor
            
        where :math:`\left\lfloor x \right\rfloor` and :math:`\left\lfloor y \right\rfloor` are the floor of the coordinates of each point within the reference subset.
        The interpolated pixel intensity at the current sub-pixel coordinate, :math:`g_{(x, y)}`, is then calculated by performing the following operation:
            
        .. math::
            
            g_{(x, y)} = \begin{bmatrix} 1 & \delta y & \delta y^2 & \delta y^3 & \delta y^4 & \delta y^5 \end{bmatrix} \cdot
            \mathbf{QK} \cdot \mathbf{C}_{g \left(\left\lfloor x \right\rfloor-2:\left\lfloor x \right\rfloor+3, \left\lfloor y \right\rfloor-2:\left\lfloor y \right\rfloor+3\right)} \cdot \mathbf{QK^T} \cdot
            \begin{bmatrix} 1 \\ \delta x \\ \delta x^2 \\ \delta x^3 \\ \delta x^4 \\ \delta x^5 \end{bmatrix}
            
        where :math:`\mathbf{QK} \cdot \mathbf{C}_{g} \cdot \mathbf{QK^T}` was precomputed for the target image :math:`g` by :py:meth:`pycorr.Image._get_QK_C_QKT`.
        """
        self.g = cpp.get_intensity(self.g_coords, self.g_img.QK_C_QKT)
        return
    
    def _get_g_m(self):
        r"""
        
        Private method to calculate the mean target subset grayscale intensity via the C++ extension :py:meth:`pycorr.cpp.get_g_m`.
        
        The quantity :math:`g_{m}` is the mean target subset pixel intensity:
            
        .. math::
            
            g_{m} = \sum_{(x, y) \in n} \frac{g_{(x, y)}}{n} 
        
        where :math:`g_{(x, y)}` is the image intensity at a point :math:`(x, y)` within the subset,
        and :math:`n` is the set of coordinates that comprise the subset.
        
        """
        self.g_m = cpp.get_g_m(self.g)
        return
    
    def _get_Delta_g(self):
        r"""
        
        Private method to calculate the square root of the sum of the square of the variance from the mean of the target subset intensities via the C++ extension :py:meth:`pycorr.cpp.get_Delta_g`.
        
        The quantity :math:`\Delta g` is calculated as follows:
            
        .. math:: 
            
            \Delta g = \sqrt{ \sum_{(x, y) \in n} \left( g_{(x, y)} - g_{m} \right)^2}
            
        where :math:`g_{(x, y)}` is the image intensity at a point :math:`(x, y)` within the subset,
        :math:`g_{m}` is the mean subset intensity, and :math:`n` is the set of coordinates that comprise the subset.
        
        """
        self.Delta_g = cpp.get_Delta_g(self.g, self.g_m)
        return
    
    def _get_derivatives(self):
        r"""
        
        Private method to calculate the partial derivatives for the reference subset via the C++ extension :py:meth:`pycorr.cpp.get_derivatives`.
        The partial derivative of :math:`f` with respect to :math:`p` are calculated via bi-quintic B-spline image intensity interpolation for a first order warp function as follows:
        
        .. math::
            \frac{\partial f}{\partial p} = \begin{bmatrix} \frac{\partial f}{\partial u} & \frac{\partial f}{\partial v} & \frac{\partial f}{\partial u_{x}} & \frac{\partial f}{\partial u_{y}} & \frac{\partial f}{\partial v_{x}} & \frac{\partial f}{\partial v_{y}} \end{bmatrix}
        
        where:
            
        .. math::
            \frac{\partial f}{\partial u} = \nabla f_{x} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \end{bmatrix} \cdot
            \mathbf{QK} \cdot \mathbf{C}_{f \left(\left\lfloor x \right\rfloor-2:\left\lfloor x \right\rfloor+3, \left\lfloor y \right\rfloor-2:\left\lfloor y \right\rfloor+3\right)} \cdot \mathbf{QK^T} \cdot
            \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix} \\
        .. math::
            \frac{\partial f}{\partial v} = \nabla f_{y} = \begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix} \cdot
            \mathbf{QK} \cdot \mathbf{C}_{f \left(\left\lfloor x \right\rfloor-2:\left\lfloor x \right\rfloor+3, \left\lfloor y \right\rfloor-2:\left\lfloor y \right\rfloor+3\right)} \cdot \mathbf{QK^T} \cdot
            \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
        .. math::
            \frac{\partial f}{\partial u_{x}} = \nabla f_{x} \cdot \left( x - x_{0} \right) \\
        .. math::
            \frac{\partial f}{\partial u_{y}} = \nabla f_{x} \cdot \left( y - y_{0} \right) \\
        .. math::
            \frac{\partial f}{\partial v_{x}} = \nabla f_{y} \cdot \left( x - x_{0} \right) \\
        .. math::
            \frac{\partial f}{\partial v_{y}} = \nabla f_{y} \cdot \left( y - y_{0} \right)
        
        where :math:`\left\lfloor x \right\rfloor` and :math:`\left\lfloor y \right\rfloor` are the floor of the coordinates of each point within the reference subset,
        :math:`x` and :math:`y` are the coordinates of the current point within the reference subset, :math:`x_{0}` and :math:`y_{0}` are the coordinates of the reference subset,
        and :math:`\mathbf{QK} \cdot \mathbf{C}_{f} \cdot \mathbf{QK^T}` was precomputed for the reference image :math:`f` by :py:meth:`pycorr.Image._get_QK_C_QKT`.
            
        For a second order warp function the partial derivatives of :math:`f` with respect to :math:`p`:
        
        .. math::
            \frac{\partial f}{\partial p} = \begin{bmatrix} \frac{\partial f}{\partial u} & \frac{\partial f}{\partial v} & \frac{\partial f}{\partial u_{x}} & \frac{\partial f}{\partial u_{y}} & \frac{\partial f}{\partial v_{x}} & \frac{\partial f}{\partial v_{y}} & 
                                            \frac{\partial f}{\partial u_{xx}} & \frac{\partial f}{\partial u_{xy}} & \frac{\partial f}{\partial u_{yy}} & \frac{\partial f}{\partial v_{xx}} & \frac{\partial f}{\partial v_{xy}} & \frac{\partial f}{\partial v_{yy}}  \end{bmatrix}
            
        hence the following additional partial derivatives are also calculated:
            
        .. math::
            \frac{\partial f}{\partial u_{xx}} = \nabla f_{x} \cdot \left( x - x_{0} \right) \cdot \left( x - x_{0} \right) \\
        .. math::
            \frac{\partial f}{\partial u_{xy}} = \nabla f_{x} \cdot \left( x - x_{0} \right) \cdot \left( y - y_{0} \right) \\
        .. math::
            \frac{\partial f}{\partial u_{yy}} = \nabla f_{x} \cdot \left( y - y_{0} \right) \cdot \left( y - y_{0} \right) \\
        .. math::
            \frac{\partial f}{\partial v_{xx}} = \nabla f_{y} \cdot \left( x - x_{0} \right) \cdot \left( x - x_{0} \right) \\
        .. math::
            \frac{\partial f}{\partial v_{xy}} = \nabla f_{y} \cdot \left( x - x_{0} \right) \cdot \left( y - y_{0} \right) \\
        .. math::
            \frac{\partial f}{\partial v_{yy}} = \nabla f_{y} \cdot \left( y - y_{0} \right) \cdot \left( y - y_{0} \right) \\
        
        """
        self.derivatives = cpp.get_derivatives(self.coord, self.f_coords, self.f_img.QK_C_QKT, self.mode)
        return
    
    def _get_hessian(self):
        """Private method to calculate the Hessian matrix for the reference subset via a C++ extension."""
        self.hessian = cpp.get_hessian(self.derivatives)
        return
    
    def _get_Delta_p(self):
        """Private method to calculate the increment in the warp vector via a C++ extension."""
        self.Delta_p = cpp.get_Delta_p(self.hessian, self.f, self.g, self.f_m, self.g_m, self.Delta_f, self.Delta_g, self.derivatives)
        return
    
    def _get_p_new(self):
        """Private method to calculate the updated warp vector via a C++ extension."""
        self.p_new = cpp.get_p_new(self.p, self.Delta_p, self.mode)
        return
    
    def _get_norm(self):
        r"""
        
        Private method to calculate the norm of the increment in the warp vector via the C++ extension :py:meth:`pycorr.cpp.get_norm`.
        
        For a first order subset warp function:
            
        .. math::
            
            \|\Delta p\| = \sqrt{\Delta u^2 + \Delta v^2 + \left( \Delta u_{x}  s \right)^2 + \left( \Delta u_{y} s \right)^2 + \left( \Delta v_{x} s \right)^2 + \left( \Delta v_{y} s \right)^2}
            
        For a second order subset warp function:
            
        .. math::
            
            \|\Delta p\| = \sqrt{\Delta u^2 + \Delta v^2 + \left( \Delta u_{x} s \right)^2 + \left( \Delta u_{y} s \right)^2 + \left( \Delta v_{x} s \right)^2 + \left( \Delta v_{y}  s \right)^2
                         + \left( \Delta u_{xx} s^2 \right)^2 + \left( \Delta u_{xy} s^2 \right)^2 + \left( \Delta u_{yy} s^2 \right)^2
                         + \left( \Delta v_{xx} s^2 \right)^2 + \left( \Delta v_{xy} s^2 \right)^2 + \left( \Delta v_{yy} s^2 \right)^2}
        
        where :math:`s` is the size of the subset (typically taken as the radius if the subset is cirdular in shape).
        
        .. note::
            
            A typical exit criterion used in IC-GN computations is :math:`\|\Delta p\|_{max} = 1 \cdot 10^{-5}.`
            
        """
        self.norm = cpp.get_norm(self.Delta_p, np.max(self.template_coords[:,0]), self.mode)
        return
    
    def _get_correlation(self):
        r"""
        
        Private method to calculate the zero-normalised sum of squared differences and zero-normalised cross-correlation coefficients via the C++ extension :py:meth:`pycorr.cpp.get_znssd` and the analytical relation develop by Pan et al. (2010).
        
        The two quantities are computed as follows:
            
        .. math::
            
            C_{ZNSSD} = \sum_{(x, y) \in n} \left[ \frac{f_{(x, y)}-f_{m}}{\Delta f} - \frac{g_{(x, y)}-g_{m}}{\Delta g} \right]^2
        
        .. math::
            
            C_{ZNCC} = 1 - \left( \frac{C_{ZNSSD}}{2} \right)
        
        where :math:`f_{(x, y)}` and :math:`g_{(x, y)}` are the image intensities for a given point within the subset in the reference and target images computed by :py:meth:`pycorr.Subset._get_f` and :py:meth:`pycorr.Subset._get_g`, respectively.
        The quantities :math:`f_{m}` and :math:`g_{m}` are the mean intensities for the subset in the reference and target images computed by :py:meth:`pycorr.Subset._get_f_m` and :py:meth:`pycorr.Subset._get_g_m`, respectively.
        The quantities :math:`\Delta f` and :math:`\Delta g` are the square root of the sum of the square of the variance from the mean of the subset intensities computed by :py:meth:`pycorr.Subset._get_Delta_f` and :py:meth:`pycorr.Subset._get_Delta_g`, respectively,
        and :math:`n` is the set of coordinates that comprise the subset.
        """
        self.znssd = cpp.get_znssd(self.f, self.g, self.f_m, self.g_m, self.Delta_f, self.Delta_g)
        self.zncc = 1- (self.znssd/2)
        return
    
    
