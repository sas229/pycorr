#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Eigen>
#include <iostream>
#include <math.h>

// ----------------------
// C++ Utility Extensions
// ----------------------

using namespace Eigen;

MatrixXd get_QK_C_QKT(const Ref<const MatrixXd> &QK, const Ref<const MatrixXd> &QKt, const Ref<const MatrixXd> &C, const Ref<const Vector2d> &image_gs_shape, const int &border)
{
    // Define variables.
    int rows = image_gs_shape(0);
    int cols = image_gs_shape(1);
    MatrixXd QK_C_QKT(rows*6, cols*6);

    // Perform matrix multiplication to pre-compute QK_B_Kt.
    #pragma omp parallel for
    for(int j = 0; j < cols;  j++)
    {
        for(int i = 0; i < rows; i++)
        {
            int ind_row = i+border-2;
            int ind_col = j+border-2;
            QK_C_QKT.block(i*6,j*6,6,6) = QK*C.block(ind_row, ind_col, 6, 6)*QKt;
        }
    }
    return QK_C_QKT;
}

MatrixXd get_f_coords(const Ref<const VectorXd> &coord, const Ref<const MatrixXd> &template_coords)
{
    // Define variables.
    int n = template_coords.rows();
    MatrixXd f_coords(n, 2);

    // Compute the reference coordinates using the subset template.
    for(int i = 0; i < n;  i++)
    {
        f_coords(i,0) = template_coords(i,0) + coord(0);
        f_coords(i,1) = template_coords(i,1) + coord(1);
    }
    return f_coords;
}

MatrixXd get_g_coords(const Ref<const VectorXd> &coord, const Ref<const VectorXd> &p, const Ref<const MatrixXd> &f_coords, double mode)
{
    // Define variables.
    int n = f_coords.rows();
    double x = coord(0), y = coord(1);
    MatrixXd g_coords(n, 2);
    VectorXd x_r = f_coords.col(0), y_r = f_coords.col(1);

    // Compute the target coordinates using the reference coordinates and the warp vector.
    for(int i = 0; i < n;  i++)
    {
        // Check order of IC-GN solver required.
        if (mode == 1){
            double u = p(0), v = p(1), u_x = p(2), u_y = p(3), v_x = p(4), v_y = p(5);
            g_coords(i,0) = x_r(i) + u + u_x*(x_r(i)-x) + u_y*(y_r(i)-y);
            g_coords(i,1) = y_r(i) + v + v_x*(x_r(i)-x) + v_y*(y_r(i)-y);
        } else if (mode == 2){
            double u = p(0), v = p(1), u_x = p(2), u_y = p(3), v_x = p(4), v_y = p(5);
            double u_xx = p(6), u_xy = p(7), u_yy = p(8), v_xx = p(9), v_xy = p(10), v_yy = p(11);
            g_coords(i,0) = x_r(i) + u + u_x*(x_r(i)-x) + u_y*(y_r(i)-y) + 0.5*u_xx*(x_r(i)-x)*(x_r(i)-x) + u_xy*(x_r(i)-x)*(y_r(i)-y) + 0.5*u_yy*(y_r(i)-y)*(y_r(i)-y);
            g_coords(i,1) = y_r(i) + v + v_x*(x_r(i)-x) + v_y*(y_r(i)-y) + 0.5*v_xx*(x_r(i)-x)*(x_r(i)-x) + v_xy*(x_r(i)-x)*(y_r(i)-y) + 0.5*v_yy*(y_r(i)-y)*(y_r(i)-y);
        }
        
    }
    return g_coords;
}

VectorXd get_intensity(const Ref<const MatrixXd> &coords, const Ref<const MatrixXd> &QK_C_QKT)
{
    // Define variables.
    int n = coords.rows(), x_floor, y_floor;
    double delta_x, delta_y, one = 1.0;
    VectorXd intensity(n);
    typedef Matrix<double, 6, 1> Vector6d;
    Vector6d delta_x_vec(6), delta_y_vec(6);

    // Compute interpolated intensities.
    for(int i = 0; i < n;  i++)
    {
        x_floor = std::floor(coords(i,0));
        y_floor = std::floor(coords(i,1));
        delta_x = coords(i,0) - x_floor;
        delta_y = coords(i,1) - y_floor;
        delta_x_vec(0) = one;
        delta_y_vec(0) = one;
        for(int j = 1; j < 6;  j++) {
            delta_x_vec(j) = delta_x_vec(j - 1) * delta_x;
            delta_y_vec(j) = delta_y_vec(j - 1) * delta_y;
        }
        intensity(i) = (delta_y_vec.transpose()*QK_C_QKT.block(y_floor*6, x_floor*6, 6, 6))*delta_x_vec;
    }
    return intensity;
}

double get_SSSIG(const Ref<const MatrixXd> &coords, const Ref<const MatrixXd> &QK_C_QKT)
{

    // Define variables.
    int n = coords.rows(), x_floor, y_floor;
    double SSSIG = 0, dx, dy;

    // Compute SSSIG.
    for(int i = 0; i < n;  i++)
    {
        x_floor = std::floor(coords(i,0));
        y_floor = std::floor(coords(i,1));
        dx = QK_C_QKT((y_floor*6), (x_floor*6)+1);
        dy = QK_C_QKT((y_floor*6)+1, (x_floor*6));
        SSSIG += 0.5*(pow(dx, 2) + pow(dy, 2));
    }
    return SSSIG;
}

double get_sigma_intensity(const Ref<const VectorXd> &f, double &f_m)
{

    // Define variables.
    int n = f.rows();
    double sum_delta_f = 0, sigma;

    // Compute the standard deviation of the subset intensity.
    for(int i = 0; i < n;  i++)
    {
        sum_delta_f += pow((f(i)-f_m), 2);
    }
    sigma = pow((sum_delta_f/n), 0.5);
    return sigma;
}

double get_Delta_f(const Ref<const VectorXd> &f, double &f_m)
{
    // Define variables.
    int n = f.rows(), i;
    double sum_delta_f_sq = 0, Delta_f;

    // Compute the square root of the sum of delta squared.
    for(int i = 0; i < n;  i++)
    {
        sum_delta_f_sq += pow((f(i)-f_m), 2);
    }
    Delta_f = sqrt(sum_delta_f_sq);

    return Delta_f;
}

double get_Delta_g(const Ref<const VectorXd> &g, double &g_m)
{
    // Define variables.
    int n = g.rows(), i;
    double sum_delta_g_sq = 0, Delta_g;

    // Compute the square root of the sum of delta squared.
    for(int i = 0; i < n;  i++)
    {
        sum_delta_g_sq += pow((g(i)-g_m), 2);
    }
    Delta_g = sqrt(sum_delta_g_sq);

    return Delta_g;
}

double get_f_m(const Ref<const VectorXd> &f)
{
    return f.mean();
}

double get_g_m(const Ref<const VectorXd> &g)
{
    return g.mean();
}

double get_norm(const Ref<const VectorXd> &Delta_p, float size, int mode)
{
    //Define variables.
    double norm;
    
    if (mode == 1){
        norm = pow(((Delta_p(0)*Delta_p(0)) + (Delta_p(1)*Delta_p(1)) + ((Delta_p(2)*size)*(Delta_p(2)*size)) + ((Delta_p(3)*size)*(Delta_p(3)*size)) + ((Delta_p(4)*size)*(Delta_p(4)*size)) + ((Delta_p(5)*size)*(Delta_p(5)*size))), 0.5);
    } else if (mode == 2){
        norm = pow(((Delta_p(0)*Delta_p(0)) + (Delta_p(1)*Delta_p(1)) + ((Delta_p(2)*size)*(Delta_p(2)*size)) + ((Delta_p(3)*size)*(Delta_p(3)*size)) + ((Delta_p(4)*size)*(Delta_p(4)*size)) + ((Delta_p(5)*size)*(Delta_p(5)*size))
        + ((0.5*Delta_p(6)*size*size)*(0.5*Delta_p(6)*size*size)) + ((0.5*Delta_p(7)*size*size)*(0.5*Delta_p(7)*size*size)) +  ((0.5*Delta_p(8)*size*size)*(0.5*Delta_p(8)*size*size)) + ((0.5*Delta_p(9)*size*size)*(0.5*Delta_p(9)*size*size))
        + ((0.5*Delta_p(10)*size*size)*(0.5*Delta_p(10)*size*size)) + ((0.5*Delta_p(11)*size*size)*(0.5*Delta_p(11)*size*size))), 0.5);
    }
    
    return norm;
}

double get_znssd(const Ref<const VectorXd> &f, const Ref<const VectorXd> &g, double &f_m, double &g_m, double &Delta_f, double &Delta_g)
{
    // Define variables.
    int n = f.rows(), i;
    double znssd = 0;

    // Compute the square root of the sum of delta squared.
    for(int i = 0; i < n;  i++)
    {
        znssd += pow((((f(i)-f_m)/Delta_f)-((g(i)-g_m)/Delta_g)),2);
    }

    return znssd;
}

MatrixXd get_derivatives(const Ref<const VectorXd> &coord, const Ref<const MatrixXd> &coords, const Ref<const MatrixXd> &QK_C_QKT, int mode)
{
    // Check order of IC-GN solver required.
    int n;
    if (mode == 1){
    n = 6;
    } else if (mode == 2){
    n = 12;
    }
    
    // Define variables.
    int m = coords.rows(), x_floor, y_floor;
    double dx, dy;
    MatrixXd derivatives(m,n);

    // Compute steepest descent images.
    for(int i = 0; i < m;  i++)
    {
        dx = coords(i,0) - coord(0);
        dy = coords(i,1) - coord(1);
        x_floor = std::floor(coords(i,0));
        y_floor = std::floor(coords(i,1));
        derivatives(i,0) = QK_C_QKT((y_floor*6), (x_floor*6)+1);
        derivatives(i,1) = QK_C_QKT((y_floor*6)+1, (x_floor*6));
        derivatives(i,2) = derivatives(i,0)*dx;
        derivatives(i,3) = derivatives(i,0)*dy;
        derivatives(i,4) = derivatives(i,1)*dx;
        derivatives(i,5) = derivatives(i,1)*dy;
        if (mode ==2){
            derivatives(i,6) = derivatives(i,0)*dx*dx;
            derivatives(i,7) = derivatives(i,0)*dx*dy;
            derivatives(i,8) = derivatives(i,0)*dy*dy;
            derivatives(i,9) = derivatives(i,1)*dx*dx;
            derivatives(i,10) = derivatives(i,1)*dx*dy;
            derivatives(i,11) = derivatives(i,1)*dy*dy;
        }
    }
    return derivatives;
}

MatrixXd get_hessian(const Ref<const MatrixXd> &derivatives)
{
    // Define variables.
    int m = derivatives.cols(), n = derivatives.cols();
    VectorXd derivatives_i, derivatives_j, derivatives_dot_derivatives;
    MatrixXd hessian(m,n);

    // Compute steepest descent images.
    for(int i = 0; i < m;  i++)
    {
        for(int j = i; j < n;  j++)
        {
            derivatives_i = derivatives.col(i);
            derivatives_j = derivatives.col(j);
            derivatives_dot_derivatives = derivatives_i.transpose()*derivatives_j;
            hessian(i,j) = derivatives_dot_derivatives.sum();
            hessian(j,i) = hessian(i,j);
        }
    }

    return hessian;
}

VectorXd get_Delta_p(const Ref<const MatrixXd> &hessian, const Ref<const VectorXd> &f, const Ref<const VectorXd> &g, double &f_m, double &g_m, double &Delta_f, double &Delta_g, const Ref<const MatrixXd> &derivatives)
{
    //Define variables.
    int m = derivatives.cols(), n = f.rows();
    VectorXd invcompvec = VectorXd::Zero(m);
    MatrixXd inv_hessian;
    VectorXd Delta_p;
    
    // Compute inverse compositional vector.
    for(int j = 0; j < m;  j++)
    {
        for(int i = 0; i < n;  i++)
        {
        invcompvec(j) += derivatives(i,j)*((f(i)-f_m)-((Delta_f/Delta_g)*(g(i)-g_m)));
        }
    }

    // Invert hessian and calculate the warp vector increment.
    inv_hessian = hessian.inverse();
    Delta_p = -inv_hessian*invcompvec;

    return Delta_p;
}

VectorXd get_p_new(const Ref<const VectorXd> &p, const Ref<const VectorXd> &Delta_p, int mode)
{
    //Define variables.
    int m, n;
    if (mode == 1){
    m = 3;
    n = 3;
    } else if (mode == 2){
    m = 6;
    n = 6;
    }
    VectorXd p_new = VectorXd::Zero(2*m);
    MatrixXd w_old(m,n), w_delta(m,n), w_new(m,n), inv_w_delta(m,n);
    
    if (mode == 1){
        //Define variables.
        double u = p(0), v = p(1), u_x = p(2), u_y = p(3), v_x = p(4), v_y = p(5);
        double du = Delta_p(0), dv = Delta_p(1), du_x = Delta_p(2), du_y = Delta_p(3), dv_x = Delta_p(4), dv_y = Delta_p(5);
    
        // Calculate the new displacement matrix: p = [u, v, u_x, u_y, v_x, v_y].
        w_old << 1+u_x, u_y, u, v_x, 1+v_y, v, 0, 0, 1;
        w_delta << 1+du_x, du_y, du, dv_x, 1+dv_y, dv, 0, 0, 1;
        inv_w_delta = w_delta.inverse();
        w_new = w_old*inv_w_delta;
    
        // Allocate the new warp vector.
        p_new(0) = w_new(0,2);
        p_new(1) = w_new(1,2);
        p_new(2) = w_new(0,0)-1;
        p_new(3) = w_new(0,1);
        p_new(4) = w_new(1,0);
        p_new(5) = w_new(1,1)-1;
        
    } else if (mode == 2){
        //Define variables.
        double u = p(0), v = p(1), u_x = p(2), u_y = p(3), v_x = p(4), v_y = p(5);
        double u_xx = p(6), u_xy = p(7), u_yy = p(8), v_xx = p(9), v_xy = p(10), v_yy = p(11);
        double du = Delta_p(0), dv = Delta_p(1), du_x = Delta_p(2), du_y = Delta_p(3), dv_x = Delta_p(4), dv_y = Delta_p(5);
        double du_xx = Delta_p(6), du_xy = Delta_p(7), du_yy = Delta_p(8), dv_xx = Delta_p(9), dv_xy = Delta_p(10), dv_yy = Delta_p(11); 
        double S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18;
        double dS1, dS2, dS3, dS4, dS5, dS6, dS7, dS8, dS9, dS10, dS11, dS12, dS13, dS14, dS15, dS16, dS17, dS18;
    
        // Calculate the new displacement matrix: p = [u, v, u_x, u_y, v_x, v_y, u_xx, u_xy, u_yy, v_xx, v_xy, v_yy].
        S1 = (2*u_x) + (u_x*u_x) + (u*u_xx);
        S2 = (2*u*u_xy) + (2*(1+u_x)*u_y);
        S3 = (u_y*u_y) + (u*u_yy);
        S4 = 2*u*(1+u_x);
        S5 = 2*u*u_y;
        S6 = u*u;
        S7 = 0.5*(v*u_xx) + (2*(1+u_x)*v_x) + (u*v_xx);
        S8 = (u_y*v_x) + (u_x*v_y) + (v*u_xy) + (u*v_xy) + v_y + u_x;
        S9 = 0.5*((v*u_yy) + (2*u_y*(1+v_y)) + (u*v_yy));
        S10 = v + (v*u_x) + (u*v_x);
        S11 = u + (v*u_y) + (u*v_y);
        S12 = u*v;
        S13 = (v_x*v_x) + (v*v_xx);
        S14 = (2*v*v_xy) + (2*v_x*(1+v_y));
        S15 = (2*v_y) + (v_y*v_y) + (v*v_yy);
        S16 = 2*v*v_x;
        S17 = 2*v*(1+v_y);
        S18 = v*v;
        
        dS1 = (2*du_x) + (du_x*du_x) + (du*du_xx);
        dS2 = (2*du*du_xy) + (2*(1+du_x)*du_y);
        dS3 = (du_y*du_y) + (du*du_yy);
        dS4 = 2*du*(1+du_x);
        dS5 = 2*du*du_y;
        dS6 = du*du;
        dS7 = 0.5*(dv*du_xx) + (2*(1+du_x)*dv_x) + (du*dv_xx);
        dS8 = (du_y*dv_x) + (du_x*dv_y) + (dv*du_xy) + (du*dv_xy) + dv_y + du_x;
        dS9 = 0.5*((dv*du_yy) + (2*du_y*(1+dv_y)) + (du*dv_yy));
        dS10 = dv + (dv*du_x) + (du*dv_x);
        dS11 = du + (dv*du_y) + (du*dv_y);
        dS12 = du*dv;
        dS13 = (dv_x*dv_x) + (dv*dv_xx);
        dS14 = (2*dv*dv_xy) + (2*dv_x*(1+dv_y));
        dS15 = (2*dv_y) + (dv_y*dv_y) + (dv*dv_yy);
        dS16 = 2*dv*dv_x;
        dS17 = 2*dv*(1+dv_y);
        dS18 = dv*dv;
        
        w_old << 1+S1, S2, S3, S4, S5, S6, S7, 1+S8, S9, S10, S11, S12, S13, S14, 1+S15, S16, S17, S18, 0.5*u_xx, u_xy, 0.5*u_yy, 1+u_x, u_y, u, 0.5*v_xx, v_xy, 0.5*v_yy, v_x, 1+v_y, v, 0, 0, 0, 0, 0, 1;
        w_delta << 1+dS1, dS2, dS3, dS4, dS5, dS6, dS7, 1+dS8, dS9, dS10, dS11, dS12, dS13, dS14, 1+dS15, dS16, dS17, dS18, 0.5*du_xx, du_xy, 0.5*du_yy, 1+du_x, du_y, du, 0.5*dv_xx, dv_xy, 0.5*dv_yy, dv_x, 1+dv_y, dv, 0, 0, 0, 0, 0, 1;
        inv_w_delta = w_delta.inverse();
        w_new = w_old*inv_w_delta;

        // Allocate the new warp vector.
        p_new(0) = w_new(3,5);
        p_new(1) = w_new(4,5);
        p_new(2) = w_new(3,3)-1;
        p_new(3) = w_new(3,4);
        p_new(4) = w_new(4,3);
        p_new(5) = w_new(4,4)-1;
        p_new(6) = w_new(3,0)*2;
        p_new(7) = w_new(3,1);
        p_new(8) = w_new(3,2)*2;
        p_new(9) = w_new(4,0)*2;
        p_new(10) = w_new(4,1);
        p_new(11) = w_new(4,2)*2;
    }

    return p_new;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(pycorr_extensions,m)
{
m.doc() = "C++ extensions for pycorr.";
m.def("get_QK_C_QKT", &get_QK_C_QKT, py::return_value_policy::reference_internal, "C++ extension to pre-compute QK_C_QKT matrix.");
m.def("get_f_coords", &get_f_coords, py::return_value_policy::reference_internal, "C++ extension to compute the subset coordinates in the reference image.");
m.def("get_g_coords", &get_g_coords, py::return_value_policy::reference_internal, "C++ extension to compute the subset coordinates in the target image for the first order IC-GN method.");
m.def("get_intensity", &get_intensity, py::return_value_policy::reference_internal, "C++ extension to interpolate a vector of pixel intensities.");
m.def("get_SSSIG", &get_SSSIG, py::return_value_policy::reference_internal, "C++ extension to calculate approximate SSSIG from steepest descent image.");
m.def("get_sigma_intensity", &get_sigma_intensity, py::return_value_policy::reference_internal, "C++ extension to calculate the standard deviation of the subset pixel intensities.");
m.def("get_Delta_f", &get_Delta_f, py::return_value_policy::reference_internal, "C++ extension to compute the square root of the sum of delta squared for the reference image.");
m.def("get_Delta_g", &get_Delta_g, py::return_value_policy::reference_internal, "C++ extension to compute the square root of the sum of delta squared for the target image.");
m.def("get_f_m", &get_f_m, py::return_value_policy::reference_internal, "C++ extension to compute the mean of a vector of the reference subset intensities.");
m.def("get_g_m", &get_g_m, py::return_value_policy::reference_internal, "C++ extension to compute the mean of a vector of the target subset intensities.");
m.def("get_norm", &get_norm, py::return_value_policy::reference_internal, "C++ extension to compute the norm of the increment in the warp vector.");
m.def("get_znssd", &get_znssd, py::return_value_policy::reference_internal, "C++ extension to compute the zero-normalised sum of squared differences correlation coefficient.");
m.def("get_derivatives", &get_derivatives, py::return_value_policy::reference_internal, "C++ extension to compute the partial derivatives for the reference subset.");
m.def("get_hessian", &get_hessian, py::return_value_policy::reference_internal, "C++ extension to compute the hessian matrix.");
m.def("get_Delta_p", &get_Delta_p, py::return_value_policy::reference_internal, "C++ extension to compute the increment in the warp vector.");
m.def("get_p_new", &get_p_new, py::return_value_policy::reference_internal, "C++ extension to compute the new warp vector.");
}