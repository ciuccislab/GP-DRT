"""
Created on Tue Dec 10 2019

@author: Jiapeng Liu, Francesco Ciucci (francesco.ciucci@ust.hk)
"""

######################################################################################
# This python file includes all necessary functions for GP-DRT model implemented in  #
# the paper "Liu, J., & Ciucci, F. (2019). The Gaussian process distribution of      #
# relaxation times: A machine learning tool for the analysis and prediction of       #
# electrochemical impedance spectroscopy data. Electrochimica Acta, 135316."         #
#                                                                                    #
# If you find it is useful, please feel free to use it, modify it, and develop based #
# on this code. Please cite the paper if you utlize this code for academic useage.   #
######################################################################################

# imports
from math import exp
from math import pi
from math import log
from scipy import integrate
import numpy as np
from numpy import linalg as la

# is a matrix positive definite?
# if input matrix is positive-definite (<=> Cholesky decomposable), then true is returned
# otherwise return false

def is_PD(A):

    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Find the nearest positive-definite matrix
def nearest_PD(A):
    
    # based on 
    # N.J. Higham (1988) https://doi.org/10.1016/0024-3795(88)90223-6
    # and 
    # https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    B = (A + A.T)/2
    _, Sigma_mat, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(Sigma_mat), V))

    A_nPD = (B + H) / 2
    A_symm = (A_nPD + A_nPD.T) / 2

    k = 1
    I = np.eye(A_symm.shape[0])

    while not is_PD(A_symm):
        eps = np.spacing(la.norm(A_symm))

        # MATLAB's 'chol' accepts matrices with eigenvalue = 0, numpy does not not. 
        # So where the matlab implementation uses 'eps(mineig)', we use the above definition.

        min_eig = min(0, np.min(np.real(np.linalg.eigvals(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1

    return A_symm

# Define squared exponential kernel, $\sigma_f^2 \exp\left(-\frac{1}{2 \ell^2}\left(\xi-\xi^\prime\right)^2 \right)$
def kernel(xi, xi_prime, sigma_f, ell):
    return (sigma_f**2)*exp(-0.5/(ell**2)*((xi-xi_prime)**2))


# the function to be integrated in eq (65) of the main text.
# $\frac{\displaystyle e^{\Delta\xi_{mn}-\chi}}{1+\left(\displaystyle e^{\Delta\xi_{mn}-\chi}\right)^2} \frac{k(\chi)}{\sigma_f^2}$
def integrand_L_im(x, delta_xi, sigma_f, ell):
    kernel_part = 0.0
    sqr_exp = exp(-0.5/(ell**2)*(x**2))
    a = delta_xi-x
    if a>0:
        kernel_part = exp(-a)/(1.+exp(-2*a))
    else:
        kernel_part = exp(a)/(1.+exp(2*a))
    return kernel_part*sqr_exp


# the function to be integrated in eq (76) of the main text.
# $\frac{1}{2} \left(\chi+\Delta\xi_{mn}\right){\rm csch}\left(\chi+\Delta\xi_{mn}\right) \frac{k(\chi)}{\sigma_f^2}$
def integrand_L2_im(x, xi, xi_prime, sigma_f, ell):
    delta_xi = xi_prime-xi
    
    y = x + delta_xi
    eps = 1E-8
    if y<-eps:
        factor_1 = exp(-0.5/(ell**2)*(x**2))
        factor_2 = y*exp(y)/(exp(2*y)-1.)
    elif y>eps:
        factor_1 = exp(-0.5/(ell**2)*(x**2))
        factor_2 = y*exp(-y)/(1.-exp(-2*y))
    else:
        factor_1 = exp(-0.5/(ell**2)*(x**2))
        factor_2 = 0.5
        
    out_val = factor_1*factor_2
    return out_val


# the derivative of the integrand in eq (76) with respect to \ell 
# $\frac{1}{2} \left(\chi+\Delta\xi_{mn}\right){\rm csch}\left(\chi+\Delta\xi_{mn}\right) \frac{k(\chi)}{\sigma_f^2}\chi^2$
# omitting the $\ell^3$ in the denominator
def integrand_der_ell_L2_im(x, xi, xi_prime, sigma_f, ell):
    f = exp(xi)
    f_prime = exp(xi_prime)
    if x<0:
        numerator = (x**2)*exp(x-0.5/(ell**2)*(x**2))*(x+xi_prime-xi)
        denominator = (-1+((f_prime/f)**2)*exp(2*x))
    else:
        numerator = (x**2)*exp(-x-0.5/(ell**2)*(x**2))*(x+xi_prime-xi)
        denominator = (-exp(-2*x)+((f_prime/f)**2))
    return numerator/denominator

# assemble the covariance matrix K as shown in eq (18a), which calculates the kernel distance between $\xi_n$ and $\xi_m$
def matrix_K(xi_n_vec, xi_m_vec, sigma_f, ell):
    N_n_freqs = xi_n_vec.size
    N_m_freqs = xi_m_vec.size
    K = np.zeros([N_n_freqs, N_m_freqs])

    for n in range(0, N_n_freqs):
        for m in range(0, N_m_freqs):
            K[n,m] = kernel(xi_n_vec[n], xi_m_vec[m], sigma_f, ell)
    return K


# assemble the matrix of eq (18b), added the term of $\frac{1}{\sigma_f^2}$ and factor $2\pi$ before $e^{\Delta\xi_{mn}-\chi}$
def matrix_L_im_K(xi_n_vec, xi_m_vec, sigma_f, ell):

    if np.array_equal(xi_n_vec, xi_m_vec):
        # considering the case that $\xi_n$ and $\xi_m$ are identical, i.e., the matrix is square symmetrix
        xi_vec = xi_n_vec
        N_freqs = xi_vec.size
        L_im_K = np.zeros([N_freqs, N_freqs])

        delta_xi = log(2*pi)
        integral, tol = integrate.quad(integrand_L_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(delta_xi, sigma_f, ell))
        np.fill_diagonal(L_im_K, -(sigma_f**2)*(integral))

        for n in range(1, N_freqs):
            delta_xi = xi_vec[n]-xi_vec[0] + log(2*pi)
            integral, tol = integrate.quad(integrand_L_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(delta_xi, sigma_f, ell))
            np.fill_diagonal(L_im_K[:, n:], -(sigma_f**2)*(integral))
            
            delta_xi = xi_vec[0]-xi_vec[n] + log(2*pi)
            integral, tol = integrate.quad(integrand_L_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(delta_xi, sigma_f, ell))
            np.fill_diagonal(L_im_K[n:, :], -(sigma_f**2)*(integral))
    else:
        N_n_freqs = xi_n_vec.size
        N_m_freqs = xi_m_vec.size
        L_im_K = np.zeros([N_n_freqs, N_m_freqs])

        for n in range(0, N_n_freqs):
            for m in range(0, N_m_freqs):

                delta_xi = xi_m_vec[m] - xi_n_vec[n] + log(2*pi)
                integral, tol = integrate.quad(integrand_L_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(delta_xi, sigma_f, ell))
                L_im_K[n,m] = -(sigma_f**2)*(integral)

    return L_im_K


# assemble the matrix of eq (18d), added the term of $\frac{1}{\sigma_f^2}$ and factor $2\pi$ before $e^{\Delta\xi_{mn}-\chi}$
def matrix_L2_im_K(xi_n_vec, xi_m_vec, sigma_f, ell):

    if np.array_equal(xi_n_vec, xi_m_vec):
        # considering the case that $\xi_n$ and $\xi_m$ are identical, i.e., the matrix is square symmetric
        xi_vec = xi_n_vec
        N_freqs = xi_vec.size
        L2_im_K = np.zeros([N_freqs, N_freqs])
        
        for n in range(0, N_freqs):
            xi = xi_vec[n]
            xi_prime = xi_vec[0]
            integral, tol = integrate.quad(integrand_L2_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(xi, xi_prime, sigma_f, ell))
            if n == 0:
                np.fill_diagonal(L2_im_K, (sigma_f**2)*integral)
            else:
                np.fill_diagonal(L2_im_K[n:, :], (sigma_f**2)*integral)
                np.fill_diagonal(L2_im_K[:, n:], (sigma_f**2)*integral)
    else:
        N_n_freqs = xi_n_vec.size
        N_m_freqs = xi_m_vec.size
        L2_im_K = np.zeros([N_n_freqs, N_m_freqs])

        for n in range(0, N_n_freqs):
            xi = xi_n_vec[n]
            for m in range(0, N_m_freqs):
                xi_prime = xi_m_vec[m]
                integral, tol = integrate.quad(integrand_L2_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(xi, xi_prime, sigma_f, ell))
                L2_im_K[n,m] = (sigma_f**2)*integral
    return L2_im_K

def compute_h_L(xi):

    h_L = 2.*pi*np.exp(xi)
    h_L = h_L.reshape(xi.size, 1)

    return h_L


# calculate the negative marginal log-likelihood (NMLL) of eq (31)
def NMLL_fct(theta, Z_exp, xi_vec):
    # load the initial value for parameters needed to optimize
    sigma_n = theta[0]
    sigma_f = theta[1]
    ell = theta[2]
    
    N_freqs = xi_vec.size
    Sigma = (sigma_n**2)*np.eye(N_freqs)                    # $\sigma_n^2 \mathbf I$
    
    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)  # $\mathcal L^2_{\rm im} \mathbf K$
    K_im_full = L2_im_K + Sigma                             # $\mathbf K_{\rm im}^{\rm full} = \mathcal L^2_{\rm im} \mathbf K + \sigma_n^2 \mathbf I$

    # begin FC - added 
    if not is_PD(K_im_full):
        K_im_full = nearest_PD(K_im_full)
    # end FC - added 

    L = np.linalg.cholesky(K_im_full)                       # Cholesky decomposition to get the inverse of $\mathbf K_{\rm im}^{\rm full}$
    
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)                     # $(\mathcal L^2_{\rm im} \mathbf K + \sigma_n^2 \mathbf I)^{-1} \mathbf Z^{\rm exp}_{\rm im}$
     
    # return the final result of $L(\bm \theta)$, i.e., eq (32). Note that $\frac{N}{2} \log 2\pi$ 
    # is not included as it is constant. The determinant of $\mathbf K_{\rm im}^{\rm full}$ is 
    # easily calculated as the product of the diagonal element of L
    return 0.5*np.dot(Z_exp.imag,alpha) + np.sum(np.log(np.diag(L)))

def NMLL_fct_L(theta, Z_exp, xi_vec):
    
    sigma_n = theta[0] 
    sigma_f = theta[1]  
    ell = theta[2]
    sigma_L = theta[3]
    
    N_freqs = xi_vec.size

    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)
    Sigma = (sigma_n**2)*np.eye(N_freqs)
    h_L = compute_h_L(xi_vec)
    
    K_im_full = L2_im_K + Sigma + (sigma_L**2)*np.outer(h_L, h_L)
    K_im_full = 0.5*(K_im_full+K_im_full.T)

    # begin FC - added 
    if not is_PD(K_im_full):
        K_im_full = nearest_PD(K_im_full)
    # end FC - added 

    L = np.linalg.cholesky(K_im_full)
    
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)
    return 0.5*np.dot(Z_exp.imag,alpha) + np.sum(np.log(np.diag(L)))

# assemble the matrix corresponding to the derivative of eq (18d) with respect to $\ell$, similar to the above implementation 
def der_ell_matrix_L2_im_K(xi_vec, sigma_f, ell):
    N_freqs = xi_vec.size
    der_ell_L2_im_K = np.zeros([N_freqs, N_freqs])

    for n in range(0, N_freqs):
        xi = xi_vec[n]
        xi_prime = xi_vec[0]
        integral, tol = integrate.quad(integrand_der_ell_L2_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(xi, xi_prime, sigma_f, ell))
        np.fill_diagonal(der_ell_L2_im_K[n:, :], exp(xi_prime-xi)*(sigma_f**2)/(ell**3)*integral)
        np.fill_diagonal(der_ell_L2_im_K[:, n:], exp(xi_prime-xi)*(sigma_f**2)/(ell**3)*integral)

    return der_ell_L2_im_K

# gradient of the negative marginal log-likelihhod (NMLL) $L(\bm \theta)$
def grad_NMLL_fct(theta, Z_exp, xi_vec):
    # load the initial value for parameters needed to optimize
    sigma_n = theta[0] 
    sigma_f = theta[1]  
    ell     = theta[2]
    
    N_freqs = xi_vec.size
    Sigma = (sigma_n**2)*np.eye(N_freqs)                    # $\sigma_n^2 \mathbf I$
    
    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)  # $\mathcal L^2_{\rm im} \mathbf K$
    K_im_full = L2_im_K + Sigma                             # $\mathbf K_{\rm im}^{\rm full} = \mathcal L^2_{\rm im} \mathbf K + \sigma_n^2 \mathbf I$
    
    # begin FC - added 
    if not is_PD(K_im_full):
        K_im_full = nearest_PD(K_im_full)
    # end FC - added 
    
    L = np.linalg.cholesky(K_im_full)                       # Cholesky decomposition to get the inverse of $\mathbf K_{\rm im}^{\rm full}$

    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)
    
    # compute inverse of K_im_full
    inv_L = np.linalg.inv(L)
    inv_K_im_full = np.dot(inv_L.T, inv_L)
    
    # calculate the derivative of matrices with respect to parameters including $sigma_n$, $sigma_f$, and $\ell$
    der_mat_sigma_n = (2.*sigma_n)*np.eye(N_freqs)              # derivative of $\sigma_n^2 \mathbf I$ to $sigma_n$
    der_mat_sigma_f = (2./sigma_f)*L2_im_K                      # note that the derivative is 2*sigma_f*L2_im_K/(sigma_f**2)
    der_mat_ell = der_ell_matrix_L2_im_K(xi_vec, sigma_f, ell)  # derivative w.r.t $\ell$
    
    # calculate the derivative of $L(\bm \theta)$ w.r.t $sigma_n$, $sigma_f$, and $\ell$ according to eq (78)
    d_K_im_full_d_sigma_n = - 0.5*np.dot(alpha.T, np.dot(der_mat_sigma_n, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_sigma_n))    
        
    d_K_im_full_d_sigma_f = - 0.5*np.dot(alpha.T, np.dot(der_mat_sigma_f, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_sigma_f))
        
    d_K_im_full_d_ell     = - 0.5*np.dot(alpha.T, np.dot(der_mat_ell, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_ell))

    grad = np.array([d_K_im_full_d_sigma_n, d_K_im_full_d_sigma_f, d_K_im_full_d_ell])
    return grad

def grad_NMLL_fct_L(theta, Z_exp, xi_vec):
    
    sigma_n = theta[0] 
    sigma_f = theta[1]  
    ell     = theta[2]
    sigma_L = theta[3]
    
    N_freqs = xi_vec.size
    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)
    Sigma = (sigma_n**2)*np.eye(N_freqs)
    h_L = compute_h_L(xi_vec)
    
    K_im_full = L2_im_K + Sigma + (sigma_L**2)*np.outer(h_L, h_L)
    
    # begin FC - added 
    if not is_PD(K_im_full):
        K_im_full = nearest_PD(K_im_full)
    # end FC - added 
    L = np.linalg.cholesky(K_im_full)

    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)
    
    # compute inverse of K_im_full
    inv_L = np.linalg.inv(L)
    inv_K_im_full = np.dot(inv_L.T, inv_L)
    
    # derivative matrices
    der_mat_sigma_n = (2.*sigma_n)*np.eye(N_freqs)
    der_mat_sigma_f = (2./sigma_f)*L2_im_K
    der_mat_ell = der_ell_matrix_L2_im_K(xi_vec, sigma_f, ell)
    der_mat_sigma_L = (2.*sigma_L)*np.outer(h_L, h_L)
    
    d_K_im_full_d_sigma_n = - 0.5*np.dot(alpha.T, np.dot(der_mat_sigma_n, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_sigma_n))    
        
    d_K_im_full_d_sigma_f = - 0.5*np.dot(alpha.T, np.dot(der_mat_sigma_f, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_sigma_f))
        
    d_K_im_full_d_ell     = - 0.5*np.dot(alpha.T, np.dot(der_mat_ell, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_ell))
        
    d_K_im_full_d_sigma_L = - 0.5*np.dot(alpha.T, np.dot(der_mat_sigma_L, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_sigma_L))   

    grad = np.array([d_K_im_full_d_sigma_n, d_K_im_full_d_sigma_f, d_K_im_full_d_ell, d_K_im_full_d_sigma_L])

    return grad