from math import exp
from math import pi
from math import log
from scipy import integrate
import numpy as np

# Squared exponential kernel
def kernel(xi, xi_prime, sigma_f, ell):
    return (sigma_f**2)*exp(-0.5/(ell**2)*((xi-xi_prime)**2))


def integrand_L_im(x, delta_xi, sigma_f, ell):
    kernel_part = 0.0
    sqr_exp = exp(-0.5/(ell**2)*(x**2))
    a = delta_xi-x
    if a>0:
        kernel_part = exp(-a)/(1.+exp(-2*a))
    else:
        kernel_part = exp(a)/(1.+exp(2*a))
    return kernel_part*sqr_exp

def integrand_L2_im(x, xi, xi_prime, sigma_f, ell):
    f = exp(xi)
    f_prime = exp(xi_prime)
    numerator = exp(x-0.5/(ell**2)*(x**2))*(x+xi_prime-xi)
    if x<0:
        numerator = exp(x-0.5/(ell**2)*(x**2))*(x+xi_prime-xi)
        denominator = (-1+((f_prime/f)**2)*exp(2*x))
    else:
        numerator = exp(-x-0.5/(ell**2)*(x**2))*(x+xi_prime-xi)
        denominator = (-exp(-2*x)+((f_prime/f)**2))
    return numerator/denominator

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

def integrand_L_re(x, delta_xi, sigma_f, ell):
    kernel_part = 0.0
    sqr_exp = exp(-0.5/(ell**2)*(x**2))
    a = delta_xi-x
    if a>0:
        kernel_part = exp(-2*a)/(1.+exp(-2*a))
    else:
        kernel_part = 1./(1.+exp(2*a))
    return kernel_part*sqr_exp

def matrix_K(xi_n_vec, xi_m_vec, sigma_f, ell):

    N_n_freqs = xi_n_vec.size
    N_m_freqs = xi_m_vec.size

    K = np.zeros([N_n_freqs, N_m_freqs])

    for n in range(0, N_n_freqs):
        for m in range(0, N_m_freqs):
            K[n,m] = kernel(xi_n_vec[n], xi_m_vec[m], sigma_f, ell)

    return K

def matrix_L_im_K(xi_n_vec, xi_m_vec, sigma_f, ell):

    if np.array_equal(xi_n_vec, xi_m_vec):
        xi_vec = xi_n_vec
        N_freqs = xi_vec.size
#       vec_L_im_K = np.zeros(2*N_freqs-1)
        L_im_K = np.zeros([N_freqs, N_freqs])

        for n in range(0, N_freqs):
            
            delta_xi = xi_vec[n]-xi_vec[0] + log(2*pi)
            integral, tol = integrate.quad(integrand_L_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(delta_xi, sigma_f, ell))
            np.fill_diagonal(L_im_K[n:, :], (sigma_f**2)*(integral))
            
            delta_xi = xi_vec[0]-xi_vec[n] + log(2*pi)
            integral, tol = integrate.quad(integrand_L_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(delta_xi, sigma_f, ell))
            np.fill_diagonal(L_im_K[:, n:], (sigma_f**2)*(integral))
#        for p in range(0, 2*N_freqs-1):
#            if p<=N_freqs-1:
#                delta_xi = xi_vec[N_freqs-1-p]-xi_vec[0] + log(2*pi)
#            else:
#                delta_xi = xi_vec[0]-xi_vec[p-N_freqs+1] + log(2*pi)
#
#            integral, tol = integrate.quad(integrand_L_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(delta_xi, sigma_f, ell))
#            vec_L_im_K[p] =  (sigma_f**2)*(integral);
#
#        for n in range(0, N_freqs):
#            L_im_K[n,:] = vec_L_im_K[N_freqs-n-1 : 2*N_freqs-n-1]

    else:
        N_n_freqs = xi_n_vec.size
        N_m_freqs = xi_m_vec.size
        L_im_K = np.zeros([N_n_freqs, N_m_freqs])

        for n in range(0, N_n_freqs):
            for m in range(0, N_m_freqs):
                delta_xi = xi_n_vec[n]-xi_m_vec[m] + log(2*pi)
                integral, tol = integrate.quad(integrand_L_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(delta_xi, sigma_f, ell))
                L_im_K[n,m] =  (sigma_f**2)*(integral);

    return L_im_K


def matrix_L2_im_K(xi_n_vec, xi_m_vec, sigma_f, ell):

    if np.array_equal(xi_n_vec, xi_m_vec):
        xi_vec = xi_n_vec
        N_freqs = xi_vec.size
        L2_im_K = np.zeros([N_freqs, N_freqs])

        for n in range(0, N_freqs):
            xi = xi_vec[n]
            xi_prime = xi_vec[0]
            integral, tol = integrate.quad(integrand_L2_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(xi, xi_prime, sigma_f, ell))

            np.fill_diagonal(L2_im_K[n:, :], exp(xi_prime-xi)*(sigma_f**2)*integral)
            np.fill_diagonal(L2_im_K[:, n:], exp(xi_prime-xi)*(sigma_f**2)*integral)
            
    else:
        N_n_freqs = xi_n_vec.size
        N_m_freqs = xi_m_vec.size

        L2_im_K = np.zeros([N_n_freqs, N_m_freqs])

        for n in range(0, N_n_freqs):
            xi = xi_n_vec[n]
            for m in range(0, N_m_freqs):
                xi_prime = xi_m_vec[m]
                integral, tol = integrate.quad(integrand_L2_im, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(xi, xi_prime, sigma_f, ell))
                L2_im_K[n,m] = exp(xi_prime-xi)*(sigma_f**2)*integral

    return L2_im_K


def matrix_L_re_K(xi_vec, sigma_f, ell):

    N_freqs = xi_vec.size
    vec_L_re_K = np.zeros(2*N_freqs-1)
    L_re_K = np.zeros([N_freqs, N_freqs])

    for p in range(0, 2*N_freqs-1):
        if p<=N_freqs-1:
            delta_xi = xi_vec[N_freqs-1-p]-xi_vec[0] + log(2*pi)
        else:
            delta_xi = xi_vec[0]-xi_vec[p-N_freqs+1] + log(2*pi)

        integral, tol = integrate.quad(integrand_L_re, -np.inf, np.inf, epsabs=1E-12, epsrel=1E-12, args=(delta_xi, sigma_f, ell))
        vec_L_re_K[p] =  (sigma_f**2)*(integral);

    for p in range(0, N_freqs):

        L_re_K[p,:] = vec_L_re_K[N_freqs-p-1 : 2*N_freqs-p-1]

    return L_re_K

def NMLL_fct(theta, Z_exp, xi_vec):
    
    sigma_n = theta[0] 
    sigma_f = theta[1]  
    ell = theta[2]
    
    N_freqs = xi_vec.size
    Sigma = (sigma_n**2)*np.eye(N_freqs)
    
    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)
    K_im_full = L2_im_K + Sigma
    L = np.linalg.cholesky(K_im_full)
    
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)
    return 0.5*np.dot(Z_exp.imag,alpha) + np.sum(np.log(np.diag(L)))

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


def grad_NMLL_fct(theta, Z_exp, xi_vec):
    
    sigma_n = theta[0] 
    sigma_f = theta[1]  
    ell = theta[2]
    
    N_freqs = xi_vec.size
    Sigma = (sigma_n**2)*np.eye(N_freqs)
    
    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)
    K_im_full = L2_im_K + Sigma
    
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
    
    d_K_im_full_d_sigma_n = - 0.5*np.dot(alpha.T, np.dot(der_mat_sigma_n, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_sigma_n))    
        
    d_K_im_full_d_sigma_f = - 0.5*np.dot(alpha.T, np.dot(der_mat_sigma_f, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_sigma_f))
        
    d_K_im_full_d_ell     = - 0.5*np.dot(alpha.T, np.dot(der_mat_ell, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_ell))

    grad = np.array([d_K_im_full_d_sigma_n, d_K_im_full_d_sigma_f, d_K_im_full_d_ell])

    return grad


# 2D used for minimization of sigma_n and ell only

def NMLL2D_fct(theta, sigma_f, Z_exp, xi_vec):
    
    sigma_n = theta[0] 
    ell = theta[1]
    
    N_freqs = xi_vec.size
    Sigma = (sigma_n**2)*np.eye(N_freqs)
    
    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)
    K_im_full = L2_im_K + Sigma
    L = np.linalg.cholesky(K_im_full)
    
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)
    return 0.5*np.dot(Z_exp.imag,alpha) + np.sum(np.log(np.diag(L)))


def grad_NMLL2D_fct(theta, sigma_f, Z_exp, xi_vec):
    
    sigma_n = theta[0] 
    ell = theta[1]
    
    N_freqs = xi_vec.size
    Sigma = (sigma_n**2)*np.eye(N_freqs)
    
    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)
    K_im_full = L2_im_K + Sigma
    
    L = np.linalg.cholesky(K_im_full)
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)
    
    # compute inverse of K_im_full
    inv_L = np.linalg.inv(L)
    inv_K_im_full = np.dot(inv_L.T, inv_L)
    
    # derivative matrices
    der_mat_sigma_n = (2.*sigma_n)*np.eye(N_freqs)
    der_mat_ell = der_ell_matrix_L2_im_K(xi_vec, sigma_f, ell)
    
    d_K_im_full_d_sigma_n = - 0.5*np.dot(alpha.T, np.dot(der_mat_sigma_n, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_sigma_n))    
        
        
    d_K_im_full_d_ell     = - 0.5*np.dot(alpha.T, np.dot(der_mat_ell, alpha)) \
        + 0.5*np.trace(np.dot(inv_K_im_full, der_mat_ell))

    grad = np.array([d_K_im_full_d_sigma_n, d_K_im_full_d_ell])

    return grad