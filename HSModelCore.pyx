#! python
# cython: boundscheck=False
# cython: wraparound=False, nonecheck=False
# cython: profile=True
import cython 
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, log, sqrt, log10, floor, abs 
from libc.time cimport time 
import scipy.special as cs #Need to bring this in because of cython_special missing API to gammaln 
cimport scipy.special.cython_special as cs

#from cython_gsl cimport *



cdef inline double my_min(double a, double b): return a if a <= b else b
cdef inline int sign(double x): return ((x > 0) - (x < 0))   
cdef inline double my_max(double a, double b): return a if a > b else b

#cdef extern from "gsl/gsl_sf_gamma.h": 
#    double gsl_sf_lngamma (double x) 
#cdef extern from "gsl/gsl_sf_psi.h":     
#    double gsl_sf_psi_n (int n, double x) 
    
################################################################################

cpdef poisson(double x, double mu): 
    """
    Args: 
        x: number
        mu: mean 
    Returns: 
        logP : log pmf
        cdf: cumulative dist. funct. 
    """
    
    cdef double logP, cdf
    cdef int i 
    logP = x*log(mu) - mu - log(cs.gamma(x+1))  
    cdf = 0 
    for i in range(<long>floor(x) + 1): 
        cdf += exp(i*log(mu) - mu - log(cs.gamma(<float>i+1)))  
    
    return logP, cdf 
    
################################################################################

cpdef negativeBinomial(double x, double mu, double r): 
    """
    NB on one element x, given mean mu and dispersion phi
    Args: 
        x: value tocompute at 
        mu: mean
        phi: dispersion parameter   
    Returns:
        logP:  log probability 
        cdf: cumulative dist. fucn. 
    """
    cdef double logP, cdf, p, log10_pvalue 
    p = mu**2/(mu**2 + r*mu)
    logP = r*log(r/(r+mu)) + cs.gammaln(r+x)   + x*log(mu/(r + mu)) - cs.gammaln(r) - cs.gammaln(x+1)  
    #because gamma(x) = (x-1)!
    cdf = 1 - cs.betainc(x+1, r, p)
    #log10_p_value = np.log10(ss.betainc(x+1, r, p)) #Read scipy.stats/betainc It is the REGULARIZED incomplte beta function
    # so no need to divide by beta(r,p)
    #cdf = 1 - (the regularized incomplete beta function)
    log10_pvalue = log10(my_min(1-cdf, cdf))  
    return logP, log10_pvalue 
#    
#    
    
################################################################################


cpdef compute_pval_sigma(double [:, ::1] TPMs, double [:] mu, double [:] r, double min_phi): 
    """
    My Negative Binomial pdf in terms of \phi and \mu, instead or p and r. Note that p = (sigma^2 - mu)/sigma^2 = \mu^2/(\mu^2 + r \mu)
    Args: 
        x: 
        mu: mean
        phi: dispersion
    Returns: 
        log10 prob: 
        log10 p-value:
        sigma_dev: deviation mwasured in sigma 
    """
    
    assert (len(TPMs) == len(mu))  and (len(TPMs) == len(r)), "Length mismatch in args"  
    cdef int L, S
    L, S = np.shape(TPMs) 
    cdef double [:,::1] logP = np.zeros((L, S)) 
    cdef double cdf, sigma, phi  
    cdef double [:,::1] log10_pvalue = np.zeros((L,S))   
    cdef double [:,::1] sigma_dev = np.zeros((L,S))  
    cdef int i, j  
      
    for i in range(L): 
        phi = 1.0/r[i] 
        if phi < min_phi: 
            for j in range(S): 
                logP[i,j], cdf = poisson(TPMs[i,j], mu[i]) 
                log10_pvalue[i,j] = log10(my_min(cdf, 1-cdf))
                sigma =  sqrt(mu[i] + phi*mu[i]**2)
                sigma_dev[i, j] = (TPMs[i, j] - mu[i])/sigma    
        else: 
            for j in range(S): 
                logP[i, j], log10_pvalue[i, j] = negativeBinomial(TPMs[i,j], mu[i], r[i]) 
                sigma =  sqrt(mu[i] + phi*mu[i]**2)
                sigma_dev[i,j] = (TPMs[i,j] - mu[i])/sigma  

    return np.asarray(logP), np.asarray(log10_pvalue), np.asarray(sigma_dev) 

    
################################################################################

      
cdef neg_loglikelihood(double r, double [:] Tvec):
    """
    negative log likehood to optimize over 
    Args: 
        r: 1/phi
        Tvec: vector of TPMs 
    Returns: 
        negative log likelihood (see HSModel notes) 
    """
    cdef int n, i 
    cdef double Tsum, neg_LL, first_term
    n = len(Tvec) 
    Tsum = 0 
    first_term = 0 
    for i in range(n): 
        first_term += cs.gammaln(Tvec[i] + r)
        #first_term += gsl_sf_lngamma(Tvec[i] + r)
        Tsum += Tvec[i]
    neg_LL = - (first_term  - n*cs.gammaln(r) - cs.gammaln(n*r + Tsum) + cs.gammaln(n*r))
    #neg_LL = - (first_term  - n*gsl_sf_lngamma(r) - gsl_sf_lngamma(n*r + Tsum) + gsl_sf_lngamma(n*r))
    return neg_LL  
                
###################################################################################                
cpdef der_NLL(double r, double [:] Tvec):  
    """
    The derivative of the likelihood function
    Args: 
       r: 1/phi
       Tvec: The vector of sample TPMs/counts 
    return: 
       derivate 
    """
    cdef int n, i
    cdef double Tsum, der_neg_LL, first_term 

    n = len(Tvec)
    first_term = 0
    for i in range(n): 
        first_term += cs.polygamma(0, Tvec[i] + r)
        #first_term += gsl_sf_psi_n(0, Tvec[i] + r) 
        Tsum += Tvec[i]
    
    der_neg_LL = -(first_term - n*cs.polygamma(0,r) - n*cs.polygamma(0,n*r + Tsum) + n*cs.polygamma(0, n*r)) 
    #der_neg_LL = -(first_term - n*gsl_sf_psi_n(0,r) - n*gsl_sf_psi_n(0,n*r + Tsum) + n*gsl_sf_psi_n(0, n*r)) 
    return der_neg_LL 

################################################################################

cpdef double_der_NLL(double r, double [:] Tvec): 
    """
    The double derivative needed for Newton's method 
    Args: 
       r: 1/phi
       Tvec: The vector of sample TPMs/counts 
    return: 
       double derivative 
    """
    cdef int n, i
    cdef double Tsum, der2_neg_LL, first_term 

    n = len(Tvec)
    first_term = 0
    for i in range(n): 
        first_term += cs.polygamma(1, Tvec[i] + r)
        #first_term += gsl_sf_psi_n(1, Tvec[i] + r) 
        Tsum += Tvec[i]
    
    der2_neg_LL = -(first_term - n*cs.polygamma(1,r) - (n**2)*cs.polygamma(1,n*r + Tsum) + (n**2)*cs.polygamma(1, n*r)) 
    #der2_neg_LL = -(first_term - n*gsl_sf_psi_n(1,r) - (n**2)*gsl_sf_psi_n(1,n*r + Tsum) + (n**2)*gsl_sf_psi_n(1, n*r)) 
    return der2_neg_LL   
                                      
###################################################################################                                                                                                                
                                                           
cpdef brent_bounded(double x1, double x2, double [:] Tvec, double xatol=1e-5, int maxiter=500): 
    """
    Args: 
        x1: lower bound 
        x2: upper bound 
    """
    
    cdef double golden, xm, tol1, tol2, ffulc, fnfc, r, p, q, fu  
    cdef int si 
    cdef int flag = 0
    cdef double sqrt_eps = sqrt(2.2e-16)
    cdef double golden_mean = 0.5 * (3.0 - sqrt(5.0))
    cdef double a = x1
    cdef double b = x2
    cdef double fulc = a + golden_mean * (b - a)
    cdef double nfc = fulc 
    cdef double xf = fulc
    cdef double rat = 0.0 
    cdef double e = 0.0
    cdef double x = xf
    cdef double fx = neg_loglikelihood(x, Tvec)
    cdef int num = 1
    cdef double fmin_data[3]
     
    fmin_data[:] = [1, xf, fx] 
   
    
    ffulc = fx
    fnfc = fx     
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xatol/3.0
    tol2 = 2.0 * tol1

    while (abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1.0
        # Check for parabolic fit
        if abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ( (abs(p) < abs(0.5*q*r)) and (p > q*(a - xf)) and  (p < q * (b - xf))):
                rat = p / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden section step
                golden = 1

        if golden:  # Do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e

        si = sign(rat) + (rat == 0)
        x = xf + si * my_max(abs(rat), tol1)
        fu = neg_loglikelihood(x, Tvec)
        num += 1
        fmin_data[:] = [num, x, fu] 
       
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxiter:
            flag = 1
            break
    # Return is: 
    # fucntional value, flag = 0 means solution found, else 1 maximum iteration reached, solution xf, num of interations  
    return fx, (flag == 0), xf, num 

                                                                                 
################################################################