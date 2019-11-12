"""
The HS Model @Swagatam Mukhopadhyay 
""" 
__author__ = "Swagatam Mukhopadhyay"
__copyright__ = "Ionis"
__credits__ = []
__license__ = "MIT"
__version__ = "3.0.0"
__maintainer__ = "Swagatam Mukhopadhyay"
__email__ = "smukhopadhyay@ionisph.com"
__status__ = "Release: Multiprocessing and Cythonized code added"

# Initial imports 
## STD
import os, sys, string 
import warnings 
## CONTRIB
import pandas as pd
import numpy  as np
import itertools
## Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display
## Storage
import cPickle as pickle
## LOCAL

#SPECIAL LIBS
from sklearn.decomposition import PCA, FastICA
from itertools import cycle
from scipy.stats import norm
import logging 
from scipy.optimize import minimize, least_squares 
from scipy.optimize import minimize_scalar  

import scipy.special as ss
from scipy.stats import poisson 
from scipy.stats import gamma
from scipy.stats import rv_discrete

from joblib import Parallel, delayed  
from multiprocessing import cpu_count, Pool 
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.stats import poisson 

from ngstools import HSModelCore 
import time 

EPSILON = 0.001 #regularizer for fold change
R_BOUNDS =  (1e-03, 1e03) #bounds on r = 1/phi in the brent method for finding max. likelihood solution 
UNIT_BOUNDS = (1e-06, 10) #bounds on unit s 
OPTIONS = {'maxiter':10000, 'xatol': 1e-05} #other max. likelihood options 


# Utility functions
################################################################################

def non_increasing(L):
    """
    Detect if list L is non-increasing 
    """
    return np.all([x>=y for x, y in zip(L, L[1:])])
################################################################################

def non_decreasing(L):
    """
    Detect if list L is non-decreasing  
    """
    return np.all([x<=y for x, y in zip(L, L[1:])])

################################################################################

def myround(L, base=2):
    """
    Round numpy array ordataframe to nearest multiple of base 
    """
    if type(L) is list: 
        L = np.asarray(L)
        
    return (base * np.round(L.astype(float)/base))
    
    
################################################################################
def poissonFit(means, variances): 
    """
    Does a Poisson fit to the data: log varinace vs log means: \sigma^2 = s*m --> log \sigma_2 := log_var = log_mean + s   
    in log10 space 
    Args: 
        means: 
        variance: 
    
    """
    Y = np.log(np.asarray(variances)) - np.log(np.asarray(means)) 
    s = np.mean(Y)
    return np.exp(s) 
    
##################################################################################

def negativeBinomial(x, mu, phi): 
    """
    My Negative Binomial pdf in terms of \phi and \mu, instead or p and r. Note that p = sigma^2 - mu)/sigma^2 = \mu^2/(\mu^2 + rm)
    Args: 
        x: 
        mu: mean
        phi: dispersion
    Returns: 
        log10 prob: NB value at x
        log10 p-value: 1 - cdf(x, mu, phi)
    """
    x = np.round(x)
    phi_precision = 1e-06 #total overkill, actual precison for 10 samples is 0.01, but here this is used just as a switch
    # to use poission computation 
    if phi < phi_precision: 
        logP = np.log(poisson.pmf(x, mu)) 
        cdf = poisson.cdf(x, mu) 
    else: 
        r = 1.0/phi
        p = mu**2/(mu**2 + r*mu)
        logP = r*np.log(r/(r+mu)) + ss.gammaln(r+x) + x*np.log(mu/(r + mu)) - ss.gammaln(r) - ss.gammaln(x+1) 
        #because gamma(x) = (x-1)!
        cdf = 1 - ss.betainc(x+1, r, p)
    
    log10_p_value = np.log10(np.minimum(1-cdf, cdf))
    #log10_p_value = np.log10(ss.betainc(x+1, r, p)) #Read scipy.stats/betainc It is the REGULARIZED incomplte beta function
    # so no need to divide by beta(r,p)
    #cdf = 1 - (the regularized incomplete beta function)
    #p_value = 1 - cdf = the regularized incomplte beta function
    return logP, log10_p_value
    
    
################################################################################

def neg_loglikelihood(r, Tvec): 
    """
    The likelihood function of r computation (r = 1/phi) 
    Args: 
       r: 1/phi
       Tvec: The vector of sample TPMs/counts 
    return: 
       negative log likelihood (we minimize this)
    """
    n = len(Tvec)
    Z = np.sum(Tvec)
    first_term = [ss.gammaln(t + r) for t in Tvec]
    neg_log_L = -( np.sum(first_term) - n*ss.gammaln(r) - ss.gammaln(n*r + Z) + ss.gammaln(n*r)) 
    return neg_log_L
    
################################################################################

def der_NLL(r, Tvec):  
    """
    The derivative of the likelihood function
    Args: 
       r: 1/phi
       Tvec: The vector of sample TPMs/counts 
    return: 
       derivate 
    """
    n = len(Tvec)
    Z = np.sum(Tvec)
    der_first_term = [ss.polygamma(0,t + r) for t in Tvec]
    der_neg_log_L = -( np.sum(der_first_term) - n*ss.polygamma(0,r) - n*ss.polygamma(0,n*r + Z) + n*ss.polygamma(0, n*r)) 
    return der_neg_log_L 

################################################################################

def double_der_NLL(r, Tvec): 
    """
    The double derivative needed for Newton's method 
    Args: 
       r: 1/phi
       Tvec: The vector of sample TPMs/counts 
    return: 
       double derivative 
    """
    n = len(Tvec)
    Z = np.sum(Tvec)
    der2_first_term = [ss.polygamma(1,t + r) for t in Tvec]
    der2_neg_log_L = -( np.sum(der2_first_term) - n*ss.polygamma(1,r) - (n**2)*ss.polygamma(1,n*r + Z) + (n**2)*ss.polygamma(1, n*r)) 
    return der2_neg_log_L 

################################################################################
def maximize_LL(Tvec, method = 'cython'): 
    """
    main computation, tried both SLSQP and L_BFGS_B, but the best performance is with minimize_scalar, it is insensitive
    to initial values, Brent's method implemented in cython 
    Args: 
       Tvec: The vector of sample TPMs/counts 
       init_r: initial 1/phi
    """
    # options={'ftol': 1e-01, 'maxiter': 5000} #reduced the f tolerance to 0.1 because, typically the log likehoods are large 
    # The function is convex, use SLSQP 
    # recall that the bounds are in r, so the range of phi is 10**(-6) to 1000
    # R = minimize(neg_loglikelihood, init_r, method='SLSQP', jac = der_NLL, bounds= ((1e-03, 1e06),), args = (TPMs, ), options=options)
    # x,f,d = scipy.optimize.fmin_l_bfgs_b(neg_loglikelihood, init_r, fprime = der_NLL, bounds= ((1e-03, 1e06),), args = (TPMs, ), 
    #                                    pgtol=1e-07, approx_grad = 1)
    if method == 'cython': 
        Res = HSModelCore.brent_bounded(R_BOUNDS[0], R_BOUNDS[1], np.asarray(Tvec).astype(float), maxiter = OPTIONS['maxiter'])
        # Res is (Func. Value, flag (success), x value (solution), num of interations 
        R = dict()
        R['success'] = Res[1]
        R['fun'] = Res[0]
        R['nfev'] = Res[3]
        R['x'] = Res[2]
        R['FI_MLE'] = HSModelCore.double_der_NLL(R['x'], np.asarray(Tvec).astype(float))
    elif method == 'python':     
        R = minimize_scalar(neg_loglikelihood, bracket=None, bounds=  R_BOUNDS, args = (Tvec, ), 
                         method='bounded', options= OPTIONS) 
        R = dict(R)
        R['FI_MLE'] = double_der_NLL(R['x'], Tvec)
    else: 
        raise ValueError('Unknown method option') 
    if not R['success']: 
        warnings.warn("Dispersion optimization failed for " + g) 
        
    return R

###############################################################################

def _maximize_LL_helper(args): 
    """
    helper fucntion for parallel
    Args:    
        same as maximize_LL 
    Returns: 
        [1.0/phi_naive.loc[g], ans[i]['x'], ans[i]['success'], num_over_unit, ans[i]['FI_MLE']]
    """
    phi_naive, Tvec, method = args[0], args[1], args[2] 
    R = maximize_LL(Tvec, method = method)
    ans_vec = [1.0/phi_naive, R['x'], R['success'], R['FI_MLE']]  
    return ans_vec 
    
################################################################################

def maximize_LL_bootstrap(Tvec, method = 'cython', BS_sample_size = 10, num_bootstrap_samples = 10):
    """
    bootstrapped version, given a Tvec computes the empirical Fisher Information 
    and MLE bootstraps
    Args: 
        r_star: maximum likelihood r 
        Tvec: The array of TPMs 
        BS_sample_size: Number of bootstrap samples 
    Returns: 
        dictionary res: 
            MLE: MLE value of r  
            FI_MLE: Fisher information for the MLE estimate 
            FI: Fisher Information averaged over bootstrapping samples  

    """
    R = maximize_LL(Tvec, method = method)
    
    MLE = R['x']
    FI_MLE = R['FI_MLE'] # the fisher info at the MLE estimate 
    MLE_ests = [MLE]  #starts off with the original sample 
    FI= [FI_MLE] 
    BS_means = [] 
    for i in np.arange(num_bootstrap_samples):
        T = np.random.choice(Tvec, size=BS_sample_size, replace= True) 
        R = maximize_LL(T, method = method)
        BS_means.append(np.mean(T))
        if R['success']: 
            MLE_ests.append(R['x']) 
            FI.append(R['FI_MLE'])    
   

    # In order to compute the error of log ratios I need the bootstrap variance of log(BS_means)
    BS_means = np.asarray(BS_means)
    res = dict()
    res['MLE'] = MLE
    res['FI_MLE'] = FI_MLE
    res['mean_MLE'] = np.mean(MLE_ests)
    res['median_MLE'] = np.median(MLE_ests)
    res['std_MLE'] = np.std(MLE_ests)
    res['mean_FI'] = np.mean(FI)
    res['BS_means'] = BS_means 
    res['success'] = len(FI) #the  number of successes 
    return res 

#############################################################################

def _maximize_LL_bootstrap_helper(args): 
    """
    Helper function for parallel job, returns vector instead of dicitonary to easily convert to dataframe 
    Args: 
        same as maximize_LL_bootstrap
    Returns:  columns=['r_naive', 'r_fancy', 'BS_converge', 'median_r_BS', 'mean_r_BS', 'std_r_BS', 
                                     'uncertainty_r', 'FI_MLE', 'BS_mu', 'std_BS_mu']
                vector with the following entries 
                self.results.loc[g] = [1.0/phi_naive.loc[g], ans['MLE'], 
                                       ans['success'], ans['median_MLE'], 
                                       ans['mean_MLE'], ans['std_MLE'], np.sqrt(1.0/ans['mean_FI']), 
                                       ans['FI_MLE'], np.mean(ans['BS_means']), np.std(ans['BS_means'])]
            
    """
    
    phi_naive, Tvec, method, BS_sample_size, num_bootstrap_samples = args[0], args[1], args[2], args[3], args[4] 
    ans = maximize_LL_bootstrap(Tvec, method = method,  BS_sample_size = BS_sample_size, \
                                num_bootstrap_samples = num_bootstrap_samples)
    ans_vec = [1.0/phi_naive, ans['MLE'], ans['success'], ans['median_MLE'], \
                                           ans['mean_MLE'], ans['std_MLE'], np.sqrt(1.0/ans['mean_FI']), \
                                           ans['FI_MLE'], np.mean(ans['BS_means']), np.std(ans['BS_means'])] 
                                       
    return ans_vec 
################################################################################


def testMaximization(r, p, sample_size, repeats, method = 'cython', plot_range = [10, 10000], step_size = 0.1, resample = False, BS_sample_size = 10, num_bootstrap_samples = 10):
    """
    This is a test function to check convergence 
    Args: 
        r: paramter of NB
        p: prob paramter of NB
        sample_size: How many number to draw? 
        repeats: how may times to repeat such sampling 
        plot_range: of r to see the likelihood
        step_size: compute the likelihood in plot_range at this steps 
    returns: 
        naive_r: naive computation of r  (= 1/phi) 
        fancy_phi: max. likelihood computation of r 
    """
    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(1,1,1)
    rs = np.arange(plot_range[0], plot_range[1], step_size)
    
    naive_r= np.zeros(repeats)
    fancy_r = np.zeros(repeats)
    #print "------------------------"
    #print "True value of r: ", r
    #print 
    
    if resample: 
        Tfixed  = np.random.negative_binomial(r, p, size=[sample_size])
        #T is fixed, we will resample from T 
        
    for i in np.arange(repeats):
        
        if resample: 
            T = np.random.choice(Tfixed, size=BS_sample_size, replace=True) 
            # print 'T is', T 
            # print 'Tfixed is', Tfixed 
        else:
            T = np.random.negative_binomial(r, p, size=[sample_size])  # create new draws
            
        naive_phi = (np.std(T)**2 - np.mean(T))/(np.mean(T))**2
        #print "naive estmate of r:", 1.0/naive_phi 
        R = maximize_LL(T.astype(float), method = method)
        #print "Fancy estimate of r: ",  R['x']

        
        n = len(T)
        Z = np.sum(T)
        Y = np.zeros_like(rs)

        for j, k in enumerate(rs): 
            Y[j] =  neg_loglikelihood(k, T)

        ax.plot(rs, Y/np.max(Y))
        ax.plot([1.0/naive_phi], [neg_loglikelihood(1.0/naive_phi, T)/np.max(Y)], marker = 'o', ms = 10, linestyle = 'none', alpha = 0.5)
        ax.plot([R['x']], [neg_loglikelihood(R['x'], T)/np.max(Y)], marker = 's', ms = 10, linestyle = 'none', alpha = 0.5)
        
        naive_r[i] = 1.0/naive_phi
        fancy_r[i] = R['x']
    
                    
    if resample: 
        
        R = maximize_LL(Tfixed.astype(float), method = method)
        r_star = R['x']
        #FI_MLE, FI = FisherInfo(r_star, Tfixed)
        res = maximize_LL_bootstrap(Tfixed, method = method, BS_sample_size = BS_sample_size, num_bootstrap_samples = num_bootstrap_samples) 
        ax.axvspan(np.mean(fancy_r) - np.sqrt(1.0/res['FI_MLE']), np.mean(fancy_r) + np.sqrt(1.0/res['FI_MLE']), alpha = 0.1, color = 'r', label = 'std from FI MLE')
        ax.axvspan(np.mean(fancy_r) - np.sqrt(1.0/res['mean_FI']), np.mean(fancy_r) + np.sqrt(1.0/res['mean_FI']), alpha = 0.75, color = 'm', fill = False, 
                   hatch = '--',
                   label = 'std from BS FI')
        ax.axvspan(np.mean(fancy_r) - np.std(fancy_r), np.mean(fancy_r) + np.std(fancy_r), alpha = 0.75, color = 'g', fill = False, 
                   hatch = '//',
                   label = ' bootstrap std')
        
    ax.set_xlabel('1/$\phi$', fontsize = 15)
    ax.set_ylabel('neg Likelihood/max(neg Likelihood)', fontsize = 15)
    ax.axvline(r, alpha = 0.5, color = 'k', label = "True Value")
    if resample: 
        ax.axvline(res['median_MLE'], alpha = 0.75, color = 'k', linestyle = '--', label = "Median MLE value")
        
    else: 
        ax.axvline(np.mean(fancy_r), alpha = 0.75, color = 'k', linestyle = '--', label = "Avg. value")
    ax.legend()
    ax.set_title('Num samples :' + str(sample_size) + ' ;' + 'True r = ' + str(r))
                    
                    
    return naive_r, fancy_r    
    
################################################################################################

def testMaximizationOnSamples(NB_samples, method = 'cython', plot_range = [1, 15], step_size = 0.1, BS_sample_size = 100, num_bootstrap_samples = 10):
    """
    This is a test function to check convergence 
    Args: 
        
        NB_params: if nort none provide a list [r, p, sample_size]
            r: parameter of NB = 1/phi
            p: prob parameter of NB, see definition of NB
            sample_size: How many number to draw? 
        repeats: how may times to repeat such sampling 
        plot_range: of r to see the likelihood
        step_size: compute the likelihood in plot_range at this steps 
    returns: 
        naive_r: naive computation of r  (= 1/phi) 
        fancy_phi: max. likelihood computation of r 
    """
    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(1,1,1)
    rs = np.arange(plot_range[0] + EPSILON, plot_range[1], step_size)
    
    naive_phi = (np.std(NB_samples)**2 - np.mean(NB_samples))/(np.mean(NB_samples))**2
    R = maximize_LL(NB_samples, method = method)
    
   
    naive_r = 1.0/naive_phi
    fancy_r = R['x']
    
    Y = np.zeros_like(rs)
    for j, k in enumerate(rs): 
        Y[j] =  neg_loglikelihood(k, NB_samples)
       
    ax.plot(rs, Y/np.max(Y), color = 'gray', label = 'log_likelihood')
    
    res = maximize_LL_bootstrap(NB_samples, method = method, BS_sample_size = BS_sample_size, num_bootstrap_samples = num_bootstrap_samples) 
    fancy_r = res['median_MLE']
    ax.axvspan(fancy_r - np.sqrt(1.0/res['FI_MLE']), fancy_r + np.sqrt(1.0/res['FI_MLE']), alpha = 0.1, color = 'r', label = 'std from FI MLE')
    ax.axvspan(fancy_r - np.sqrt(1.0/res['mean_FI']), fancy_r + np.sqrt(1.0/res['mean_FI']), alpha = 0.75, color = 'm', fill = False, 
               hatch = '--',
               label = 'std from BS FI')
    temp = res['median_MLE']
    std_temp = res['std_MLE']
    ax.axvspan( temp - std_temp, temp + std_temp, alpha = 0.75, color = 'g', fill = False, 
               hatch = '//',
               label = ' bootstrap std')
        
    ax.set_xlabel('1/$\phi$', fontsize = 15)
    ax.set_xlim(plot_range)
    
    ax.axvline(res['mean_MLE'], alpha = 0.75, color = 'k', linestyle = '--', label = "Avg. value")
    ax.legend()
                    
                    
    return naive_r, fancy_r, res, R   
##############################################################################################################    
    
def compute_pval_sigma(TPMs, mu, r, min_phi): 
    """
    Helper function for the parallel loop in compute_log_p_value_parallel, s_ means replace by self. in 
    passing the arguments

    Args: 
        TPMs: TPM values 
        mu: mean estimate 
        r: inverse phi 
        min_phi: Poisson regime switch 
        
    Returns: 
        log_Prob
        log10_p_values 
        sigma_dev: deviation in sigma 
    """
    phi = 1.0/r  
    if  phi < min_phi:  #i.e., phi <  min phi, then phi = 0 
        log_Prob, log10_p_values = negativeBinomial(TPMs, mu, 0) #poisson regime 
        sigma = np.sqrt(mu)
    else: 
        log_Prob, log10_p_values = negativeBinomial(TPMs, mu, phi) 
        sigma =  np.sqrt(mu + phi*mu**2) 
    sigma_dev = (TPMs-mu)/sigma 
    return log_Prob, log10_p_values, sigma_dev  
        
        
def _compute_pval_sigma_helper(args): 
    return compute_pval_sigma(*args)         
        
################################################################################
########### MAIN MODEL OBJECT ##################################################
################################################################################


class HSModel:
    """
    The Negative Binomial Model (toungue and cheek name is Highly Sensitive (or Hospital Stay) model
    (NBModel is a member function)  
    s^2 \sigma^2 = \phi s^2 m^2 + s m, where s is the unknown unit, m is the mean and \sigma 
    is the standard deviation. phi is the dispersion parameter. Objective is to learn mean phi and s from the mean 
    behavior of all genes, and then learn phi that best explains the rescaled (by unit s) data for every gene
    
    The model is meant to make the best use mathematically possible of multiple controls and multiple replicates 
    and compare robustly acorss experiments. Bias and normlization issues are handled smartly
    @ Swagatam M. 2016 
    """
    def __init__(self, ex_mat, PD_regime = [0.5, 10], NB_regime =[1, 1000000], init_phi = 0.01, TPM_thresh = 10, gene_specific = True):
        
        """
        Args:
            ex_mat: control experiment data frame (gene vs. columns for replicates) 
            PD_regime: Expected TPM regime for Poisson behavior, the model is not sensitive to this, used only to initialize 
                       s in nonlinear optimization routine to minimise RMS errors to fit mean behavior 
            
            NB_regime: Fits the NB model to all genes in the TPM regime defined by these TPM bounds
    
        Attributes: 
        
            ex_mat: ex_mat
            PD_regime: Poisson regime 
            NB_regime: Negative Binomial regime  
            means: the ex_mat means along axis 1 
            var: ex_mat vars along axis 1 
            PD_params: log linear fit pf variance against mean as in polyfit,  \sigma^2 = s m 
            init_unit: unit s from Poisson fit 
            optimize_return: the return of scipy.optimize.minimize for the fit of the mean behavior  
            phi: learned mean phi
            unit: learned s,  MULTIPLY the data by unit to get the right scale
        
        """
        assert len(PD_regime) == 2, "PD_range is two length array"
        assert len(NB_regime) == 2, "NB_range is two length array"
        assert isinstance(ex_mat, pd.DataFrame), "ex_mat should be a pandas Dataframe with columns as replicates"
        if (len(ex_mat.columns) < 6) & gene_specific: 
            warnings.warn("This model is inapplicable in gene-specific mode for less than \
                                                   six samples, setting gene_specific = False")
            self.gene_specific = False 
        else: 
            self.gene_specific = gene_specific 
        
        self.ex_mat = ex_mat
        self.PD_regime = PD_regime
        self.NB_regime = NB_regime 
        self.TPM_thresh = TPM_thresh 
        
        self.means = self.ex_mat.mean(axis = 1)
        self.var = self.ex_mat.var(axis = 1)
        
        
        pass_inds = self.means > self.TPM_thresh 
        print 'Num of genes passed', np.sum(pass_inds)
        self.pass_genes = self.means[pass_inds].index.values
        
        
        # Poission regime
        PD_inds = (self.means > self.PD_regime[0])  & (self.means < self.PD_regime[1])
        # Now linearfit 
        
        x1 = np.log(self.means[PD_inds])
        y1 = np.log(self.var[PD_inds])
        self.PD_params = np.polyfit(x1, y1, 1)
        if np.abs(self.PD_params[0] - 1)> 0.20: # More than 20% error in the expected slope of 0.5 
            warnings.warn('Did not expect more than 20% deviation from Poisson behavior, exponent: ' + str(self.PD_params[0]))
        
        NB_inds = (self.means > self.NB_regime[0])  & (self.means < self.NB_regime[1])

        self.init_unit = np.exp(self.PD_params[1]) # \sigma^2 = s*m, if I multiply the data by s, s^2 \sigma' = s^2 m', which is unitless
        self.init_phi = np.median((self.var[NB_inds] - self.means[NB_inds]*self.init_unit)/self.means[NB_inds]**2) 
        # Turns out having a small init value helps in stead of learning it from the data 
        
        #Y has to be in original units, and also means to be passed to the error and der_error functions 
        Y = np.log(self.var[NB_inds]) - 2.0*np.log(self.means[NB_inds])
        
        error = lambda x: np.mean((Y - self.NBModel(x[0], x[1], self.means[NB_inds]))**2)
        der_error = lambda x: self._der_error(x[0], x[1], Y, self.means[NB_inds])
        
        

        options={'ftol': 1e-06, 'maxiter': 10000} # typically the log likehoods are large 
        # The function is convex, use SLSQP 
        # recall that the bounds are in r, so the range of phi is 10**(-6) to 1000
         
        #bounds = (R_BOUNDS, UNIT_BOUNDS)
        #self.optimize_return = minimize(error, (self.init_phi, self.init_unit), method='SLSQP', jac = der_error, bounds= bounds, 
        #                                options=options)
        
        t1 = time.time()
        bounds = [[R_BOUNDS[0], UNIT_BOUNDS[0] ], [R_BOUNDS[1], UNIT_BOUNDS[1]] ]
        self.optimize_return = least_squares(error, (self.init_phi, self.init_unit), jac = der_error, bounds= bounds, 
                                        ftol =options['ftol'], max_nfev = options['maxiter'], loss = 'soft_l1')
        
        print "took {:.3f} seconds in computing average phi and scale parameter".format(time.time() - t1)
        if not self.optimize_return.success: 
            warnings.warn("NOT OPTIMZED, check optimize_return")
        
        self.phi = self.optimize_return.x[0]
        self.unit = self.optimize_return.x[1]
   

    def _der_error(self, phi, unit, Y, m): 
        """
        private function to pass to optimize jacobian, 2D optimize over phi and s, derivative of error of fit  
        Args: 
            phi: dispersion param
            unit: s 
            Y: log (\sigma^2) - 2*log(m)
            m: means 
        """
        df_dphi, df_dunit = self.der_NBModel(phi, unit, m)
        der_unit = np.mean(2*(self.NBModel(phi, unit, m) - Y)*df_dunit)
        der_phi = np.mean(2*(self.NBModel(phi, unit, m) - Y)*df_dphi)
        return np.asarray([der_phi, der_unit]) 
        
    @staticmethod    
    def NBModel(phi, unit, m, log_scale = 'e'): 
        """
        returns right hand side of the fitting equation log (\sigma^2) - 2*log(m) = log(phi) +  log(1 + 1/(s phi m)) == f(phi, s, m)
        Args: 
            phi: dispersion param
            unit: s 
            m: means 
        """
        if log_scale == 'e': 
            f = np.log(phi) + np.log(1.0 + unit/(phi*m)) 
        elif log_scale == '2': 
            f = np.log2(phi) + np.log2(1.0 + unit/(phi*m)) 
        elif log_scale == '10': 
            f = np.log10(phi) + np.log10(1.0 + unit/(phi*m))
        else: 
            raise AssertionError("Unrecognized log scale: valid options are 'e', '2', '10'. ") 
        return f
    
    @staticmethod
    def der_NBModel(phi, unit, m):
        """
        der of f(phi, s, m) with respect to phi and s 
        """
        df_dphi = 1.0/phi -  unit/(phi*(unit + phi*m))
        df_dunit = 1.0/(unit + phi*m)
        return df_dphi, df_dunit 
   

    def computeDispersionParallel(self, BS_sample_size = 10, num_bootstrap_samples = 10, chunksize = 100, method = 'cython'): 
        """
        Computes the dispersion paramter phi for every gene by maximizing log likelihood (see notes) 
        Args: 
            methos: options are "python", 'cython' 
            TPM_thresh: Only genes with TPM > TPM_thresh are considered 
            bootstrap: the number of BS sampling if bootstrap 
        Attributes: 
            TPM_thresh: 
            results: PD dataframe with the following columns ['phi_naive', 'phi_fancy', 'converge', 'mean scaled', 'var scaled', 'frac_G1']
                     where phi_naive is the naive computation of phi, phi_fancy is the likelihood based computation of phi, whether the optimizaiton 
                     has converged, scaled means (by unit), scaled variance (by unit), and number of TPMs >= unit for the gene
                     dataframe is indexed by genes 
            bootstrap: bootstrap sample size, zero means no boostrap and the algorithm swithces to reporting the MLE alone 
        """
        t1 = time.time() 
        assert self.gene_specific, "Method only applicable for gene_specific computation" 
        BS_test = (num_bootstrap_samples > 0) & (BS_sample_size > 0)  
        if BS_test: 
            assert (BS_sample_size < len(self.ex_mat.columns)), "Bad bootstrap sample size"
            test1 = (BS_sample_size < 0.75*len(self.ex_mat.columns))
            test2 = BS_sample_size < 6
            if test1 | test2: 
                warnings.warn("This not a good BS_sample size! See docs.")
        self.BS_sample_size = BS_sample_size
        self.num_bootstrap_samples = num_bootstrap_samples
        phi_naive = (self.var[self.pass_genes] - self.means[self.pass_genes]*self.unit)/self.means[self.pass_genes]**2 
        
        # help parallel processing     
        num_cores = cpu_count()
        print "Using {} cores".format(num_cores)

        # Do not scale data for exact computation of phi, there are many reasons why that is ambiguous 
        
        if not num_bootstrap_samples:
            pool_args = []
            for g in self.pass_genes: 
                pool_args.append((phi_naive.loc[g], self.ex_mat.loc[g].tolist(), method))  
            my_pool = Pool(num_cores)
            ans = my_pool.map(_maximize_LL_helper, pool_args, chunksize = chunksize)
            my_pool.close() 
            my_pool.join()
            #ans = Parallel(n_jobs=num_cores)(delayed(_maximize_LL_helper)(phi_naive.loc[g], self.ex_mat.loc[g].tolist(), init_r.loc[g]) for g in self.pass_genes)
            self.results = pd.DataFrame(np.asarray(ans), index= self.pass_genes, columns=['r_naive', 'r_fancy', 'converge', 'FI_MLE'])
        else:
            pool_args = []
            for g in self.pass_genes: 
                pool_args.append((phi_naive.loc[g], self.ex_mat.loc[g].tolist(), method, BS_sample_size, num_bootstrap_samples))
            my_pool = Pool(num_cores)
            ans = my_pool.map(_maximize_LL_bootstrap_helper, pool_args, chunksize = chunksize)
            my_pool.close() 
            my_pool.join()
            #ans = Parallel(n_jobs=num_cores)(delayed(_maximize_LL_bootstrap_helper)(phi_naive.loc[g], self.ex_mat.loc[g].tolist(), init_r.loc[g],\
            #        BS_sample_size, num_bootstrap_samples) for g in self.pass_genes)
            self.results = pd.DataFrame(np.asarray(ans), index= self.pass_genes, 
                           columns=['r_naive', 'r_fancy', 'BS_converge', 'median_r_BS', 'mean_r_BS', 'std_r_BS', 
                                        'uncertainty_r(using mean FI)', 'FI_MLE', 'BS_mu', 'std_BS_mu'])
        print "took {:.3f} seconds in computing gene specific dispersion parameters".format(time.time() - t1)
        
    def computePhi(self, min_phi = 0.001): 
        """
        Args: 
            min_phi: The minimum phi resolvable form the data. study the uncertainty vs. r plot in long log space to determine this 
        Attributes: 
            gene_data: pandas data frame with all genes phi, error in phi, mu, error in mu, discrete_phi value, and discerte_phi class index 
        """
        assert self.gene_specific, "Method only applicable for gene_specific computation" 
        assert hasattr(self, 'results'), "First compute Dispersion"
           
        #The error in mu is the standard error 
        ans = [] 
        if 'BS_converge' in self.results.columns: #then bootstrap estimate 
            for g in self.pass_genes: 
                loc_r = self.results.loc[g, 'median_r_BS']
                phi = 1.0/loc_r
                mu = self.results.loc[g, 'BS_mu']
                error_mu = self.results.loc[g, 'std_BS_mu']
            
                if phi < min_phi: 
                    phi = 0
                    error_phi = 0 
                    tag = 1 
                else: 
                    error_r = self.results.loc[g, 'uncertainty_r(using mean FI)']
                    error_phi = error_r/loc_r**2 
                    tag = 0 
                ans.append([phi, mu, error_phi, error_mu, tag]) 
            self.gene_data = pd.DataFrame(ans, index= self.pass_genes, columns = ['phi', 'mu', 'error_phi', 'error_mu', 'Poisson_or_indeter'])
        else:
            for g in self.pass_genes: 
                loc_r = self.results.loc[g, 'r_fancy']
                phi = 1.0/loc_r 
                mu = self.means.loc[g] 
                if phi < min_phi: 
                    phi = 0
                    error_phi = 0 
                    tag = 1 
                else: 
                    error_r = np.sqrt(1.0/self.results.loc[g, 'FI_MLE']) 
                    error_phi = error_r/loc_r**2
                    tag = 0 
                ans.append([phi, mu, error_phi, tag])
            self.gene_data = pd.DataFrame(ans, index= self.pass_genes, columns = ['phi', 'mu', 'error_phi', 'Poisson_or_indeter'])    
                    
    def compute_log_p_value(self, case_data, min_phi = 0.001, est_option = 'BS'): 
        """
        Computes both p_value of individual observations 

        Args: 
            case_data: pandas dataframe with genes for index and TPMs for columns  
            gene_specific: If gene specific is false computes the p-value using the avergage phi 
            est_option: if option is MLE uses MLE stimate, or uses BS of MLE estimates 
        Returns: 
            log_p_value: p-value of observation in log10
            log_prob: probability of observation from control model in log (NOT log10) 
        """
            
        if est_option == 'MLE': 
            warnings.warn('The BS estimate should perform better for modest sample size, using MLE though')
       
        log10_p_values = pd.DataFrame(index = case_data.index, columns = case_data.columns)
        log_Prob = pd.DataFrame(index = case_data.index, columns = case_data.columns)
        
        if self.gene_specific: 
            assert hasattr(self, 'results'), "First compute Dispersion using computeDispersionParallel()" 
            
        for g in case_data.index:
            if g in self.pass_genes: 
                entries = case_data.loc[g].as_matrix()
                if self.gene_specific: 
                    if est_option == 'MLE': 
                        mu = self.means.loc[g] 
                        r = self.results.loc[g, 'r_fancy']
                        
                    elif est_option == 'BS': 
                        mu = self.results.loc[g, 'BS_mu']
                        r = self.results.loc[g, 'median_r_BS'] 
                        
                    else: 
                        raise ValueError("Unknown estimation method")
                    phi = 1.0/r  
                    
                    if  phi < min_phi:  #i.e., phi <  min phi, then phi = 0 
                        log_Prob.loc[g], log10_p_values.loc[g] = negativeBinomial(entries, mu, 0) #poisson regime 
                    else: 
                        log_Prob.loc[g], log10_p_values.loc[g] = negativeBinomial(entries, mu, phi) 
                    
                else:
                    log_Prob.loc[g], log10_p_values.loc[g] = negativeBinomial(entries, self.means.loc[g], self.phi) 
                    
        return log_Prob, log10_p_values 
           
        
    def computeSigmaDiff(self, case_data,  min_phi = 0.001, est_option = 'BS'): 
        """
        Compute the change measured in sigma (standard deviation) units, 1-sigma 2-sigma etc. of the observed case_data 
  
        Args: 
                case_data: pandas dataframe with genes for index and TPMs for columns  
                gene_specific: If gene specific is false computes the p-value using the avergage phi 
                est_option: if option is MLE uses MLE stimate, or uses BS of MLE estimates 
        Returns: 
                sigma_dev: deviation in units of sigma
    
        """
        if est_option == 'MLE': 
                warnings.warn('The BS estimate should perform better for modest sample size, using MLE though')
       
        sigma_dev = pd.DataFrame(index = case_data.index, columns = case_data.columns)
        
        if self.gene_specific: 
            assert hasattr(self, 'results'), "First compute Dispersion using computeDispersionParallel()" 
        
        for g in case_data.index:
            if g in self.pass_genes: 
                entries = case_data.loc[g].as_matrix()
                if self.gene_specific: 
                    if est_option == 'MLE': 
                        mu = self.means.loc[g] 
                        r = self.results.loc[g, 'r_fancy']
                    elif est_option == 'BS': 
                        mu = self.results.loc[g, 'BS_mu']
                        r = self.results.loc[g, 'median_r_BS']     
                    else: 
                        raise ValueError('Unknown estimation method')
                    phi = 1.0/r  
                    if  phi < min_phi:  #i.e., phi <  min phi, then phi = 0 
                        sigma = np.sqrt(mu) #minimum sigma is 1, Possion regime var = mu  
                    else: 
                        sigma = np.sqrt(mu + phi*mu**2 ) #Negative Binomial 
                else:
                    mu = self.means.loc[g]
                    sigma = np.sqrt(mu + self.phi*mu**2) 
                sigma_dev.loc[g] = (entries - mu)/sigma       
        return sigma_dev
        
    
    def detectOddGenes(self, log10_p_thresh = -1, frac_samples = 1, min_phi = 0.001, est_option = 'BS'): 
        """
        Returns a list of genes that are behaving oddly, i.e., has high p-value even for control because they have sub-Poisson behavior or
        they smoke weird stuff 
        
        Args: 
            log10_p_thresh: The thresh below which log10_p_value is significant for this (self-reflective) test  
            frac_samples: in what fraction of samples should be see this behavior? 
            min_phi: Possion model for phi below this value (the lmit of detectibility of phi)
            
        Attributes: 
            min_phi: the minimum phi below which the algorithm returns Poisson behavior 
            log_Prob: The (self-reflective) log probability of observing the controls which the HSModel was trained on
            log10_p_values: the log10 of p-values computed for control data 
            odd_genes: The genes that have high p-value for control 
        """
        assert hasattr(self, 'results'), "First compute Dispersion using computeDispersionParallel()"
        self.min_phi = min_phi 
        self.est_option = est_option
        num_samples = np.int(frac_samples*len(self.ex_mat.columns)) 
        self.log_Prob, self.log10_p_values, self.z_score, _, _, _ = self.compute_log_p_value_sigma_parallel(self.ex_mat, min_phi = min_phi, \
                                                                             est_option = est_option, using = 'parallel')
        genes_sig_counts = (self.log10_p_values < log10_p_thresh).sum(axis = 1)
      
        inds = genes_sig_counts >= num_samples
        self.odd_genes = genes_sig_counts[inds]
       
        
    def compute_log_p_value_sigma_parallel(self, case_data, min_phi = 0.01, est_option = 'BS', using = 'parallel', chunksize = 1000): 
        """
        Computes both p_value of individual observations 
        Parallel version of the regular function with same name 
        
        Args: 
            case_data: pandas dataframe with genes for index and TPMs for columns  
            gene_specific: If gene specific is false computes the p-value using the avergage phi 
            est_option: if option is MLE uses MLE stimate, or uses BS of MLE estimates 
        Returns: 
            log_p_value: p-value of observation in log10
            log_prob: probability of observation from control model in log (NOT log10) 
        """
        if est_option == 'MLE': 
            warnings.warn('The BS estimate should perform better for modest control sample size')
        
        if (est_option == 'BS'):
            warnings.warn("Dispersion computed without bootstrapping, incompatible est_option, reverting to MLE extimate") 
            est_option = 'MLE'

        if self.gene_specific: 
            assert hasattr(self, 'results'), "First compute Dispersion using computeDispersionParallel()"
        
        relevant_genes = list(set(case_data.index.tolist()).intersection(set(self.pass_genes.tolist())))
        assert len(relevant_genes), "Is case data indexed by genes? No common genes between control and case?"
                
        if self.gene_specific:
            if est_option == 'MLE':
                all_mus = self.means.loc[relevant_genes].values
                all_rs = self.results.loc[relevant_genes, 'r_fancy'].values
            elif est_option == 'BS':
                all_mus = self.results.loc[relevant_genes, 'BS_mu'].values
                all_rs = self.results.loc[relevant_genes, 'median_r_BS'].values 
            else: 
                raise ValueError('Unknown option for estimator!')
        else: 
            all_mus = self.means.loc[relevant_genes].values
            all_rs = np.ones(len(relevant_genes))*(1.0/self.phi)
        
        #select genes in case that cross TPM thresh 
        sel_inds = case_data.mean(axis=1) > self.TPM_thresh 
        unseen_genes = list(set(case_data[sel_inds].index.tolist()) - set(relevant_genes)) 
        
        print 'Num of genes not seen in control and expressed above TPM_thresh in case', len(unseen_genes)
        
        unseen_mus = self.TPM_thresh*np.ones(len(unseen_genes))
        unseen_rs = (1.0/self.phi)*np.ones(len(unseen_genes))
        pool_genes = relevant_genes + unseen_genes 
        pool_mus =  np.concatenate((all_mus, unseen_mus),axis = 0) 
        pool_rs = np.concatenate((all_rs, unseen_rs),axis = 0) 
        pool_TPMs = case_data.loc[pool_genes].as_matrix().astype(float) 
        
        if using == 'parallel': 
            num_cores = cpu_count()
            pool_args = [] 
            for i in range(len(pool_genes)): 
                pool_args.append((pool_TPMs[i], pool_mus[i], pool_rs[i], min_phi))    
            print 'Using ', str(num_cores), 'cores'
            my_pool = Pool(num_cores)   
            ans = my_pool.map(_compute_pval_sigma_helper, pool_args, chunksize = chunksize) 
            my_pool.close() 
            my_pool.join() 
            ans = np.asarray(ans) 
            log10_p_values = pd.DataFrame(ans[:,1,:], index = pool_genes, columns = case_data.columns)
            log_Prob = pd.DataFrame(ans[:,0,:], index = pool_genes, columns = case_data.columns)  
            sigma_dev = pd.DataFrame(ans[:,2,:], index = pool_genes, columns = case_data.columns)                       
        elif using == 'C': 
            ans = HSModelCore.compute_pval_sigma(np.ascontiguousarray(pool_TPMs, dtype = float), \
                                                 np.asarray(pool_mus, dtype = float), 
                                                 np.asarray(pool_rs, dtype = float), min_phi) 
            log10_p_values = pd.DataFrame(ans[1], index = pool_genes, columns = case_data.columns)
            log_Prob = pd.DataFrame(ans[0], index = pool_genes, columns = case_data.columns)  
            sigma_dev = pd.DataFrame(ans[2], index = pool_genes, columns = case_data.columns)
        else: 
            raise ValueError("Unknown value for USING") 
            
        return log_Prob, log10_p_values, sigma_dev, pool_genes, relevant_genes, unseen_genes 
    
    
    def computePrimaryResults(self, case_data, method = 'fancy', min_phi = 0.01, est_option = 'BS', **kwargs): 
        """
        computes the primary results dictonary with a fixed set of keys that are the grunge of the computation, 
        this dictionary is updated with filtering
        The results dictionry stores case_data, to make the computation and future filtering error-proof 
        
        Args: 
            case_data: data frame of case
            **kwargs are for compute_log_10_p_value 
        Ruturns: 
            Results: keys are 'log2_fold_change', 'z_score', 'logP' and 'log10PV' and 'genes_queried' and 'case_data' 
            You can update the Results dict by using findSignificantGenes function 
            
        """
        
        common_genes = [] 
        unseen_genes = [] 
        if method == 'fancy': 
            # MAJOR CHNAGE: passing all of case_data now, not just filtered, and Bayesian approach is adopted to "fill in"
            # genes unseen in case  
            logP, log10PV, z_score, genes_queried, common_genes, unseen_genes = self.compute_log_p_value_sigma_parallel(case_data, min_phi = min_phi, \
                                                                             est_option = est_option, **kwargs)
        else: 
            genes_queried = list(set(case_data.index.tolist()).intersection(set(self.pass_genes.tolist())))
            case_data_filtered = case_data.loc[genes_queried]
            warnings.warn("method depricated, index/gene list of *Results* dataframes will not be the same as for \
                          *fancy* method, this only reports on \
                          genes that passed TPM thresh in Control")
            logP, log10PV = self.compute_log_p_value(case_data_filtered, min_phi = min_phi, est_option = est_option)
            z_score = self.computeSigmaDiff(case_data_filtered, min_phi = min_phi, est_option = est_option)

        Results = dict() 
        Results['logP'] = logP
        Results['log10PV'] = log10PV
        Results['genes_queried'] = genes_queried
        Results['common_genes'] = common_genes 
        Results['unseen_genes'] = unseen_genes 
        Results['z_score'] = z_score 
        Results['TPM_thresh'] = self.TPM_thresh 
        Results['case_data'] = case_data 
        Results['control_data'] = self.ex_mat 
        return Results 
        
        
    
################################################################################        
################# END MODEL OBJECT #############################################   
################################################################################



############# Results Maker ####################################################
def significantDEgenes(Results, log2_fold_thresh = 1, p_value_thresh = 0.05, FDR = 0, aggregator = 'mean', check_sign = True): 
    """
    Computes differentially expressed genes. log2 fold change is tested for the aggregate expression in case vs control
    and not for the individual samples---this is because the gene expression is supposed to fluctuate and
    mean/median is the best estimator of fold change
    Alternatively, one colud check whether fold change is tolelable within FDR at a case.aggregated control basis
    I prefer to define fold change as the mean behavior, and look at FDR form the point of view
    of sign change and p_value. 
    Function UPDATES the Results dict, but overrides nothing 
    Args: 
        Results: dict from HSModel method, computePrimaryResults 
        p_value_thresh: p_value threshold 
        check_sign: Check within FDR that the direction of change is also consistent 
        FDR: False Discovery Rate, proportion of samples in which p_value thresh criteria and sign change consistency
        chould be met 
        aggregator: median or mean for fold change computation
        check-sign: boolean to check consistency of sign change for fold change 
    Returns:
        Results dictionary copy, updated by ... 
        Results['DE_genes'] : Differentially expressed genes that are significant 
        Results['log10PV_aggregated'] : Aggregated p-vuale for display purposes (go figure what aggregating p-value means, 
                                        I am not the frequentist here! )
        Results['log2_fold_change'] : log2 fold change 
        Results['log2_fold_change_aggregated'] : log2 fold change aggregated  
        
        Results['votes_for_pos_change'] :  fraction of votes of case samples that the gene was up regulated compared to control  
        Results['votes_for_neg_change'] : same for down 
        Results['z_score_aggregated'] : aggregated chnage in units of sigma from control aggregated 
        Results['genes_strong_PV'] : genes with strong P-value of change irrespective of fold change filter, but cheked for sign
        change with FDR 
    """
   
    # The order of all operations where numpy matrices are used should be the order of genes_seen! Selector indices should
    # be in that order to do the right selection on genes! 
    
    assert(FDR < 0.5), "Very high FDR, values between 0 and 0.5 is acceptable in this version"
    params = {'log2_fold_thresh': log2_fold_thresh, 'p_value_thresh':p_value_thresh, 'FDR': FDR, \
              'aggregator': aggregator, 'check_sign':check_sign}
              
    genes_seen = np.asarray(Results['genes_queried'])  
    case_data_filtered = (Results['case_data'].loc[genes_seen].fillna(0) + EPSILON).as_matrix() # NOTE ORDER 
    control_data_filtered = (Results['control_data'].loc[genes_seen].fillna(Results['TPM_thresh'])).as_matrix()  # NOTE ORDER 
    # I am filling control unseen data with TPM thresh and case with EPSILON because this is the most conservative approach 
    num_samples = case_data_filtered.shape[1]
    #Compute pvalues etc. on the filterted set 
    log10PV = Results['log10PV'].loc[genes_seen] #NOTE ORDER 
    # Now filtering
    test_magnitude = np.abs(log10PV.as_matrix()) > np.abs(np.log10(p_value_thresh)) #test p_value thresh
   
    sigma_dev = Results['z_score'].loc[genes_seen] # NOTE ORDER 
    # Aggregate the data, compute fold change 
    
    if aggregator == 'median':
        case_agg = np.median(case_data_filtered, axis = 1)
        control_agg = np.median(control_data_filtered, axis = 1) 
        log10PV_agg = log10PV.median(1) 
        sigma_dev_agg = sigma_dev.median(1)
    elif aggregator == 'mean': 
        case_agg = np.mean(case_data_filtered, axis = 1)
        control_agg = np.mean(control_data_filtered, axis = 1)
        log10PV_agg = log10PV.mean(1) 
        sigma_dev_agg = sigma_dev.mean(1)
    else: 
        raise AssertionError ("Unknown method, only 'median' and 'mean' allowed") 
    
    log2FC = np.log2(case_data_filtered/control_agg[:, np.newaxis]) 
    test_FC = np.abs(log2FC) >= np.abs(log2_fold_thresh)
    inds_PVFC = np.sum(np.logical_and(test_magnitude, test_FC), axis = 1)/float(num_samples) >= (1-FDR)
    inds_PV = np.sum(test_magnitude, axis = 1) >= (1-FDR)
    
    log2FC_agg = np.log2(case_agg/control_agg)  
    
    sign_change = np.sign(case_data_filtered - control_agg[:, np.newaxis])      
    pos_change_num = np.sum(sign_change == 1, axis = 1)  #how many samples had pos. change? 
    neg_change_num = np.sum(sign_change == -1, axis = 1)  # how many negative? 
    vote_sign_change = np.maximum(pos_change_num, neg_change_num) #What's the vote 
    proportion_pos_votes = pos_change_num/float(num_samples) 
    proportion_neg_votes = neg_change_num/float(num_samples)
    inds_sign  = vote_sign_change/float(num_samples) >= (1-FDR) #NOTE ORDER 
    
    if check_sign: 
        inds_PVFC = inds_PVFC & inds_sign
        inds_PV = inds_PV & inds_sign
    
    DEgenes =  genes_seen[inds_PVFC]
    genes_strong_PV = genes_seen[inds_PV]
    
    
    Res2 = Results.copy()  
    Res2['DE_genes'] = DEgenes
    Res2['log10PV_aggregated'] = log10PV_agg
    Res2['log2_fold_change']  = pd.DataFrame(log2FC, index = genes_seen, columns = Results['case_data'].columns) #NOTE ORDER
    Res2['log2_fold_change_aggregated'] = pd.Series(log2FC_agg, index = genes_seen)  #NOTE ORDER 
    Res2['votes_for_pos_change'] = pd.DataFrame(proportion_pos_votes, index = genes_seen) #NOTE ORDER 
    Res2['votes_for_neg_change'] = pd.DataFrame(proportion_neg_votes, index = genes_seen) #NOTE ORDER 
    Res2['z_score_aggregated'] = sigma_dev_agg 
    Res2['genes_strong_PV'] = genes_strong_PV 
    Res2['params'] = params
    Res2.update(Results)
    return Res2 

##############################################################################

def savePrimaryResults(primaryResultsDict, h5Store, path):
    """ save results of HSModel.computePrimaryResults to a pandas h5Store at specified path
    
    Args: 
            primaryResultsDict - results from HSModel.computePrimaryResults
            h5Store - a writable pandas.io.pytables.HDFStore object
            path - a string corresponding to a path within the HDFStore where data will be written. 
    Ruturns: 
            None
    """
    for k,v in primaryResultsDict.items():
        valueType = type(v)
        if valueType==pd.core.frame.DataFrame or type(v) == pd.core.series.Series:
            h5Store.put(os.path.join(path, k), v)
        elif valueType in set([type([]),type({}),np.ndarray]):
            h5Store.put(os.path.join(path, k), pd.Series(v))
        else:
            sys.stderr.write('WARNING %s not saved, unexpected type %s'%(k, type(v)))




############# DGE Simulator ####################################################


class NB_gen(rv_discrete):
    "Negative Binomial distribution random variable creator, USED CDF! MUCH FASTER"
#     def _pdf(self, x, mu, phi):
#         r = 1.0/phi
#         p = mu**2/(mu**2 + r*mu)
#         logP = r*np.log(r/(r+mu)) + ss.gammaln(r+x) + x*np.log(mu/(r + mu)) - ss.gammaln(r) - ss.gammaln(x+1) 
#         return np.exp(logP) 
    
    def _cdf(self, x, mu, phi): 
        r = 1.0/phi
        p = mu**2/(mu**2 + r*mu)
        cdf = 1 - ss.betainc(x+1, r, p)
        return cdf 

################################################################################
############# Simulator class ##################################################
################################################################################


class DGESimulator: 
    """
    A class to simulate DGE gene expression! 
    WARNING: This uses custom Negative binomial distribution function and rv_discrte in scypi.stats
    (.rvs method) to generate random number from Negative Binomial, and it is buggy in scipy for large 
    TPM. Pass a max TPM to control issues 
    
    """
    def __init__(self, gene_names, exp_names, log_normal_params = [-2.5, 1.0], gamma_params = [1.0, 2.0, 1.0], max_mu = None,  
                 phis = None, mus = None): 
        """
        Args: 
            num_genes: Number of genes (genes are named 1-num_genes)
            num_experiments: Number of experiments 
            log_normal_params: the params for the log normal distribution of phi, usually learned from real DGE data 
            gamam_params: the params for lof mu (mean TPM) Gamma distribution, usuallly learned from real DGE data 
            max_mu: Thresh max. mu ##to control bugginess of rvs##
            phis: you can provide a list of phis corresponding to the gene_names 
            mus: ditto for mu 
        """
        self.gene_names = gene_names 
        num_genes = len(gene_names)
        self.exp_names = exp_names 
        num_exps = len(exp_names)
        self.log_normal_params = log_normal_params 
        self.gamma_params = gamma_params 
        self.ex_mat = pd.DataFrame(index = gene_names, columns = exp_names)
        self.max_mu = max_mu
        
        assert (phis is not None) or (log_normal_params is not None), "Either provide phi or distribution of phi" 
        assert (mus is not None) or (gamma_params is not None), "Either provide mus or distribution of mus" 
        if phis is not None: 
            assert(len(phis) == len(gene_names)), "Length mismatch, gene_names and phis"  
            self.phis = phis
        else: 
            self.phis = np.exp(np.random.normal(loc = log_normal_params[0], scale = log_normal_params[1], size = num_genes))
        
        if mus is not None: 
            assert(len(mus) == len(gene_names)), "Length mismatch, gene_names and mus"
            assert(np.all(mus > 0)), "all mean mus should be postive definite"
            self.mus = mus
        else: 
            self.mus = np.exp(gamma.rvs(gamma_params[0], loc=gamma_params[1], scale=gamma_params[1], size=num_genes))
        self.gene_params = pd.DataFrame(np.vstack((self.mus, self.phis)).T, index = gene_names, columns = ['mu', 'phi'])
        NB = NB_gen(name="NB") 
        for i, g in enumerate(gene_names): 
            self.ex_mat.loc[g] =  NB.rvs(self.gene_params.loc[g, 'mu'], self.gene_params.loc[g, 'phi'], size = num_exps)


    def differentialExpressExperiment(self, de_genes, exp_names = None): 
        """
        creates a new experiment with genes in gene list differentially expressed, this means that the genes
        not differentially epxressed gets a new assignment of random variable according to their intrinsic gene
        params 
        Args: 
            dictionary de_genes with keys genes and values log2_folds if gene exists in control, or TPM if it doesn't  
        """
        mean_phi = np.mean(self.gene_params['phi']) 
        if exp_names is None: 
            exp_names = self.exp_names 
        NB = NB_gen(name="NB")     
        de_ex_mat = pd.DataFrame(index = de_genes.keys(), columns = exp_names)
        for g in de_genes.keys(): 
            if g in self.gene_params.index:
                mu = self.gene_params.loc[g,'mu']*2.0**(de_genes[g])  
                if (self.max_mu is not None) and (mu > self.max_mu): 
                    warnings.warn("differentially expressed TPM exceeds maximum mu, ignoring gene" + str(g))
                phi = self.gene_params.loc[g,'phi']
            else: 
                mu = de_genes[g] 
                phi = np.copy(mean_phi) 
                   
            de_ex_mat.loc[g] =  NB.rvs(mu, phi, size = len(exp_names))

        return de_ex_mat
            
            
################################################################################
############# PLOTTING FUNCTIONS AND TESTINGS ##################################
################################################################################
def runDiagnostics(HS): 
    """
    Takes a HS model and runs a few diagnostics, plots a few interesting plots
    Most complete when you have run computeDispersionParallel() to compute phi
    and have bootstrapped, otherwise, makes do with what is available, with 
    a few warnings printed 
    
    First one is the plot of the mean model fit
    Utility function to check the log (\sigma^2) - 2*log(m) = log(phi) +  log(1 + 1/(s phi m)) == f(phi, s, m) fit 
    """
    
    
    # Plot 1: The mean model fit 
    
    fig = plt.figure(figsize=(20,30))
    ax = plt.subplot2grid((5,2),(0,0))  
    
    inds = HS.means > HS.TPM_thresh
    ms = (HS.means[inds]).tolist() 
    vs = (HS.var[inds]).tolist()  
    Y = np.log10(vs) - 2.0*np.log10(ms)
    ax.scatter(np.log10(ms), Y, alpha = 0.25)

    x = np.linspace(np.min(np.log10(ms)), np.max(np.log10(ms)), 1000)
    f = HS.NBModel(HS.phi, HS.unit, 10.0**x, log_scale = '10')
    ax.set_xlabel('log10 means in units', fontsize = 15)
    ax.set_ylabel('Y', fontsize = 15)
    ax.set_title('Deviation from $\sigma^2 \sim \mu^2 $ behavior, \n $Y \equiv  log_{10}(\sigma^2) - log_{10} (\mu^2) $, \n $  = log_{10}(\phi) + log_{10} (1 + s/\phi\mu) $') 
    ax.plot(x, f, 'r', label = 'optimized fit')
    ax.legend() 
    
    # PLot of sigma vs mu in log scale 
    ax = plt.subplot2grid((5,2),(0,1))  
    ax.scatter(np.log10(ms), np.log10(vs), alpha = 0.25)
    ax.set_xlabel('log10 means in units', fontsize = 15)
    ax.set_ylabel('log10 sigmas', fontsize = 15)
    ax.set_title('NB behavior') 
    f_fit = f + 2*x 
    ax.plot(x, f_fit, 'r', label = 'optimized fit')
    ax.legend() 
    
    mean = HS.means
    var = HS.var
    medians = HS.ex_mat.median(axis = 1)
    genes = HS.pass_genes

    
    results = dict()
    #plot 7: Gamma fit to log mean TPM 
        
    ax = plt.subplot2grid((5,2),(1, 0))

    mu_naive = mean[genes].values  
    H = ax.hist(np.log(mu_naive), 100, normed = True, alpha = 0.75, label = "Naive")

    params2 = gamma.fit(np.log(mu_naive)) #Recall
    #The probability density above is defined in the 'standardized' form. 
    #To shift and/or scale the distribution use the loc and scale parameters. 
    #Specifically, gamma.pdf(x, a, loc, scale) is identically equivalent to 
    # gamma.pdf(y, a) / scale with y = (x - loc) / scale.
    x = np.linspace(np.log(np.min(mu_naive)), np.log(np.max(mu_naive)),1000)
    # fitted distribution
    pdf_fitted = gamma.pdf(x, params2[0], loc= params2[1], scale= params2[2])
    ax.set_xlabel('log mu', fontsize = 15)
    ax.plot(x, pdf_fitted,'k-', alpha = 0.5, label = 'Gamma fit') 
    ax.set_ylim([0,0.6])
    ax.legend()
    results['gamma_of_means(alpha, loc, beta)'] = params2 
        
    if not hasattr(HS, 'results'): 
        warnings.warn('Run computeDispersionParallel() for other diagnostics')
    
    else:
        r_naive = np.asarray(HS.results.r_naive.tolist())
        r_fancy = np.asarray(HS.results.r_fancy.tolist())
        phi_fancy = 1.0/r_fancy 
        phi_naive = 1.0/r_naive 
        
        # Plot 2: log10 r vs log10 uncertainty r both for MLE ans BS estimate 
        ax = plt.subplot2grid((5,2),(1, 1))
        
        if HS.num_bootstrap_samples > 0:  
            ax.scatter(np.log10(HS.results.median_r_BS.tolist()), 
                       np.log10(HS.results['uncertainty_r(using mean FI)'].tolist()), alpha = 0.5, label = 'BS', color = 'm')
        else: 
            warnings.warn('Skipping bootstrap diagnostics, run computePhiParallel() with bootstrap option')
        
        ax.scatter(np.log10(HS.results.r_fancy.tolist()), np.log10(1.0/np.sqrt(HS.results.FI_MLE.tolist())),
                   alpha = 0.5, label = 'MLE' )
        ax.set_xlabel('log10 r', fontsize = 15)
        ax.set_ylabel('log10 uncertainty_r', fontsize = 15)
        ax.legend()
        
        
        
        
        #Plot 3; Distribution of phi (log normal) and fit 
        ax = plt.subplot2grid((5,2),(2, 0))
        
        phi_fancy = 1.0/np.asarray(HS.results.r_fancy.tolist())
        phi_naive = 1.0/np.asarray(HS.results.r_naive.tolist()) 
        H = ax.hist(np.log(phi_fancy[phi_fancy >0]), 100, normed = True, label = "Fancy", color = 'm', alpha = 0.75)
        H = ax.hist(np.log(phi_naive[phi_naive >0]), 100, normed = True, alpha = 0.75, label = "Naive")

        data = np.log(phi_naive[phi_naive > 0]) 
        param = norm.fit(data) # distribution fitting
        # now, param[0] and param[1] are the mean and 
        # the standard deviation of the fitted distribution
        
        results['log_normal_phi(mu, sigma)'] = param 
        x = np.linspace(np.min(np.log(phi_naive[phi_naive >0])), np.max(np.log(phi_naive[phi_naive > 0])),1000)
        # fitted distribution
        pdf_fitted = norm.pdf(x,loc=param[0],scale=param[1])
        ax.set_xlabel('log phi', fontsize = 15)
        ax.set_title('distribution of log $\phi$: dispersion parameter')
        ax.plot(x, pdf_fitted,'k-', alpha = 0.5, label = 'Gaussian fit') 
        ax.set_ylim([0,0.5])
        ax.legend()
        
        #Plot 4: Median TPM vs uncertianty
        ax = plt.subplot2grid((5,2),(2, 1))
        log10_medians = np.log10(medians.loc[genes].tolist())
        
        if HS.num_bootstrap_samples > 0: 
            u_r = np.log10(HS.results['uncertainty_r(using mean FI)'].tolist()) 
            ax.scatter(log10_medians, u_r, alpha = 0.2, color = 'm', label = "BS" )
       
        u_r = np.log10(1.0/np.sqrt(HS.results.FI_MLE.tolist()))
        ax.set_xlabel('log10 medians', fontsize = 15)
        ax.set_ylabel('log10 uncertainty_r', fontsize = 15)
        ax.scatter(log10_medians, u_r, alpha = 0.2, label = 'MLE')
        ax.legend()
        
        
        #Plot 5: r_naive vs uncertianty 
        
        ax = plt.subplot2grid((5,2),(3, 0))
        if HS.num_bootstrap_samples > 0 : 
            u_r = np.log10(HS.results['uncertainty_r(using mean FI)'].tolist())
            inds = r_naive > 0
            ax.scatter(np.log10(np.abs(r_naive[inds])), u_r[inds], alpha = 0.2, label = 'naive_r > 0  BS')

            inds = r_naive <= 0 
            ax.scatter(np.log10(np.abs(r_naive[inds])), u_r[inds], alpha = 0.2, label = 'naive_r < 0 BS', color = 'r')
            
        
        u_r = np.log10(1.0/np.sqrt(HS.results.FI_MLE.tolist()))
            
        ax.set_xlabel('log10(|r_naive|)', fontsize = 15)
        ax.set_ylabel('log10 uncertainty_r', fontsize = 15)
        
        inds = r_naive > 0
        ax.scatter(np.log10(np.abs(r_naive[inds])), u_r[inds], alpha = 0.2, label = 'naive_r > 0 MLE ', color = 'g')

        inds = r_naive <= 0 
        ax.scatter(np.log10(np.abs(r_naive[inds])), u_r[inds], alpha = 0.2, label = 'naive_r < 0 MLE', color = 'm')
        ax.legend() 
        
        # plot 6; Poission behavior expectation (naive) i.e. var -mean vs median TPM
        ax = plt.subplot2grid((5,2),(3, 1))
        diff = np.asarray((var - mean).loc[genes].tolist() )
       
        inds = diff > 0 
        ax.set_xlabel('log10(|var -mean|)', fontsize = 15)
        ax.set_ylabel('log10 median TPM', fontsize = 15)
        ax.scatter(np.log10(np.abs(diff)[inds]), log10_medians[inds], alpha = 0.2, label = 'var -mean > 0 ')
        inds = diff <=0 
        ax.scatter(np.log10(np.abs(diff)[inds]), log10_medians[inds], alpha = 0.2, label = 'var -mean < 0 ', color = 'r')
        ax.legend() 

 
        
        #plot 8; r_naive vs median TPM 
        
        ax = plt.subplot2grid((5,2),(4, 0))
        
        ax.set_xlabel('log10(|r_naive|)', fontsize = 15)
        ax.set_ylabel('log10 median TPM', fontsize = 15)
        
        inds = r_naive > 0
        ax.scatter(np.log10(np.abs(r_naive[inds])), log10_medians[inds], alpha = 0.2, label = 'naive_r > 0 ')

        inds = r_naive <= 0 
        ax.scatter(np.log10(np.abs(r_naive[inds])), log10_medians[inds], alpha = 0.2, label = 'naive_r < 0', color = 'r')
        ax.legend()         
    return results 
################################################################################  


def plotColoredMeanVar(HSMod,  HSMod2 = None, numbins = 10, 
                       cmap = 'RdYlBu', min_max = [-1, 1], xlim = None, ylim = None, scatter_params = {'s':20, 'alpha': 0.75, 'edgecolor': 'none'}): 
    """
    Fancy mean var plot with points colored
    Provide two HSModels for which you have called compute Phis, or just one (nothing to compare). The second plot (model)
    is always colored by the color group of the first
    Args: 
        HSMod: HS model1 (must have called computePhi())
        HSMod2: (optional) HSmodel 2 (ditto) 
        min_max: log10 min and log10 max of phi to be considered in color resolution
        numbins: number of coloring to be done 
        xlim: xaxis limit 
        ylim: yaxis limit 
    """
    if HSMod2 is None: 
        pass_genes = HSMod.pass_genes
    else: 
        pass_genes = np.asarray(list(set(HSMod.pass_genes).intersection(set(HSMod2.pass_genes)))) 
    ms = (HSMod.means.loc[pass_genes]).tolist() 
    vs = (HSMod.var.loc[pass_genes]).tolist()  

    phis = np.log10(HSMod.gene_data.loc[pass_genes, 'phi'].values.astype(float)) #log10 of phis 
    x = np.linspace(np.min(np.log10(ms)), np.max(np.log10(ms)), 1000)
    
    
    if HSMod2 is None: 
        fig, ax = plt.subplots(1, figsize = (10,10)) 
    else: 
        fig, (ax, ax1) = plt.subplots(1,2,  figsize = (20,10), sharex = True, sharey = True)
        
        
    phi_diffs = (phis - np.log10(HSMod.phi)) #I color by the diffs, that way the center is the white 
    #or center color of the divergent scale
    
    bins = np.linspace(min_max[0], min_max[1], numbins)  #I put the diffs in bins
    zerobin_index =  np.digitize(0, bins) #Need to know the bin index of zero and subtract the bin index from all bin indices
    binned_phi_diffs = np.digitize(phi_diffs, bins) - zerobin_index 
    bin_boundaries = np.arange(numbins) + 1 #if you digitize bins to bins you get the bin index 
   
    Norm = mpl.colors.Normalize(vmin = np.min(binned_phi_diffs), vmax =np.max(binned_phi_diffs))
    M = cm.ScalarMappable(norm = Norm, cmap = cmap)

    A = ax.scatter(np.log10(ms), np.log10(vs), c = M.to_rgba(binned_phi_diffs), **scatter_params)
    ax.set_xlabel('log10 means', fontsize = 25)
    ax.set_ylabel('log10 var', fontsize = 25)
    f = HSMod.NBModel(HSMod.phi, HSMod.unit, 10.0**x, log_scale = '10')
    f_fit = f + 2*x 
    ax.plot(x, f_fit, 'gray', linestyle = '-', linewidth = 3, alpha = 0.75, label = 'NB fit')
    ax.plot(x, x, 'gray', linestyle = '--', linewidth = 3, alpha = 0.75, label = 'Poisson')

    M.set_array(numbins)
    if xlim is not None: 
        ax.set_xlim(xlim)
    if ylim is not None: 
        ax.set_ylim(ylim) 

    ax.legend() 
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    cbar = plt.colorbar(M, ticks = bin_boundaries-zerobin_index, norm = Norm,
                        boundaries = bin_boundaries-zerobin_index - 0.5, format = "%0.2f", cax = cbar_ax) 
    cbar.ax.set_yticklabels(np.around(10**(bins + np.log10(HSMod.phi)), 2))   # horizontal colorbar

    
    
    if HSMod2 is not None: 
        
        ms2 = (HSMod2.means.loc[pass_genes]).tolist() 
        vs2 = (HSMod2.var.loc[pass_genes]).tolist()  
 
        A = ax1.scatter(np.log10(ms2), np.log10(vs2), c = M.to_rgba(binned_phi_diffs), **scatter_params)
        ax1.set_xlabel('log10 means', fontsize = 25)
        f = HSMod2.NBModel(HSMod2.phi, HSMod2.unit, 10.0**x, log_scale = '10')
        f_fit = f + 2*x 
        ax1.plot(x, f_fit, 'gray', linestyle = '-', linewidth = 3, alpha = 0.75, label = 'NB fit')
        ax1.plot(x, x, 'gray', linestyle = '--', linewidth = 3, alpha = 0.75, label = 'Poisson')
        M.set_array(numbins)
        ax1.legend() 
        
###########################################################################################################################


def NBfit( data, BS_sample_size = None, num_bootstrap_samples = None):
    """
    NB fit given data
    Args: 
        data: gene expression vector 
    Returns: 
        x: support of the function 
        pmf: probability mass function (empirical), i.e., normalized histogram 
        NB pmf: fitted negative binomial pmf
        mu: NB parameter for mean 
        phi: NB parameter for dispersion 
        res: (ask Swag) The output of MLE estimate with boostrapping 
    """
    
    assert(len(data) >=6), "too few samples to estimate variance" 
    
    if BS_sample_size is None: 
        BS_sample_size = np.max((6, len(data)/2)) 
    if num_bootstrap_samples is None: 
        num_bootstrap_samples = 10 
        warnings.warn("You may want to choose an appropriate bootstrap sample size")
    x = np.arange(0, np.max(data) + 3*np.std(data)) 
    
    res = maximize_LL_bootstrap(data, BS_sample_size= BS_sample_size, num_bootstrap_samples= num_bootstrap_samples)
    phi = 1/res['median_MLE'] 
    mu = np.mean(data)
    logP, _  = negativeBinomial(x, mu, phi)
    n = np.bincount(np.round(data).astype(int)) 
    pmf = n/float(np.sum(n)) 
    return x, pmf, np.exp(logP), mu, phi, res 
    

###########################################################################################################################    

def plotNBfit( data, plot = True, xlim=None, ylim=None, BS_sample_size = None, num_bootstrap_samples = None):
    """
    Plot the distribution of expression data (data)
    Args: 
        data: gene expression vector 
        xlim: xlimits of graph
        ylim: ylimits of graph
    """
    
    assert(len(data) >=6), "too few samples to estimate variance" 
    
    if BS_sample_size is None: 
        BS_sample_size = np.max((6, len(data)/2)) 
    if num_bootstrap_samples is None: 
        num_bootstrap_samples = 10 
        warnings.warn("You may want to choose an appropriate bootstrap sample size")
    x = np.arange(0, np.max(data) + 1) 
    res = maximize_LL_bootstrap(data, BS_sample_size= BS_sample_size, num_bootstrap_samples= num_bootstrap_samples)
    phi = 1/res['median_MLE'] 
    logP, _  = negativeBinomial(x, np.mean(data), phi)
    if plot: 
        plt.figure() 
        n = np.bincount(np.round(data).astype(int)) #Important to do a bincount to get discrete value counts 
        plt.bar(np.arange(len(n)), n/float(np.sum(n)), 0.25, edgecolor = 'None', color = 'gray', alpha = 0.5, \
                label = 'TPMs', align = 'center')
        plt.plot(x, np.exp(logP), color = 'g', linewidth = 4, label = "HSModel learned NB")
        plt.xlabel('TPM')
        plt.ylabel('PDF')
        plt.legend() 

    return phi, res 


#######################################################################################################

def plotGenePdf(choose_gene, HS, case_TPMs = None , xlim = None, ylims = None, plot_poisson = False, bins = 10): 
    """
    Plot the distribution of a gene, given HS model and the sample_TPMs dataframe
    Args: 
        choose_gene: gene name
        HS: Hs Model (after computing phi using computePhi())
        case_TPMs: genes by sample dataframe, if none uses the HS model data  
        xlim: xlimits of graph
        ylim: ylimits of graph
        plot_poisson: Do I also plot the Poisson distribution? 
        bins: number of bins for the histogram 
        
    """
    
    example_TPMs = HS.ex_mat.loc[choose_gene].values
    example_TPMs  = example_TPMs[~np.isnan(example_TPMs)]
    plt.figure() 

    _ = plt.hist(example_TPMs, bins = bins, normed = True, edgecolor = 'None', color = 'gray', alpha = 0.5, label = 'TPMs')
    
    if case_TPMs is not None:
        case_TPMs_loc = case_TPMs.loc[choose_gene].values
        _ = plt.hist(case_TPMs_loc, bins = bins, normed = True, edgecolor = 'None', color = 'red', alpha = 0.5, label = 'case TPMs')

    temp = HS.gene_data.loc[choose_gene]
    x = np.arange(0, np.max(example_TPMs) + 1)
    logP, _  = negativeBinomial(x, temp['mu'], temp['phi'])
    plt.plot(x, np.exp(logP), color = 'g', linewidth = 4, label = "HSModel learned NB")
    if plot_poisson: 
        plt.plot(x, poisson.pmf(x, temp['mu']), linewidth = 4, color = 'm', label = "Poisson")
    plt.xlabel('TPM')
    plt.ylabel('PDF')
    plt.title('Gene:' + choose_gene + '  : $\phi=$' + str(np.around(temp['phi'], 3)))
    if xlim is not None: 
        plt.xlim(xlim)
    if ylims is not None: 
        plt.ylim(ylim) 
    plt.legend() 
####################################################################################

def plotGenePdfCompare(choose_gene, HSModels, model_names, colors = ['grey', 'orange', 'red'], xlim = None, ylims = None, 
                plot_poisson = False, bins = 50): 
    """
    Plot the distribution of a gene, given HS model and the sample_TPMs dataframe
    Args: 
        choose_gene: gene name
        HS: Hs Model (after computing phi using computePhi())
        case_TPMs: genes by sample dataframe, if none uses the HS model data  
        xlim: xlimits of graph
        ylim: ylimits of graph
        plot_poisson: Do I also plot the Poisson distribution? 
        bins: number of bins for the histogram 
        
    """
    from scipy.stats import poisson
    plt.figure(figsize = (20,10)) 

    for i, HS in enumerate(HSModels): 
        example_TPMs = HS.ex_mat.loc[choose_gene].values
        example_TPMs  = example_TPMs[~np.isnan(example_TPMs)]
        _ = plt.hist(example_TPMs, bins = bins, density = True, edgecolor = 'None', color = colors[i], 
                     alpha = 0.5, label = model_names[i] + ' TPMs')
       
       
        temp = HS.gene_data.loc[choose_gene]
        if xlim is not None: 
            max_val = xlim[1]
        else: 
            max_val = np.max(example_TPMs) + 1
        x = np.arange(0, max_val) 
        logP, _  = HSModel.negativeBinomial(x, temp['mu'], temp['phi'])
        poisson_pmf = poisson.pmf(x, temp['mu'])
        plt.plot(x, np.exp(logP), color = colors[i], linewidth = 6, 
                 label = model_names[i] + ' NB dist.'+ '  : $\phi=$' + str(np.around(temp['phi'], 3)))
        plt.plot(x, poisson_pmf, color = colors[i], linewidth = 6, linestyle = '--',
                 label = model_names[i] + ' Poisson dist.')

        plt.xlabel('TPM')
        plt.ylabel('PDF')
        plt.title('Gene:' + choose_gene )
        if xlim is not None: 
            plt.xlim(xlim)
        if ylims is not None: 
            plt.ylim(ylim) 
        plt.legend() 

####################################################################################


def plotMuPhiDists(gene_params, bins = [10, 10]): 
    
    """
    plots the distribution of phi and mu, similar format to HS Model diagnostics 
    Args: 
        gene_params: N x2 array of mus and phis 
    """
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot2grid((1,2),(0,0))  
    phis = gene_params[:,1]
    H = ax.hist(np.log(phis), bins[0], normed = True, color = 'm', alpha = 0.75, edgecolor = 'None')

    data = np.log(phis) 
    param = norm.fit(data) # distribution fitting
    # now, param[0] and param[1] are the mean and 
    # the standard deviation of the fitted distribution

    x = np.linspace(np.min(np.log(phis)), np.max(np.log(phis)),1000)
    # fitted distribution
    pdf_fitted = norm.pdf(x,loc=param[0],scale=param[1])
    ax.set_xlabel('log $\phi$', fontsize = 15)
    ax.plot(x, pdf_fitted,'k-', alpha = 0.5, label = 'Gaussian fit') 
    ax.set_ylim([0,0.6])
    ax.legend()
    #-------------------
    
    ax = plt.subplot2grid((1,2),(0,1))  
    mus = gene_params[:,0]
        #H = ax.hist(np.log(mu_fancy), 100, normed = True, label = "BS", color = 'm', alpha = 0.75)
    H = ax.hist(np.log(mus), bins[1], normed = True, alpha = 0.75, edgecolor = 'None')

    param2 = gamma.fit(np.log(mus)) #Recall
    #The probability density above is defined in the standardized form. 
    #To shift and/or scale the distribution use the loc and scale parameters. 
    #Specifically, gamma.pdf(x, a, loc, scale) is identically equivalent to 
    # gamma.pdf(y, a) / scale with y = (x - loc) / scale.
    x = np.linspace(np.log(np.min(mus)), np.log(np.max(mus)),1000)
    # fitted distribution
    pdf_fitted = gamma.pdf(x, param2[0], loc= param2[1], scale= param2[2])
    ax.set_xlabel('log $\mu$', fontsize = 15)
    ax.plot(x, pdf_fitted,'k-', alpha = 0.5, label = 'Gamma fit') 
    ax.set_ylim([0,0.6])
    ax.legend()
    
    res = dict() 
    res['log_mu_fit_params'] = param2 
    res['log_phi_fit_params'] = param 
    return res
#####################################################################



def checkMyNB(mu, phi, size = 200): 
    """
    check my random number generator from NB working!
    
    """  
    NB = NB_gen(name="NB")
   
    samples = NB.rvs(mu, phi, size = size)
    print "-------------------------------------"
    print "Empirical mean: ", np.mean(samples) 
    print "Empirical variance: ", np.var(samples)
    print "Expected variance: ", mu + phi*mu**2
    print "-------------------------------------"
    #Working fine! Good. Will be slow for > 10^5 TPM
    return samples
    
##################################################################   
    
def varianceStabilize(Cdf, method = 'NB_precise', phi = None): 
    """
    variance Stabilize the data
    Args: 
        Cdf (datframe of expression)
        method: options are Posison, NB NB_precise 
        phi: the NB parameter 
    Returns: 
        stabilized data 
    """

    CC = Cdf.copy() 

    if method == 'Poisson': 
        CC = np.sqrt(CC)
    elif method == 'NB': 
        assert (phi is not None), "provide phi"
        r = 1.0/phi
        CC = np.sqrt(r)*np.arcsinh(np.sqrt(CC/r)) 
    elif method == 'NB_precise': 
        assert (phi is not None), "provide phi"
        r = 1.0/phi
        a = 0.385
        CC = np.sqrt(r - 0.5)*np.arcsinh(np.sqrt((CC + a)/(r - 2*a)))
    else: 
        raise ValueError ("Unknown option") 
    return CC     
    
##########################################################


def plotGenePdfCompare(choose_gene, HSModels, model_names, colors = ['grey', 'orange', 'red'], xlim = None, ylims = None, 
                plot_poisson = False, bins = 50): 
    """
    Plot the distribution of a gene, given HS model and the sample_TPMs dataframe
    Args: 
        choose_gene: gene name
        HS: Hs Model (after computing phi using computePhi())
        case_TPMs: genes by sample dataframe, if none uses the HS model data  
        xlim: xlimits of graph
        ylim: ylimits of graph
        plot_poisson: Do I also plot the Poisson distribution? 
        bins: number of bins for the histogram 
        
    """
    
    plt.figure() 

    for i, HS in enumerate(HSModels): 
        example_TPMs = HS.ex_mat.loc[choose_gene].values
        example_TPMs  = example_TPMs[~np.isnan(example_TPMs)]
        _ = plt.hist(example_TPMs, bins = bins, density = True, edgecolor = 'None', color = colors[i], alpha = 0.5, label = model_names[i] + ' TPMs')

       
        temp = HS.gene_data.loc[choose_gene]
        if xlim is not None: 
            max_val = xlim[1]
        else: 
            max_val = np.max(example_TPMs) + 1
        x = np.arange(0, max_val) 
        logP, _  = HSModel.negativeBinomial(x, temp['mu'], temp['phi'])
        plt.plot(x, np.exp(logP), color = colors[i], linewidth = 6, label = model_names[i] + ' NB dist.'+ '  : $\phi=$' + str(np.around(temp['phi'], 3)))
        plt.xlabel('TPM')
        plt.ylabel('PDF')
        plt.title('Gene:' + choose_gene )
        if xlim is not None: 
            plt.xlim(xlim)
        if ylims is not None: 
            plt.ylim(ylim) 
        plt.legend() 
        
###########################################################################        
##### STANDARD ERROR computaition related fucntions #######################
###########################################################################        


def StirlingSecondKind(n, k):
    """
    Stirling number of Second Kind 
    """
    row = [1]+[0 for _ in xrange(k)]
    for i in xrange(1, n+1):
        new = [0]
        for j in xrange(1, k+1):
            stirling = j * row[j] + row[j-1]
            new.append(stirling)
        row = new
    return row[k]
###########################################################################        


def standardErrorPoissonVariance(N, mu): 
    """
    Stadard error for Poisson 
    Args: 
        N: number of replicates (samples)
        mu: Poisson rate
    Returns: 
        The standard error in ESTIMATE of variance of Poisson from sample, i.e., std of variance estimate 
    """
    # first compute raw moments 
    mu4 = StirlingSecondKind(4, 0) + mu*StirlingSecondKind(4, 1) + mu**2*StirlingSecondKind(4, 2) + mu**3*StirlingSecondKind(4, 3) + mu**4*StirlingSecondKind(4, 4)
    mu3 = StirlingSecondKind(3, 0) + mu*StirlingSecondKind(3, 1) + mu**2*StirlingSecondKind(3, 2) + mu**3*StirlingSecondKind(3, 3) 
    mu2 = mu + mu**2 # because I know that variance, the second central moment is rate, and var = mu_2 - mu^2 
    # compute central moments 
    k2 = mu
    k4 = mu4 - 4*mu*mu3 + 6*mu**2*mu2 - 3*mu**4 
    f = np.sqrt((float((N-1)**2)/N**3)*(k4 - (float(N-3)/(N-1))*k2**2)) 
    return f

###########################################################################        


def standardErrorPoissonMean(N, mu): 
    """
    standard error of Poisson Mean
    """
    #standard error is variance/N for N samples 
    #variance of Poisson is rate
    return np.sqrt(mu/float(N)) 

###########################################################################        

def standardErrorNBVariance(N, mu, phi): 
    """
    Standrard error of NB variance estimate 
    """
    # Moment generating function of NB is ((1 - p)/(1 - pe^t))^r for the t eth moment 
    # I like to wrine NB not in terms of r and p but mu and phi, r = 1/phi, p = mu**2/(mu**2 + r*mu) 
    p = mu/float(mu + phi*mu**2)
    r = mu*p/(1.0 - p)
    # Compute moments, I used the moment generating function p**r/(1 - (1-p) e*t)*r. This gives the correct first moment, r(1-p)/p. Wikipedia uses a different expresison, but see
    # https://probabilityandstats.wordpress.com/2015/02/28/deriving-some-facts-of-the-negative-binomial-distribution/
    mu2 = r*(p - 1)*(r*(p - 1) - 1)/p**2
    mu3 = -r*(p - 1)*(p**2 - 3*p*(p + r*(p - 1) - 1) + (p - 1)**2*(r**2 + 3*r + 2))/p**3
    mu4 = r*(p - 1)*(-p**3 + 7*p**2*(p + r*(p - 1) - 1) - 6*p*(p - 1)**2*(r**2 + 3*r + 2) + (p - 1)**3*(r**3 + 6*r**2 + 11*r + 6))/p**4
    k4 = mu4 - 4*mu*mu3 + 6*mu**2*mu2 - 3*mu**4 
    k2 = mu + phi*mu**2
    f = np.sqrt((float((N-1)**2)/N**3)*(k4 - (float(N-3)/(N-1))*k2**2)) 
    return f
###########################################################################        


def standardErrorNBMean(N, mu, phi): 
    """
    standard error of Poisson Mean
    """
    #standard error is variance/N for N samples 
    #variance of Poisson is rate
    return np.sqrt((mu + phi*mu**2)/float(N))  
