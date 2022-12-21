from scipy.special import expi
from numpy import exp, sqrt, log
import numpy as np

def get_NS_ONB(theta, tau_lower=1, tau_upper=30):
    
    '''
    Return the three orthonormal bases for Nelson-Siegel according to Gram–Schmidt process in Theorem 1
    
        theta: parameter in Nelson-Siegel model
    tau_lower: lower bound for maturity in years
    tau_upper: upper bound for maturity in years
    '''
    
    A = theta * (tau_upper - tau_lower)
    B = theta * tau_lower
    D = (exp(-B) - exp(-A-B)) / A
    psi02norm2 = (exp(-2*B) - exp(-2*(A+B))) / (2*A) - D**2
    psi02norm = sqrt(psi02norm2)
    F = (log(abs(A+B)) - log(abs(B))) / A
    H = (expi(-(A+B)) - expi(-B)) / A
    J = (D + 1) * H - D*F - (expi(-2*(A+B)) - expi(-2*B)) / A 
    K = - J / psi02norm2
    L = -F + H + J*D/psi02norm2
    psi03norm2 = 1/B/(A+B) + 2*exp(-A-B)/A/(A+B) + 2*expi(-A-B)/A - 2*exp(-B)/A/B - 2*expi(-B)/A - exp(-2*A-2*B)/A/(A+B) \
                - 2*expi(-2*A-2*B)/A + exp(-2*B)/A/B + 2*expi(-2*B)/A + K**2 * (exp(-2*B) - exp(-2*(A+B))) / (2*A) + L**2\
                + 2*K*(H - (expi(-2*A-2*B)-expi(-2*B))/A) + 2*L*(F-H) + 2*K*L*D
    psi03norm = sqrt(psi03norm2)

    psi01 = lambda x: np.array([1 for _ in x])
    psi02 = lambda x: (exp(-(A*x+B)) - D) / psi02norm
    psi03 = lambda x: ( (1 - exp(-(A*x+B))) / (A*x+B) + K*exp(-(A*x+B)) + L ) / psi03norm
    
    return psi01, psi02, psi03