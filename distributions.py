import numpy as np
import theano
import theano.tensor as T
import theano.tensor.slinalg as slinalg
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams(seed=234)

norm = lambda loc,scale: lambda x: np.exp(-((x-loc)/scale)**2/2)/np.sqrt(2*np.pi)/scale

unif = lambda lower,upper: lambda x: 1/(upper-lower) * T.ge(x,lower).all()* T.le(x, upper).all()


def mnorm(sig,mu=0):
    k = np.shape(sig)[0]
    left = (2*np.pi)**(-k/2)*T.nlinalg.det(sig)**-.5
    inv = T.nlinalg.matrix_inverse(sig)
    def dens(x):
        return left*np.exp(-.5* T.dot(T.dot(x-mu, inv), (x-mu).T))
    return dens

def multivar(cov):
    rv_n = srng.normal([np.shape(cov)[0]])
    L = slinalg.cholesky(cov)
    return T.dot(L,rv_n)

def coin(wght=0.5):
    return srng.binomial(p=wght)
