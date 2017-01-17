import numpy as np
import theano
import theano.tensor as T
import distributions as dists
from theano import pp
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from theano.tensor import slinalg
from theano.sandbox import linalg
import scipy.stats as stats
import model


#stats.multivariate_normal

srng = RandomStreams(seed=234)

#declarations

def p(theta):
    return model.L_gs(theta[0], theta[1])

# We need to know p at this point
M = T.dmatrix('M')
phi = dists.multivar(M)
theta = T.dvector('theta')
L = T.iscalar('L')
e = T.dscalar('e')

M_inv = linalg.matrix_inverse(M)
theta_prev = theta
p_phi = dists.mnorm(M) # T -> T
 #set to multivariate normal

def half_phi(phi,theta):
    return [ phi + e / 2 * T.jacobian(T.log(p(theta)), [theta]), theta]

# theta update, Matrix inverse or 1/M? matinv!
def theta_up(phi,theta):
    return [ phi, theta + e * T.dot(M_inv, phi)]

single_step =  lambda phi, theta: half_phi(*theta_up(*half_phi(phi,theta)))

(phis,thetas), updates = theano.scan(single_step,non_sequences=[phi, theta], n_steps = L )
phi_new,theta_new = phis[-1],thetas[-1]
#calc post
r = p(theta_new)*p_phi(phi_new)/(p(theta)*p_phi(phi))
#reject accept
#acc = T.le(r,dists.coin())
acc = dists.coin(r)
ret = acc*theta_new+(1-acc)*theta

smplr = theano.function([theta,e,L,M],ret)


# p is p is posterior density, theta is last theta, phi is last phi, e is learning rate, L is number of steps, M is Moment? (Cov matrix)
def step(p,theta,e,L,M):
    pass

smplr([0,0],1/20,20,[])

