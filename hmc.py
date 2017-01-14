import numpy as np
import theano
import theano.tensor as T
import distributions
from theano import pp
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed=234)

rep = lambda l: (l() for i in iter(int,1))

# gaussian shells
def l_gs(theta,c,r,w):
  return sum(circle(theta,c[i],r[i],w[i]) for i in (0,1))

def circle(theta,c,r,w):
    return np.exp( -(np.linalg.norm(theta-c)-r)**2/(2*w**2))/(2*np.pi*w**2)

#model
w=[0.1,0.1]
r=[2,2]
#c1 and c2 are 7 units apart. priors on -6 to 6, with allowed error
error = 0.2
c= next(prop for prop in rep(lambda:[np.random.uniform(-6,6,2) for k in (0,1)])
                            if np.abs(np.linalg.norm(prop[0]-prop[1])-7)<error)

def L_gs(a,b):
    return l_gs([a,b],c,r,w)

a = T.dscalar('a')
b = T.dscalar('b')

p = L_gs(a,b)

j = theano.gradient.jacobian(np.log(p),[a,b])
fun = theano.function([a,b],j)

def log_grad(p):
    theta = [T.dscalar() for i in range(2)]
    return theano.function(theta,theano.gradient.jacobian(np.log(p(*theta)),theta))

tmp = log_grad(L_gs)
tmp(1,1)

# p is p is posterior density, theta is last theta, phi is last phi, e is learning rate, L is number of steps, M is Moment? (Cov matrix)
def step(p,theta,phi,e,L,M):
    M_inv = np.linalg.inv(M)
    theta_prev = theta
    phi_prev = phi
    srng.normal(std=M)
    p_phi = mnorm(0,M) #set to multivariate normal
    for i in range(L):
        #half step update
        phi = phi+e/2*T.jacobian(np.log(p(theta,y)),[theta])
        #theta update, Matrix inverse or 1/M? matinv!
        theta = theta + e*M_inv*phi
        #half step update
        phi = phi + e / 2 * T.jacobian(np.log(p(theta, y)), [theta])
    #calc post
    r = p(theta,y)*p_phi(phi)/(p(theta_prev,y)*p_phi(phi_prev))
    #reject accept
    acc = T.le(r,np.random.uniform())
    return acc*theta+(1-acc)*theta_prev



theta_=[T.dscalar('x'),T.dscalar('y')]
step(L_gs,T_)

print(c)