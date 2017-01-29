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

def single_step(phi,theta,M_inv):
    half_phi= lambda phi,theta: ( T.flatten(phi + e / 2 * T.jacobian(T.log(p(theta)), [theta])), theta)
    theta_up = lambda phi,theta: (phi, T.flatten(theta + e * T.dot(M_inv, phi)))
    return half_phi(*theta_up(*half_phi(phi,theta)))

def sample_step(theta,e,L,M):
    M_inv = linalg.matrix_inverse(M)
    p_phi = dists.mnorm(M) # T -> T
    #scan
    (phis,thetas), updates = theano.scan(single_step,outputs_info=[phi, theta],non_sequences=M_inv ,n_steps = L )
    phi_new,theta_new = phis[-1],thetas[-1]
    #calc post
    r = p(theta_new)*p_phi(phi_new)/(p(theta)*p_phi(phi))
    #reject accept
    acc = T.gt(r,dists.srng.uniform())
    return  acc*theta_new+(1-acc)*theta


n=T.iscalar()
out, updates = theano.scan(sample_step,outputs_info=theta,non_sequences=[e,L,M],n_steps=n)
smplr = theano.function([theta,e,L,M,n],out)


fd_up={M:np.eye(2)/6, theta:[0,0],e:1/5}
fd={M:np.eye(2)/6, L:20,e:1/20,theta:[1,1]}


#[i.eval(fd_up) for i in theta_up(*half_phi(phi,theta))]

# p is p is posterior density, theta is last theta, phi is last phi, e is learning rate, L is number of steps, M is Moment? (Cov matrix)
def sample(theta,e,L,M):
    while True:
        t_new = smplr(theta,e,L,M)
        yield t_new
        theta=t_new


smplr([0,0],1/20,20,np.eye(2)/6,200)

gen = sample([0,0],1/10,10,np.eye(2)/6)

def iterate(f,pt):
    while True:
        pt = f(pt)
        yield pt


def iter_nth(gen,n):
    return [next(gen) for i in range(n)][-1]


def head_n(gen,n=1000):
    return np.array([next(gen) for i in range(n)])

lambda pt: single_step(*pt)

ratio = lambda pts: sum(not np.allclose(a, b) for (a,b) in zip(pts,pts[1:]))/pts.shape[0]

def test(theta=[0,0],L=10,M=np.eye(2),n=5000):
    pts=smplr(theta,1/L,L,M,n)
    print("ratio: %f"%ratio(pts))
    print("means: "+str(np.mean(pts,0) ))
    print("circle_means: "+str(np.mean(model.c,0)))
    print("var: "+str(np.var(pts,0)))
    return pts

M_ = M=np.eye(2)/np.var(np.array(model.c).T,1)
pts = test(M=M_,L=20)