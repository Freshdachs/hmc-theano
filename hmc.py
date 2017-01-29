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

def gen_sample(p):
    # We need to know p at this point
    M = T.dmatrix('M')
    phi = dists.multivar(M)
    theta = T.dvector('theta')
    L = T.iscalar('L')
    e = T.dscalar('e')


    def single_step(phi, theta, M_inv,e):
        half_phi = lambda phi, theta: (phi + e / 2 * T.jacobian(T.log(p(theta)), theta), theta)
        theta_up = lambda phi, theta: (phi, theta + e * T.dot(M_inv, phi))
        return half_phi(*theta_up(*half_phi(phi, theta)))

    def sample_step(theta,e,L,M):

        M_inv = linalg.matrix_inverse(M)
        p_phi = dists.mnorm(M) # T -> T
        #scan
        (phis,thetas), updates = theano.scan(single_step,outputs_info=[phi, theta],non_sequences=[M_inv,e] ,n_steps = L )
        phi_new,theta_new = phis[-1],thetas[-1]
        #calc post
        r = p(theta_new)*p_phi(phi_new)/(p(theta)*p_phi(phi))
        #reject accept
        acc = T.gt(r,dists.srng.uniform())
        #return  acc*theta_new+(1-acc)*theta
        return T.switch(acc,theta_new,theta)


    n=T.iscalar()
    out, updates = theano.scan(sample_step,outputs_info=theta,non_sequences=[e,L,M],n_steps=n)
    return theano.function([theta,e,L,M,n],out)


"""
M_ = np.eye(2)/np.var(np.array(model.c).T,1)
fd_up={M:np.eye(2)/6, theta:[0,0],e:1/5}
#fd={M:np.eye(2)/6, L:20,e:1/20,theta:[1,1]}
fd={M:M_, L:20,e:1/20,theta:[1,1]}
"""

#[i.eval(fd_up) for i in theta_up(*half_phi(phi,theta))]

# p is p is posterior density, theta is last theta, phi is last phi, e is learning rate, L is number of steps, M is Moment? (Cov matrix)
"""
def sample(theta,e,L,M):
    while True:
        t_new = smplr(theta,e,L,M)
        yield t_new
        theta=t_new
"""

smplr_gs=gen_sample(model.L_gs)

def test(theta=[0,0],L=10,M=np.eye(2),n=5000):
    pts=smplr_gs(theta,1/L,L,M,n)
    ratio = lambda pts: sum(not np.allclose(a, b) for (a, b) in zip(pts, pts[1:])) / pts.shape[0]
    print("ratio: %f"%ratio(pts))
    print("means: "+str(np.mean(pts,0) ))
    print("circle_means: "+str(np.mean(model.c,0)))
    print("var: "+str(np.var(pts,0)))
    return pts

M_ =np.eye(2)/np.var(np.array(model.c).T,1)
#pts = test(M=M_,L=5)

def sample(f,i,n=5000,L=15,M=np.eye(2)):
    sampler = gen_sample(f)
    return sampler(i,1/L,L,M,n)