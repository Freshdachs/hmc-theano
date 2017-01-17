import numpy as np

rep = lambda l: (l() for i in iter(int,1))

#model
w=[0.1,0.1]
r=[2,2]
#c1 and c2 are 7 units apart. priors on -6 to 6, with allowed error
error = 0.2
c= next(prop for prop in rep(lambda:[np.random.uniform(-6,6,2) for k in (0,1)])
                            if np.abs(np.linalg.norm(prop[0]-prop[1])-7)<error)

# gaussian shells
def l_gs(theta,c,r,w):
  return sum(circle(theta,c[i],r[i],w[i]) for i in (0,1))

def circle(theta,c,r,w):
    return np.exp( -(np.linalg.norm(theta-c)-r)**2/(2*w**2))/(2*np.pi*w**2)

def L_gs(a,b):
    return l_gs([a,b],c,r,w)