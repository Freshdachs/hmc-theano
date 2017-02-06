import numpy as np
import distributions as dists


rep = lambda l: (l() for i in iter(int,1))

# gaussian shells example

w=[0.1,0.1]
r=[2,2]
#c1 and c2 are 7 units apart. priors on -6 to 6, with allowed error
error = 0.2
c= next(prop for prop in rep(lambda:[np.random.uniform(-6,6,2) for k in (0,1)])
                            if np.abs(np.linalg.norm(prop[0]-prop[1])-7)<error)

def l_gs(theta,c,r,w):
  return sum(circle(theta,c[i],r[i],w[i]) for i in (0,1))

def circle(theta,c,r,w):
    norm = lambda v: np.sqrt((v**2).sum())
    return np.exp( -(norm(theta-c)-r)**2/(2*w**2))/(2*np.pi*w**2)

def L_gs(theta):
    return l_gs(theta,c,r,w)


# Egg Box Example
def l_egg(x,y):
    return np.exp((2+np.cos(x/2)*np.cos(y/2))**5)

#uniform prior 0..10*pi for x and y
frm_egg = np.array([0,0])
to_egg = np.array([10*np.pi, 10*np.pi])

def prior_egg(pt):
    return dists.unif(frm_egg,to_egg)(pt)



def l_p_egg(theta):
    x,y = theta[0], theta[1]
    return prior_egg(theta)*l_egg(x,y)


# 2 Variables
def p_x(x):
    return dists.unif(0.5, 1.5)(x)


def p_y_x(y, x):
    return dists.norm(y, 1)(x)


def p_xy(theta):
    y,x = theta[0],theta[1]
    return p_y_x(y, x) * p_x(x)