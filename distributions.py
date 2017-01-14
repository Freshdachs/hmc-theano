import numpy as np
import theano
import theano.tensor as T

norm = lambda loc,scale: lambda x: np.exp(-((x-loc)/scale)**2/2)/np.sqrt(2*np.pi)/scale
#mnorm = lambda mu,cov:lambda x:

unif = lambda lower,upper: lambda x: 1/(upper-lower) * T.ge(x,lower)* T.le(x, upper)


x = T.scalar('x')
low = T.scalar('low')
up = T.scalar('up')
dy = theano.gradient.jacobian(norm(low,up)(x),[low,up,x])
df = theano.function([low,up,x],dy)
