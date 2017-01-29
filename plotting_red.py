
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
#import seaborn as sns
from itertools import islice
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import plotly.plotly as py
import plotly.offline as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from pandas.tools.plotting import autocorrelation_plot


#shameless copy/paste


def plotfun(f,frm,to):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(frm[0], to[0], 0.05)
    Y = np.arange(frm[1], to[1], 0.05)
    X, Y = np.meshgrid(X, Y)
    Z=np.vectorize(f)(X,Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    #ax.set_zlim(0, 250)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
def plotpts(pts,L):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    pts = np.array(pts)
    X = pts[:,0]
    Y = pts[:,1]
    Z=np.vectorize(L)(X,Y)
    surf = ax.scatter(X, Y, Z, cmap=cm.coolwarm)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()
    
"""   
#plots a 2d histogram from a 2D points collection
def hist2d(pnts):
    pnts_ar = np.array(pnts)
    sns.jointplot(pnts_ar[:,0],pnts_ar[:,1], kind='scatter');
    plt.hist(pnts_ar,50,normed=True)
    plt.show()
    plt.plot(pnts)
    plt.show()
"""


def trace3d_ly(pts,f,title='Egg Box Sampling',rng=None):
    X,Y,Z=pts[:,0],pts[:,1],np.array([f(*pt) for pt in pts])
    trace = go.Scatter3d(x=X, y=Y, z=Z,
        marker=dict(size=4,color=Z,colorscale='Viridis',),
            line=dict(color='#1f77b4',width=1))
    data = [trace]
    layout = dict(
        width=800,
        height=700,
        autosize=True,
        title=title,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range = rng
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=rng
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            camera=dict(
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                eye=dict(
                    x=-1.7428,
                    y=1.0707,
                    z=0.7100,
                )
            ),
            aspectratio = dict( x=1, y=1, z=0.7 ),
            aspectmode = 'manual'
        ),
    )
    fig = dict(data=data, layout=layout)
    #return py.iplot(fig, filename=title, height=700, validate=False)
    return py.iplot(fig, filename=title, validate=False)

def plot_2d(x,y):
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.scatter(x, y)
    plt.show()

def dens2d(x,y,label='2d_density'):
    x = np.array(tmp)[100:,2]
    y = np.array(tmp)[100:,3]
    colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]

    fig = FF.create_2D_density(
        x, y, colorscale=colorscale,
        hist_color='rgb(255, 237, 222)', point_size=3
    )

    py.iplot(fig,filename=label)

def plot_regrs(w,b,frm=0,to=1):
    x = np.linspace(frm,to)
    bnds= np.sort([w*x+b for w,b in zip(w,b)],0)
    trace1 = go.Scatter(x = np.append(x,x[::-1]), y=np.append(bnds[-1],bnds[0,::-1]),
        fill='tozerox',
        fillcolor='rgba(0,100,80,0.2)',
        line=go.Line(color='transparent'), name='outline')
    trace2 = go.Scatter(x = x, y = np.mean(bnds,0),fill='lines',name='Maximum Likelihood Estimation')

    #data = [go.Scatter(x=x, y=w*x+b, mode='lines') for w,b in zip(w,b)]
    py.iplot([trace1,trace2])


def plot_regrs_pts(w,b,x,y):
    frm ,to = min(x) ,max(x)
    x = np.linspace(frm,to)
    bnds= np.sort([w*x+b for w,b in zip(w,b)],0)
    trace1 = go.Scatter(x = np.append(x,x[::-1]), y=np.append(bnds[-1],bnds[0,::-1]),
        fill='tozerox',
        fillcolor='rgba(0,100,80,0.2)',
        line=go.Line(color='transparent'), name='outline')
    trace2 = go.Scatter(x = x, y = np.mean(bnds,0),fill='lines',name='Maximum Likelihood Estimation')
    pts = go.Scatter(x=x,y=y, mode = 'markers')
    #data = [go.Scatter(x=x, y=w*x+b, mode='lines') for w,b in zip(w,b)]
    py.iplot([trace1,trace2,pts])