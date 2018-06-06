import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize

# Convenient functions
def logmag(arr): return np.log10(np.abs(arr))

# Plotting 1D data

def ax_settings(ax,xlim=None, ylim=None, aspect=None, xlabel=None, ylabel=None, title=None):
    """Apply kwargs to axis."""
    if xlim: ax.set(xlim=xlim)
    if ylim: ax.set(ylim=ylim)
    if aspect: ax.set(aspect=aspect)
    if xlabel: ax.set(xlabel=xlabel)
    if ylabel: ax.set(ylabel=ylabel)
    if title: ax.set(title=title)

def plot1D(x,arr,
           fig=None, ax=None, savename=None, figsize=None,
           xlim=None, ylim=None, aspect=None, xlabel=None, ylabel=None, title=None,grid=True,
           **kwargs):
    """Simple 1D plot."""
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    plot = ax.plot(x, arr, **kwargs)
    ax_settings(ax, xlim=xlim,ylim=ylim,aspect=aspect,xlabel=xlabel,ylabel=ylabel,title=title)
    if grid: ax.grid()
    if savename: plt.savefig(savename,bbox_inches='tight')
    return fig, ax

# Plotting 2D Cartesian data
def cplot(x1,x2,arr,
          fig=None,ax=None,figsize=None,
          dpi=200,savename=None,colorbar=None,
          xlim=None, ylim=None, aspect=None, xlabel=None, ylabel=None, title=None, 
          **kwargs):
    """2D pcolormesh."""
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    if len(x1.shape) == 1: x1, x2 = np.meshgrid(x1,x2,indexing='ij')
    plot = ax.pcolormesh(x1,x2,arr,**kwargs)
    ax_settings(ax, xlim=xlim,ylim=ylim,aspect=aspect,xlabel=xlabel,ylabel=ylabel,title=title)
    if colorbar: plt.colorbar(plot,fraction=0.046,pad=0.04,shrink=.7)
    if savename: plt.savefig(savename,dpi=dpi,bbox_inches='tight')
    return fig, ax

def coeff_plot(array,fig=None,ax=None,savename=None,dpi=200,
               vmin=-10,vmax=0,colorbar=True,reverse=True,
               xlim=None,ylim=None,aspect=None,xlabel=None,ylabel=None,title=None,
               **kwargs):
    fig, ax = plt.subplots()
    kx, ky = np.arange(array.shape[0]), np.arange(array.shape[1])
    plot = ax.pcolormesh(kx,ky,logmag(array).T,vmin=vmin,vmax=vmax,**kwargs) 
    ax_settings(ax, xlim=xlim,ylim=ylim,aspect=aspect,xlabel=xlabel,ylabel=ylabel,title=title)
    if colorbar: cbar = plt.colorbar(plot)
    if reverse: cbar.ax.invert_yaxis()
    if savename: plt.savefig(savename,dpi=dpi,bbox_inches='tight')
    return fig, ax 

# Plotting 2D Polar data
def extend_angle(*arrays):
    """Complete the periodic mesh to remove missing slice in polar pcolormesh."""
    return [np.concatenate([arr,arr[[0],:]],axis=0) for arr in arrays]
    
def polar_plot(θθ,rr,array,
               fig=None,ax=None,savename=False,dpi=200,colorbar=True,
               return_plot=False,**kwargs):
    """Wrapper to create a polar plot of a quantity."""
    if fig==None: fig, ax = plt.subplots(figsize=(4,6),subplot_kw=dict(projection='polar'))
    half = θθ.min() >= 0
    if not half: θ_full, r_full, arr_full = extend_angle(θθ,rr,array)
    else: θ_full, r_full, arr_full = θθ,rr,array
    plot = ax.pcolormesh(θ_full,r_full,arr_full,**kwargs)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if half:
        ax.set_thetamin(0)
        ax.set_thetamax(180)
    if colorbar: 
        if not half: plt.colorbar(plot,ax=ax,orientation='horizontal')
        else: 
            cax = fig.add_axes([.12,.3,.78,.03])
            cbar = plt.colorbar(plot,cax=cax,orientation='horizontal')
    if savename: plt.savefig(savename,dpi=dpi,bbox_inches='tight')
    if return_plot: return fig, ax, plot
    return fig, ax    
    
def polar_vel_plot(θθ,rr,ur, uθ,
                   sel=None,fig=None,ax=None,savename=False,dpi=200,
                   double=False,ticks=False,grid=True,**kwargs):
    """Wrapper for a polar quiver plot."""
    if sel==None: sel = (slice(0,-1,8),slice(0,-1,8))
    if fig==None: fig, ax = plt.subplots(figsize=(5,4), subplot_kw=dict(projection='polar'))
    half = θθ.min() >= 0
    ux, uy = cartesian_velocities(θθ,rr,ur, uθ)
    speed = np.hypot(ur, uθ)
    plot = ax.quiver(θθ[sel],rr[sel],ux[sel],uy[sel],speed[sel],**kwargs)
    if not ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if half:
        ax.set_thetamin(0)
        ax.set_thetamax(180)
    if grid: ax.grid()
    plt.colorbar(plot,ax=ax,orientation='vertical')
    if savename: plt.savefig(savename,dpi=dpi,bbox_inches='tight')    
    return fig, ax    

# Plotting immersed boundaries
cdict_K_black = {
           'red':  ((0.0, 0.5, 0.5), (1.0, 0.5, 0.5)),
           'green':((0.0, 0.5, 0.5), (1.0, 0.5, 0.5)),
           'blue': ((0.0, 0.5, 0.5), (1.0, 0.5, 0.5)),
           'alpha':((0.0, 0.0, 0.0), (0.01, 0.0, 0.0),(1.0, 1, 1)) }
cmap_K = LinearSegmentedColormap('cmap_K', cdict_K_black)
plt.register_cmap(cmap=cmap_K)

def color_norm(vmin, vmax):
    """Wrapper to create a normalize object to scale a colorbar (particularly for quiver plots."""
    return Normalize(vmin=vmin, vmax=vmax)
