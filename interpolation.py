# coding: utf-8

import math as m
import numpy as np
from mpi4py import MPI

from dedalus import public as de
import compound_interpolation as cip

from numpy.polynomial import chebyshev as cb

# The mode functions for each basis type

def rescale(x,a,b,c,d):
    return c + (d - c)*(x - a)/(b - a)

def cheb_mode(x, m, a, b):
    x_scaled = rescale(x,a,b,-1,1)
    return cb.chebval(x_scaled, np.append(np.zeros(m),1))

def cheb_modes(x, M, a, b):
    return np.array([cheb_mode(x,m,a,b) for m in M])

def fourier_mode(x, kx,a,b):
    return np.exp(1j*kx*(x-a))

def fourier_modes(x, kx,a,b):
    temp = np.exp(1j*np.outer(kx,x-a))
    return temp

def sin_modes(x, kx, a, b):
    return np.sin(np.outer(kx,x-a))

def cos_modes(x, kx, a, b):
    return np.cos(np.outer(kx,x-a))

# Getting and sorting bases and modes

basis_names = {de.Fourier:'Fourier',de.SinCos:'SinCos',de.Chebyshev:'Chebyshev',de.Compound:'Compound'}

def modes(basis,parity=0):
    """Return appropriate mode functions of basis."""
    if basis == 'Fourier': return fourier_modes
    elif basis == 'SinCos':
        if parity == 1: return cos_modes
        elif parity==-1:return sin_modes
        else: raise ValueError('Parity not specified')
    elif basis == 'Chebyshev': return cheb_modes
    else: raise NameError('Incorrect basis type')

def get_basis_type(basis):
    """Get type of basis."""
    return basis_names[type(basis)]
        
def get_modes(basis,x,parity=0):
    """Get the grid values of the basis mode functions."""
    func = modes(basis_names[type(basis)],parity=parity)
    if max(x.shape) == x.size: x = x.flatten()
    return func(x,basis.elements,*basis.interval)

def get_parities(u):
    """Get parities of each dimension of field. Return 0 if not SinCos basis."""
    bases = u.domain.bases
    parities = np.zeros(len(bases))
    for i, basis in enumerate(bases):
        if get_basis_type(basis)=='SinCos': parities[i] = u.meta[basis.name]['parity']
    return parities

def transpose_type(arr):
    """Build tuple to cycle backward through axes."""
    indices = list(range(arr.ndim))
    return [indices[-1]]+indices[:-1]

def is_last(bases):
    """List of booleans for checking if the first Fourier basis.
    
    It's the last one to be transformed, from complex to real."""
    lasts = [False for _ in bases]
    i,_ = next(((i,name) for i, name in enumerate(bases) if name=='Fourier'), (None,None))
    if i is not None: lasts[i] = True
    return lasts

def combine(A, B, last=False):
    """Correct combination of mode grid values and coefficients."""
    if last: 
        B[0,:] = B[0,:]/2
        return 2*np.dot(A,B).real
    else: return np.dot(A,B)
# The interpolating function

def interp(u,*grids):
    """Interpolate the field at the grid points.

    This isn't parallelised.
    Give in x, y, z order (last basis is non-separable)."""
    bases = u.domain.bases
    basis_types = [get_basis_type(basis) for basis in bases]
    if basis_types[-1]=='Compound': # Do compound interpolation
        return cip.compound_interpolate(u,*grids)
    lasts = is_last(basis_types)
    parities = get_parities(u)
    basis_modes = [get_modes(basis,grid,parity=parity) for basis,grid,parity in zip(bases,grids,parities)]
    u0 = u['c'].copy()
    for modes, last in zip(basis_modes[::-1],lasts[::-1]):
        u0 = combine(u0,modes,last=last)
        u0 = u0.transpose(transpose_type(u0))
    return u0

def interpolate_2D(u, x, z, comm=None, basis_types=('Fourier','Chebyshev')):
    """
    Interpolation for a dedalus field at grid given by the points np.meshgrid(x,z).

    
    """
    # Get mpi communications 
    domain = u.domain
    comm = domain.dist.comm
    rank, size = comm.rank, comm.size
    
    # Get bases and shapes
    xbasis,zbasis = domain.bases
    lcshape = domain.dist.coeff_layout.local_shape(1)
    gcshape = domain.dist.coeff_layout.global_shape(1)
    
    # Build global coefficients in local processor with MPI
    if size > 1:
        # prepare to send local coefficients to rank 0
        sendbuf = u['c'].copy()
        recvbuf = None
        if rank == 0: recvbuf = np.empty([size, *lcshape], dtype=np.complex128)
        comm.Gather(sendbuf, recvbuf, root=0)
        
        # send global coefficients to every processor from rank 0
        if rank == 0: gcoeffs = np.reshape(recvbuf,gcshape)
        else: gcoeffs = np.empty(gcshape, dtype=np.complex128)
        comm.Bcast(gcoeffs, root=0)    
    else: gcoeffs = u['c']    
    
    # Build correct z basis interpolation functions
    if basis_types[1] == 'SinCos':
        u_parity = u.meta['z']['parity']
        if u_parity == 1:   zfunc = lambda z : trig_cos_vals(z, gcshape[1], interval=zbasis.interval)
        elif u_parity ==-1: zfunc = lambda z : trig_sin_vals(z, gcshape[1], interval=zbasis.interval)
    elif basis_types[1] == 'Chebyshev':
        zfunc = lambda z : cheb_vals(z, gcshape[1], interval=zbasis.interval)
    
    # Get the grid values of all the modes (split for Fourier)
    xsc = fourier_cos_vals(x.flatten(), gcshape[0], interval=xbasis.interval)
    xss = fourier_sin_vals(x.flatten(), gcshape[0], interval=xbasis.interval)
    zs = zfunc(z.flatten())

    # Perform the transform
    B, C = gcoeffs.real, gcoeffs.imag
    F = np.dot(xsc,B) + np.dot(xss,C)
    G = np.dot(F,zs)
    return G


















