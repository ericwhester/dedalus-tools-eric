# coding: utf-8

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
    return np.array([cheb_mode(x,m,a,b) for m in M],order='F')

def fourier_mode(x, kx,a,b):
    return np.exp(1j*kx*(x-a),order='F')

def fourier_modes(x, kx,a,b):
    temp = np.exp(1j*np.outer(kx,x-a),order='F')
    return temp

def sin_modes(x, kx, a, b):
    return np.sin(np.outer(kx,x-a),order='F')

def cos_modes(x, kx, a, b):
    return np.cos(np.outer(kx,x-a),order='F')

# Getting and sorting bases and modes

basis_names = {de.Fourier:'Fourier',
               de.SinCos:'SinCos',
               de.Chebyshev:'Chebyshev',
               de.Compound:'Compound'}

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
        return 2*np.tensordot(A,B,axes=([-1],[0])).real
    else: return np.tensordot(A,B,axes=([-1],[0]))
# The interpolating function

def interp(u,*grids,comm=None,parallel='single'):
    """Interpolate the field at the grid points.
    Give in x, y, z order (last basis is non-separable).
    parallel='single' will communicate data to rank 0, 
    perform a local transform, then communicate to other processors."""

    bases = u.domain.bases
    grids = [grid.flatten() for grid in grids]
    basis_types = [get_basis_type(basis) for basis in bases]
    if basis_types[-1]=='Compound': return cip.compound_interpolate(u,*grids)
    lasts = is_last(basis_types)
    parities = get_parities(u)
    basis_modes = [get_modes(basis,grid,parity=parity) for basis,grid,parity in zip(bases,grids,parities)]
    
    if not comm:
        comm = u.domain.dist.comm
        rank, size = comm.Get_rank(), comm.Get_size()
    
    if size > 1: # Give global coefficients to each processor and call non-parallelised
        lcshape = u.domain.dist.coeff_layout.local_shape(1)
        gcshape = u.domain.dist.coeff_layout.global_shape(1)

        # Allgather: load full coefficient matrix in all nodes
        sendbuf = u['c']
        recvbuf = np.zeros(np.prod(gcshape),dtype=u['c'].dtype).reshape(gcshape)
        comm.Allgather(sendbuf,recvbuf)
        u0 = recvbuf
    else:
        u0 = u['c'].copy()

    for modes, last in zip(basis_modes[::-1],lasts[::-1]):
        u0 = combine(u0,modes,last=last)
        u0 = u0.transpose(transpose_type(u0))
    return np.ascontiguousarray(u0)
