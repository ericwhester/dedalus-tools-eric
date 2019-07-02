import numpy as np
import skfmm
import dedalus.public as de
import interpolation as ip
from mpi4py import MPI
commw = MPI.COMM_WORLD
comms = MPI.COMM_SELF

class Mask:
    """
    Class for mask functions.
    Creates callable mask functions with prescribed profiles as a function of distance from desired interface.
    
    Parameters:
    old_function (func): 
        function from grid to [0,1]. Input mask. 0.5 level set is desired interface
    mask_function (func): 
        vectorised function to [0,1]. Desired normalized mask profile near interface.
    smooth (float):
        Desired smoothness of mask.
    shift (float):
        Desired shift of mask.
    domain (dedalus domain):
        Domain of mask.
    basis_types (None, tuple of dedalus basis classes):
        SinCos (Dirichlet, or Neumann) or Fourier (Periodic) basis for each dimension
    parities (None, or tuple of +/- 1s):

    Attributes:
    field: dedalus field to be interpolated using interp method.
    """
    
    def __init__(self,old_func,mask_profile,smooth,shift,domain,
                 factor=1,narrow=0.,basis_types=None,parities=None):
        self.old = old_func
        self.mask = mask_profile
        self.smooth = smooth
        self.shift = shift
        
        self.make_new_domain(domain,factor=factor,basis_types=basis_types) # Create new grid
        self.make_old_mask_values() # Calculate old mask function on new grid
        self.make_distance_function(narrow=narrow) # Calculate distance function on new grid
        self.make_new_mask_values() # Calculate new mask function on new grid
        self.make_new_function(parities=parities) # Create new mask field, to be interpolated
    
    def make_new_domain(self,domain,factor=1,basis_types=None):
        """Create new uniformly spaced grid."""
        if basis_types==None: basis_types = [de.Fourier for b in domain.bases]
        self.bases = [Basis(b.name,int(b.coeff_size*factor),interval=b.interval) for Basis,b in zip(basis_types,domain.bases)]
        self.domain = de.Domain(self.bases,grid_dtype=np.float64,comm=comms) # local to each processor

    def make_old_mask_values(self,):
        """Calculate the old mask function on the new grid."""
        self.old_mask_vals = self.old(*self.domain.grids())
    
    def make_distance_function(self,narrow=0.):
        """Calculate distance function from old mask function on new grid."""
        dx = [b.grid_spacing()[0] for b in self.domain.bases]
        distance = skfmm.distance(self.old_mask_vals-.5,dx=dx,narrow=narrow)
        if isinstance(distance,np.ma.MaskedArray):
            distance[distance.mask & (self.old_mask_vals < .4)] = -narrow
            distance[distance.mask & (self.old_mask_vals > .6)] =  narrow
        self.distance = distance
    
    def make_new_mask_values(self,):
        """Calculate new mask values from distance function. Positive shift is out of object."""
        self.new_mask_vals = self.mask((self.distance+self.shift)/self.smooth)
        
    def make_new_function(self,parities=None):
        """Create n-dimensional linear interpolation of new mask values."""
        field = self.domain.new_field()
        if parities:
            for dim,parity in zip(field.meta,parities): field.meta[dim]['parity'] = parity
        field['g'] = self.new_mask_vals
        self.field = field
        
    def __call__(self,*grid,check=True):
        """Return new mask values on grid."""
        for arr in grid: 
            if max(arr.shape) != arr.size: raise ValueError('Must give 1D arrays.')
        return ip.interp(self.field,*grid,comm=comms)
    
