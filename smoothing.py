import math as m
import numpy as np
from mpi4py import MPI
import time
import glob, os
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.tools import field_tools as ft
from dedalus.tools import file_tools as flt
from dedalus.tools import interpolation as ip

def smooth(mask_func,δ,*gs,domain=None,comm=None,res=32):
    """Smooth given mask function, with defined width δ.
    
    Create periodic domain with uniform grid,
    apply gaussian filter on 0/1 array,
    initialize field with smooth array,
    interpolate on to provided grid points."""
    
    if comm == None: comm = MPI.COMM_SELF
    bases = domain.bases
    lengths = [b.interval[1] - b.interval[0] for b in bases]
    b0s = [de.Fourier(b.name+'0', int(l*res), b.interval, dealias=1) for b, l in zip(bases,lengths)]
    d0 = de.Domain(b0s,grid_dtype=np.float64,comm=comm) 
    g0s = [b0.grid(1) for b0 in b0s]
    gg0s = np.meshgrid(*g0s,indexing='ij')

    mask = mask_func(*gg0s)
    smooth = gaussian_filter(mask,δ*res/4)

    K0 = d0.new_field()
    K0['g'] = smooth
    
    return ip.interp(K0,*gs)
