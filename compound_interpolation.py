# coding: utf-8                                                                           
"""                                                                                       
This module allows you to interpolate cylinder initial conditions defined on a compound basis.                                                                                     
"""

import numpy as np
import time
import h5py
import math as m
from scipy.special import erf
import glob, os
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD

import dedalus.public as de
from dedalus.core.field import Field
from dedalus.extras import flow_tools
from dedalus.tools import file_tools as flt
from dedalus.tools import field_tools as ft
from dedalus.tools import logging
from dedalus.tools import post
from dedalus.tools import interpolation as ip

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)

def sizes_to_slices(dims,arr):
    """Create list of numpy slice objects given size of subdomain slices."""
    sl = [(slice(None),)*(dims-1) + (slice(sum(arr[:index]), sum(arr[:index+1])),) for index in range(len(arr))]
    print(arr)
    print(sl)
    return sl

def get_sub_domains(field):
    """Create list of subdomains for field on compound domain."""
    domain = field.domain
    comm = domain.dist.comm
    bases = domain.bases
    sub_bases = bases[-1].subbases
    sub_domains = [de.Domain(bases[:-1]+[sub_basis], grid_dtype=np.float64,comm=comm) for sub_basis in sub_bases]
    return sub_domains

def get_grid_slices(field):
    """Create list of slices for subslices of field grid."""
    sizes = [sub_basis.grid_size(1) for sub_basis in field.domain.bases[-1].subbases]
    return sizes_to_slices(len(field['g'].shape),sizes)

def get_sub_fields(field):
    """Create and initialize list of subfields for a field on a compound domain."""
    sub_domains = get_sub_domains(field)
    sub_grid_slices = get_grid_slices(field)
    sub_fields = []
    for index, sub_domain in enumerate(sub_domains):
        sub_fields.append(sub_domain.new_field(scales=1))
        sub_fields[-1].meta = field.meta # parities problem
        sub_fields[-1]['g'] = field['g'][sub_grid_slices[index]]
    return sub_fields

def get_intervals(field):
    """Return the subgrid intervals given field on compound domain."""
    return [basis.interval for basis in field.domain.bases[-1].subbases]

def check_sub_slice(field,r):
    """Return list of subslices of given r grid, and corresponding numpy slices."""
    intervals = get_intervals(field)
    if r.min() < intervals[0][0] or r.max() > intervals[-1][-1]: raise ValueError('Extrapolating outside domain.')
        
    r_subs = []
    for index, interval in enumerate(intervals):
        r_subs.append(r[(r>=interval[0])&(r<=interval[-1])]) # possible interval endpoint bug
        
    sub_slices = sizes_to_slices(len(field['g'].shape),[sub_arr.size for sub_arr in r_subs])
    return r_subs, sub_slices

def compound_interpolate(field, *grids):
    """Perform interpolation of field on compound domain."""
    final_arr = np.zeros([grid.size for grid in grids])
    sub_fields = get_sub_fields(field)
    sub_zs, sub_slices = check_sub_slice(field, grids[-1])
    for sub_slice, sub_field, sub_z in zip(sub_slices, sub_fields, sub_zs):
        final_arr[sub_slice] = ip.interp(sub_field,*grids[:-1],sub_z)
    return final_arr
