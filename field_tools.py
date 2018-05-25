import numpy as np
from mpi4py import MPI
from dedalus import public as de

def reset(*fields, scales=1):
    """
    Short wrapper to rescale fields.

    Parameters
    ---------
    first : dedalus fields
    scale : int, float or tuple

    Returns
    -------
    dedalus field
    """

    for field in fields: field.set_scales(scales)
    return 

def get_grids(field, scales=1):
    """
    Short wrapper to get doman, basis, grid, and elements corresponding to field.

    Parameters
    ---------
    first : dedalus field
    scale : int
    """

    domain = field.domain
    bases = domain.bases
    if isinstance(scales, int) or isinstance(scales,float): scales = (scales,)*len(bases)
    x = [basis.grid(scale) for basis,scale in zip(bases,scales)]
    kx= [basis.elements for basis in bases]
    field.set_scales(scales)
    return domain, bases, x, kx

def compound_coefficients(field):
    """
    Return subdomain elements and coefficients for compound basis.

    Parameters
    ---------
    first : dedalus field
    """

    domain, bases, x, kx = get_grids(field)
    kxs = [subbasis.elements for subbasis in bases[-1].subbases]
    cs = [bases[-1].sub_cdata(field['c'], i, 0) for i in range(len(bases[-1].subbases))]
    return kxs, cs

def higher_res(*fields, scales=2):
    """
    Short wrapper to get higher resolution grid and fields of dedalus fields.

    Parameters
    ---------
    first : dedalus field
    second : int

    Returns
    -------
    x_new : numpy array
        Higher res grid
    u_new : dedalus field
        Higher resolution field
    """

    domain, bases, x_new, kx = get_grids(fields[0],scales=scales)
    new_fields = domain.new_fields(len(fields),scales=scales)
    reset(*fields,scales=scales)
    for new_field, field in zip(new_fields, fields):
        new_field['g'] = field['g']
    reset(*fields)
    return x_new + list(new_fields)
