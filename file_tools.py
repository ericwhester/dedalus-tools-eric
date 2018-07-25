import numpy as np
import os
import h5py

def save_data(filename, dset, dname, group='/',overwrite=False):
    """Save array and name to a group in an hdf5 file.
    
    Parameters
    ----------
    
    dsets: numpy array
        dataset
    dnames: string
        dataset name
    filename: string
        file name
    group: string, optional
        subgroup of hdf5 file to write to
    overwrite: boolean, optional
    
    Returns
    -------
    None
    """
    with h5py.File(filename,'a') as f:
        if group not in f.keys(): f.create_group(group)
        g = f[group]
        if dname in g.keys(): 
            if overwrite: 
                g[dname][...] = dset
                return
            else: del g[dname]
        g[dname] = dset
    return

def make_group(filename,name,group='/'):
    """Make a group in an hdf5 file."""
    with h5py.File(filename,'a') as f:
        g = f[group]
        if name not in g.keys():
            g.create_group(name)
    return


def load_data(filename, *dnames, group='/',show=False,flatten=True,sel=None,checkint=True):
    """Load list of arrays given names of group in an hdf5 file.
    
    Parameters
    ----------
    dnames: list
        strings of dataset names
    filename: string
        file name
    group: string, optional
        subgroup of hdf5 file to write to
    overwrite: boolean, optional
    show: boolean, optional
    flatten: boolean, optional
        return number if single value
    sel: slice object, optional
        return slice of data array
    Returns
    -------
    List of numpy arrays

    """

    with h5py.File(filename,'r') as f:
        arrs = []
        g = f[group]
        for dname in dnames:
            if show: print(dname)    
            if not sel: sel = Ellipsis
            arr = g[dname][sel]
            if flatten:
                if arr.size == 1: arr = arr.item()
                elif np.prod(arr.shape) == max(arr.shape): arr = arr.flatten()                
                elif arr.shape[0] == 1: arr = arr[0,Ellipsis]
            if checkint and isinstance(arr,float) and arr.is_integer(): arr = int(arr)
            arrs.append(arr)
    return arrs

def makedir(path):
    """Simple wrapper to make a path if it doesn't exist
    
    Parameters
    ----------
    path : string
    """
    if not os.path.isdir(path): os.makedirs(path)

def get_keys(filename, group='/'):
    """ Helper to get keys of an hdf5 file/group.

    Parameters
    ----------
    filename : string
        The hdf5 file name
    group : the subgroup of the file, optional
    """
    with h5py.File(filename, 'r') as f:
        g = f[group]
        keys = sorted(list(g.keys()))
    return keys

def print_keys(filename, group='/', formatted=False):
    """Wrapper to print keys of an hdf5 file/group.

    Parameters
    ----------
    filename : string
        The hdf5 file name
    group : the subgroup of the file, optional
    """
    keys = get_keys(filename, group=group)
    if formatted:
        for key in keys: print(key)
    else:
        print(keys)
    return

def delete(filename,*names,group='/'):
    """Delete file or group in an hdf5 file."""
    with h5py.File(filename,'a') as f:
        g = f[group]
        for name in names:
            if name in g.keys(): del g[name]
    return

def move(filename,origin,destination):
    """Move file or group in an hdf5 file."""
    with h5py.File(filename,'a') as f:
        f[destination] = f[origin]
    return

def save_domain(path,domain,):
    """Save domain object information."""
    orders = [str(i) for i in range(len(domain.bases))]
    types = [type(basis).__name__ for basis in domain.bases]
    names = [basis.name for basis in domain.bases]
    sizes = [basis.base_grid_size for basis in domain.bases]
    intervals = [basis.interval for basis in domain.bases]
    for typ,name,order,size,interval in zip(types,names,orders,sizes,intervals):
        order = str(order)
        for arr, name in zip([typ,name,size,interval],['type','name','size','interval']): save_data(path,arr,name,group=order)

def load_domain(path,comm=None):
    """Load domain object."""
    from dedalus import public as de
    if comm:
        from mpi4py import MPI
        if comm=='world': comm = MPI.COMM_WORLD
        elif comm=='self': comm = MPI.COMM_SELF
    classes = {'Fourier':de.Fourier,'Chebyshev':de.Chebyshev,'SinCos':de.SinCos}
    orders, bases = sorted(get_keys(path)), {}
    for order in orders:
        name,typ,size,interval = load_data(path,'name','type','size','interval',group=order)
        bases[order] = classes[typ](name,size,interval=interval)
    domain = de.Domain([bases[order] for order in orders], grid_dtype=np.float64, comm=comm)
    return domain

def pickle_save(obj,name,dr=''):
    """Save python object with pickle."""
    if dr: flt.makedir(dr)
    with open(join(dr,name+'.pickle'),'wb') as f:
        pickle.dump(obj,f)

def pickle_load(name,dr=''):
    """Load python object with pickle."""
    with open(join(dr,name+'.pickle'),'rb') as f:
        return pickle.load(f)
