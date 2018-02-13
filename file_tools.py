import numpy as np
import os
import h5py

def save_data(dset, dname, savename, group='/',overwrite=False):
    """Save array and name to a group in an hdf5 file.
    
    Parameters
    ----------
    
    dsets: numpy array
        dataset
    dnames: string
        dataset name
    savename: string
        file name
    group: string, optional
        subgroup of hdf5 file to write to
    overwrite: boolean, optional
    
    Returns
    -------
    None
    """
    with h5py.File(savename,'a') as f:
        if group not in f.keys(): f.create_group(group)
        g = f[group]
        if dname in g.keys(): 
            if overwrite: 
                g[dname][...] = dset
                return
            else: del g[dname]
        g[dname] = dset
    return

def make_group(name,savename,group='/'):
    """Make a group in an hdf5 file."""
    with h5py.File(savename,'a') as f:
        g = f[group]
        if name not in g.keys():
            g.create_group(name)
    return


def load_data(savename, dnames, group='/',show=False,flatten=True):
    """Load list of arrays given names of group in an hdf5 file.
    
    Parameters
    ----------
    dnames: list
        strings of dataset names
    savename: string
        file name
    group: string, optional
        subgroup of hdf5 file to write to
    overwrite: boolean, optional
    show: boolean, optional
    flatten: boolean, optional
        return number if single value
        
    Returns
    -------
    List of numpy arrays

    """

    with h5py.File(savename,'r') as f:
        arrs = []
        g = f[group]
        for dname in dnames:
            if show: print(dname)    
            arr = g[dname][...]
            if flatten and arr.size == 1: arr = arr.item()
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

def delete(name,savename,group='/'):
    """Delete file or group in an hdf5 file."""
    with h5py.File(savename,'a') as f:
        g = f[group]
        if name in g.keys(): del g[name]
    return
