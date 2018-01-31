import numpy as np
import os
import h5py

def save_data(dsets, dnames, savename, group='/',overwrite=False):
    """Save list of arrays and names to a group in an hdf5 file.
    
    Parameters
    ----------
    
    dsets: list
        numpy arrays for datasets
    dnames: list
        strings of dataset names
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
        for dset, dname in zip(dsets, dnames):
            print(dname)            
            if dname in g.keys(): 
                if overwrite: 
                    g[dname][...] = dset
                    return
                else: del g[dname]
            g[dname] = dset
    return

def load_data(dnames, savename, group='/',show=False,flatten=True):
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

def print_keys(filename, group='/', formatted=False):
    """Wrapper to print keys of an hdf5 file/group.

    Parameters
    ----------
    filename : string
        The hdf5 file name
    group : the subgroup of the file, optional
    """
    with h5py.File(filename, 'r') as f:
        g = f[group]
        keys = sorted(list(g.keys()))
        if formatted:
            for key in keys: print(key)
        else:
            print(keys)
    return

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
