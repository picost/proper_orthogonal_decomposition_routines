# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:29:10 2016

This script implements the POD decomposition according to the snapshot method
using scipy.linalg.svd algorithm.

@author: picost
"""

import scipy
import scipy.linalg
import numpy as np

def POD(snapshots,tshd=None,n_modes=None,projection=False,coordinates=True,
        eigenvalues=True,normalize=True,debug=False):
    """Returns a dict with the result of the POD.

    Parameters
    ----------
    snapshots : (M,N) np.ndarray
        Matrix to decompose which columns represent the snapshots
    tshd : float, optional
        Number between 0 and 1. If not None, the number of mode will be chosen
        such that the amount of energy/variance represented by the selected
        modes is at least tshd*100% of the total amount.
        /!\ The number of modes will be anyway limited by n_modes if its value
        is not set None /!\
    n_modes : int, optional
        Maximum number of modes to be kept. /!\ Priority-holder versus tshd /!\
        If not None, the number of modes kept will be the min of n_modes and
        the maximum number of modes (min snapshots.shape). If n_mode and tshd
        are not None, then the number of modes conserved will be the min of the
        total number of modes to reach the threshold and n_mode.
    projection : bool, optional
        If set to True, the projected snapshots matrix on the reduced basis is
        computed and exported to the result dictionnary.
    coordinates : bool, optional
        If set to True, the coordinates of the snapshots are computed and
        exported to the result dictionnary.
    eigenvalues : bool, optional
        If set to True, the eigenvalues are given in the result dictionnary.
    normalize : bool, optional
        If set to True, the snapshots are normalized before the decomposition.
        Default is True.

    Returns
    -------
    res : dictionnary containing the following data.
        modes : (M,n_modes) ndarray
            Matrix which columns are the modes of the decomposition.
        n_modes: int
            Number of modes kept.
        var_ratio:
            Ratio of variance/energy accounted for by the modes.
        eigenvalues: ndarray
            Array of eigenvalues for each modes (even those dropped).
            Returned if eigenvalues == True
        coos: ndarray
            Matrix of the coordinate of the modes on each snapshot, stored by
            columns.
            Returned if coordinates == True
        projected_snapshots: (M,N) ndarray
            Matrix of snapshots projected on the truncated basis.
            Returned if projection == True
        normalized: bool
            value of normalize argument when the function was called
        mean: ndarray
            Collumn array of mean values of each variables on the snapshots.
            This vector is needed to reconstruct the values of the field if coo
            rdinates on the modes are given.
            field = (modes @ coos)*std + mean
            Returned if normalize == True
        std: ndarray
            Collumn array of std of values of each variables on the snapshots
            This vector is needed to reconstruct the values of the field if coo
            rdinates on the modes are given.
            field = (modes @ coos)*std + mean
            Returned if normalize == True
    Example
    -------
    >>> import numpy as np
    >>> A = np.array([[1,2,3],[4,50,4],[7,8,9]])
    >>> POD(A,tshd=0.95,normalize=True,projection=True,coordinates=True,
            eigenvalues=True,n_modes=None)

        {'projected_snapshots': array([[  1.,   2.,   3.],
                                       [  4.,  50.,   4.],
                                       [  7.,   8.,   9.]]),
         'modes': array([[ -7.07106781e-01,   5.55111512e-17],
                         [ -1.66533454e-16,  -1.00000000e+00],
                         [ -7.07106781e-01,   8.32667268e-17]]),
         'coos': array([[ 1.73205081,  0.        , -1.73205081],
                        [ 0.70710678, -1.41421356,  0.70710678]]),
         'n_modes': 2,
         'eigenvalues: array([2.44948974e+00, 1.73205081e+00, 5.35627810e-17]),
         'var_ratio': 1.0}
     >>> POD(A,tshd=0.95,normalize=False,projection=True,coordinates=True,
             eigenvalues=True,n_modes=None)

        {'projected_snapshots':
            array([[  1.79566797,   1.98225462,   2.42542983],
                   [  4.0023765 ,  49.999947  ,   3.99828387],
                   [  6.78811996,   8.00472545,   9.15300346]]),
         'modes': array([[-0.04715824,  0.25298193],
                         [-0.98101792, -0.19389562],
                         [-0.18809556,  0.94784209]]),
         'coos': array([[ -5.28789887, -50.64997699,  -5.75840649],
                        [  6.1122941 ,  -1.60608035,   8.51394214]]),
         'n_modes': 2,
         'eigenvalues': array([ 51.24979306,  10.60315256,   1.01580876]),
         'var_ratio': 0.98384239093957537}

    Relevant information
    --------------------
    The POD is computed using scipy.linalg svd to compute the singular value
    decomposition of the matrix of snapshots. (even if not optimally efficient)
    The link between POD and SVD is well explained in the PhD. thesis of
    M. Bergmann:

    @PhdThesis{bergmann2004phd,
      Title = {Optimisation aérodynamique par réduction de modèle POD et
          contrôle optimal. Application au sillage laminaire d'un cylindre
          circulaire.},
      Author = {Bergmann, Michel},
      School = {INP Loraine},
      Year = {2004},
      Keywords = {POD, ROM, Reduced order model},
      Url  = {http://www.math.u-bordeaux1.fr/~mbergman/HTML/These.html}
    }

    """
    assert not isinstance(n_modes,bool), "n_modes be None or have type int"
    assert n_modes is None or isinstance(n_modes,int) , \
        "n_modes be None or have type int"
    assert isinstance(coordinates,bool), "coordinates must have type bool"
    assert isinstance(eigenvalues,bool), "eigenvalues must have type bool"
    assert isinstance(normalize,bool), "normalize must have type bool"
    assert isinstance(projection,bool), "projection must have type bool"
    assert tshd is None or isinstance(tshd,float), "tshd must have type float or be None"
    res = {}
    res['normalized'] = normalize
    #The snapshots are normalized if requested
    if normalize:
        mean = snapshots.mean(axis=1)
        res['mean'] = mean.reshape(-1,1)
        std = snapshots.std(axis=1)
        std[std == 0] = 1
        snapshots = (snapshots-mean.reshape(-1,1))/std.reshape(-1,1)
        res['std'] = std.reshape(-1,1)
    try:
        #The SVD (singular value decomposition) of the matrix is performed.
        U,S,V = scipy.linalg.svd(snapshots,full_matrices=False)
    except ValueError:
        if debug:
            print("The SVD algorithm failed, returning the snapshot matrix.")
            return(snapshots)
    #Setting the maximum number of modes conserved
    #This number is n_modes if specified or K = len(S) = min(M, N)
    #(see numpy.linalg.svd documentation)
    K = len(S)
    if n_modes is not None:
        n_modes_max = n_modes
    else:
        n_modes_max = K
    #If a variance threshold is specified, then the modes are selected so that
    #the accumulated variance of the modes that are kept is at least tshd*total
    #variance
    if tshd is not None:
        if tshd > 0 and tshd < 1:
        ##loop to identify which modes should be selected in order to keep
        #tshd*100% "d'energie"
            etot = S.sum()
            en = 0
            n_modes=0
            while (not (en > tshd*etot) and not (n_modes>K-1)):
                en+= S[n_modes]
                n_modes+= 1
        else:
            #If tshd doesn't have a value between 0 and one then keep the maxi
            #-mum bumber of modes
            n_modes = K
    n_modes = min(n_modes,n_modes_max)
    res['n_modes'] = n_modes
    #compute the variance ratio for the n_modes first modes
    en = S[0:n_modes].sum()
    etot = S.sum()
    enpct = en/etot
    res['var_ratio'] = enpct
    if eigenvalues:
        res['eigenvalues'] = np.copy(S)
    if coordinates or projection:
        #if the coordinates or projections are requested, then the truncated
        #matrix of singular values is assembled
        if n_modes < K:
            S[n_modes:]=np.zeros(K-n_modes)
            #singular values of dropped modes are replaced by zeros
        new_sigm = np.zeros((K,K))
        new_sigm[0:K,0:K] = np.diag(S)
        #Computing the coordinates of the snapshots on the modes
        coos = np.dot(new_sigm,V)
        res['coos'] = coos[0:n_modes,:]
        if projection:
            #computing the projection of ths snapshots on the space spaned by
            #the selected modes
            new_snaps = np.dot(U,coos)
            if normalize:
                #denormalize the projected snapshots before returning them
                new_snaps = new_snaps*std.reshape(-1,1) + mean.reshape(-1,1)
            res["projected_snapshots"] = new_snaps
    res['modes'] = U[:,0:n_modes]
    return(res)