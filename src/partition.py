# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:52:14 2019

@author: W-book
"""
from scipy.cluster import  hierarchy
import numpy as np
import scipy as sp
import numpy as np


# cutting the tree
# functions

def find_dendrogram_path(node, Z):
    """returns the dendrogram-path "leaf to root" of a node
  
    node : int
    Z : linkage matrix from scipy.cluster.hierarchy.linkage
    """
    i = np.where(Z[:,:2] == node)[0][0]
    correlation_distance = Z[i,2]
    dendrogram_path = np.array(correlation_distance)
    while i != (Z.shape[0] - 1):
        node = Z.shape[0] + 1 + i
        i = np.where(Z[:,:2] == node)[0][0]
        correlation_distance = Z[i,2]
        dendrogram_path = np.append(dendrogram_path, correlation_distance)
    
    return dendrogram_path

def gap_function(x, dendrogram_path):
    """returns the output for x of the gap function corresponding to the dendrogram_path
  
    x : float
    dendrogram_path : output of the find_dendrogram_path fonction
    """
    support = np.insert(dendrogram_path, 0, 0)
    branches_lengths = np.diff(support)
    output = 0
    for i in np.linspace(0, dendrogram_path.size - 1, dendrogram_path.size, dtype = "int16"):
    # not optimised but does the job
        if (support[i] <= x) and (x < support[i + 1]):
            output = branches_lengths[i]
      
    return output

def global_gap_function(x, Z):
    """returns the output for x of the global function corresponding to the linkage matrix
  Note : Here the constant is not included because we are looking for the argmax
  
  x : float
  Z : linkage matrix from scipy.cluster.hierarchy.linkage
  """
    output = 0
    max_corr_dist = Z[-1,2]
    N = Z.shape[0] + 1
    for a in np.linspace(0, Z.shape[0], Z.shape[0] + 1, dtype = "int16"):
        dendrogram_path = find_dendrogram_path(a, Z)
        output = output + gap_function(x, dendrogram_path)
    
    output = output/(N*max_corr_dist)
    
    return output
