#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Basic imports
import mne 
import numpy as np



# custom functions

# This should be somewhere in mne python too nowadays
def _get_ica_map(ica, components=None):
    """Get ICA topomap for components"""
    fast_dot = np.dot
    if components is None:
        components = list(range(ica.n_components_))
    maps = fast_dot(ica.mixing_matrix_[:, components].T,
                    ica.pca_components_[:ica.n_components_])
    return maps

# Function to calculate the weighted cosine similarity 
# inputs:
    # topo1, topo2 (topographies to calculate the distance between)
    # distance (the distance tensor to use for weighting)
def topo_dist(topo1, topo2, distance):
    numerator = (topo1.T.dot(distance)).dot(topo2)
    norm1 = np.sqrt((topo1.T.dot(distance)).dot(topo1))
    norm2 = np.sqrt((topo2.T.dot(distance)).dot(topo2))
    if norm1 != 0 and norm2 != 0:
        distance = numerator /(norm1*norm2)
    else:
        distance = 0
    return distance

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


# Variables needed:
# 0. parameter to define if we use the whitening or not,
# False means we _do_ use the whitening
no_weight = False

    
# 1. the combined epochs
combined_epochs_fname = "PATH TO YOUR EPOCHS"
epochs = mne.read_epochs(combined_epochs_fname)


# 2. the ICA decomposition of the combined epochs    
ica_fname = "PATH TO YOUR ICA DECOMP"
ICA_dec = mne.preprocessing.read_ica(ica_fname)
ICA_dec.apply(epochs)

# Extract the mixing matrix from it
mixing_matrix =  _get_ica_map(ICA_dec)

# Infer the parameters here
n_recordings = 25
n_components = mixing_matrix.shape[0]
n_sensors = int(mixing_matrix.shape[1]/n_recordings)


# initialize the sensor-wise covariance matrix to a zero-matrix
covariance = np.zeros((n_sensors,n_sensors))


# Extract the single-recording topoplots:
    # 1. initialize the matrix where you store them as empty one
topoplots = np.zeros((n_sensors, n_recordings*n_components))

# place different day analogue topoplots next to each other
# I have not tested this code for n_recordings > 2
# It should rea-arrange the topoplots in a differently shaped matrix
# keeping the order such that the topoplots relative to the same component
# on different recording are kept contiguous 
for topo in np.arange(n_components):
    component = topo * n_recordings
    for recording in np.arange(n_recordings):
        topo_rec =  mixing_matrix[topo, 
                           (recording)*n_sensors : (recording+1)*n_sensors]
        topoplots[:,component+recording] = topo_rec


# Calculate the average over the topoplots for every sensor
sensor_means = np.mean(topoplots, axis = 1)

# Center the topoplots around this average
for topo in np.arange(n_components):
    for recording in np.arange(n_recordings):
        topoplots[:, n_recordings*topo + recording] = \
            topoplots[:, n_recordings*topo + recording ] - sensor_means


# now build the sensor-wise covariance matrix statistically 
# starting from the data points in the topoplots matrix
for sensor1 in np.arange(n_sensors):
    for sensor2 in np.arange(n_sensors):
        covariance[sensor1,sensor2] += 1./(n_components*n_recordings) * \
            np.dot(topoplots[sensor1,:], topoplots[sensor2,:]) / \
                (np.std(topoplots[sensor1,:])*np.std(topoplots[sensor2,:]))
        
        

# I cannot plain invert the covariance bc det(covariance) = 0
# Let us use a pseudoinverse
# covariance is real symmetric -> diagonalizable
check_symmetric(covariance)

# e_vals, e_vects = np.linalg.eigh(covariance)

u,s,vh = np.linalg.svd(covariance)   
 

# covariance is real symmetric -> diagonalizable
check_symmetric(covariance)

u,s,vh = np.linalg.svd(covariance)   
 

# Threshold the too small singular values, especially when the data is maxfiltered
threshold = 1E-3
small =  np.where(s < threshold)[0]

# keep only eigenspaces where the are eigenvalues bigger than 
# threshold and project out components in the eigenspace of smaller eigenvalues

for index in small:
    u[:,index] = np.zeros(u.shape[0])
    vh[index,:] = np.zeros(vh.shape[1])
            
diag_inv = []
for singv in s:
    if singv >= threshold:
        diag_inv.append(1./singv) 
    else:
        diag_inv.append(0)
    
diag = np.diag(diag_inv)

# The pseudoinverse of the covariance is now our metric tensor:
distance = np.dot(np.dot(vh,diag), u)


topoplots = np.empty((n_sensors, n_recordings*n_components))

# place different day analogue topoplots next to each other
for topo in np.arange(n_components):
    component = topo * n_recordings
    for recording in np.arange(n_recordings):
        topo_rec =  mixing_matrix[topo, 
                           (recording)*n_sensors : (recording+1)*n_sensors]
        topoplots[:,component+recording] = topo_rec


# Calculate the average over the topoplots for every sensor
sensor_means = np.mean(topoplots, axis = 1)
topo_means = np.mean(topoplots, axis = 0)

# Center the sensors around this average
# for topo in np.arange(2 * n_components):
#     topoplots[:, topo] = topoplots[:, topo] - sensor_means


# Center the topoplots individually s.t. the sensor avg is 0
for topo in np.arange(2 * n_components):
    topoplots[:, topo] = topoplots[:, topo] - topo_means[topo]

# instantiate an empty array to collect our pairwise distances
distances = []

# Here we use the identity tensor as the metric tensor if the
# no_weight parameter is set to True
#no_weight = True
if no_weight:
    distance = np.identity(n_sensors)
  
# Else we use the pseudoinverse of the covariance matrix calculated above    
for topo_j in np.arange(n_components *n_recordings ):
    for topo_i in np.arange(topo_j):
        if topo_i != topo_j:
            distances.append(topo_dist(topoplots[:,topo_i], 
                                     topoplots[:,topo_j], 
                                     distance))
            
# Extra thresholding to account for computational errors
import math
for dist_index in np.arange(len(distances)):
    dist = distances[dist_index] 
    if  math.isclose(dist, 1, rel_tol=1e-05):
        distances[dist_index]= 1.0
 
# N.B. this matrix is now an array bc I did not need to plot it in my pipeline 
# Before this was a matrix, but it can be made into the same matrix as follows:
    
whitened_distance_matrix = np.zeros((n_recordings*n_components,n_recordings*n_components))

for topo_j in np.arange(n_components *n_recordings ):
    # diagonal set to 1
    whitened_distance_matrix[topo_j, topo_j] = 1
    for topo_i in np.arange(topo_j):
        if topo_i != topo_j:
            # calculate the upper triangular
            whitened_distance_matrix[topo_j, topo_i] = topo_dist(topoplots[:,topo_i], 
                                     topoplots[:,topo_j], 
                                     distance)
            # mirror it to the lower triangular
            whitened_distance_matrix[topo_i, topo_j] = whitened_distance_matrix[topo_j, topo_i] 
# quick fix
threshold = 1E-3
small =  np.where(s < threshold)[0]

# keep only eigenspaces where the are eigenvalues bigger than 
# threshold and project out components in the eigenspace of smaller eigenvalues

# use the svd matrices as projectors. Don't think I need it actually if I only
# cut the svs
# u, vh

for index in small:
    u[:,index] = np.zeros(u.shape[0])
    vh[index,:] = np.zeros(vh.shape[1])
            
diag_inv = []
for singv in s:
    if singv >= threshold:
        diag_inv.append(1./singv) 
    else:
        diag_inv.append(0)
    
diag = np.diag(diag_inv)

# The pseudoinverse of the covariance is now our metric tensor:
distance = np.dot(np.dot(vh,diag), u)



topoplots = np.empty((n_sensors, n_recordings*n_components))

# place different day analogue topoplots next to each other
for topo in np.arange(n_components):
    component = topo * n_recordings
    for recording in np.arange(n_recordings):
        topo_rec =  mixing_matrix[topo, 
                           (recording)*n_sensors : (recording+1)*n_sensors]
        topoplots[:,component+recording] = topo_rec


# Calculate the average over the topoplots for every sensor
sensor_means = np.mean(topoplots, axis = 1)
topo_means = np.mean(topoplots, axis = 0)

# Center the sensors around this average
# for topo in np.arange(2 * n_components):
#     topoplots[:, topo] = topoplots[:, topo] - sensor_means


# Center the topoplots individually s.t. the sensor avg is 0
for topo in np.arange(2 * n_components):
    topoplots[:, topo] = topoplots[:, topo] - topo_means[topo]

# instantiate an empty array to collect our pairwise distances
distances = []

# Here we use the identity tensor as the metric tensor if the
# no_weight parameter is set to True
#no_weight = True
if no_weight:
    distance = np.identity(n_sensors)
  
# Else we use the pseudoinverse of the covariance matrix calculated above    
for topo_j in np.arange(n_components *n_recordings ):
    for topo_i in np.arange(topo_j):
        if topo_i != topo_j:
            distances.append(topo_dist(topoplots[:,topo_i], 
                                     topoplots[:,topo_j], 
                                     distance))
            
# Extra thresholding to account for computational errors
import math
for dist_index in np.arange(len(distances)):
    dist = distances[dist_index] 
    if  math.isclose(dist, 1, rel_tol=1e-05):
        distances[dist_index]= 1.0
 
    
whitened_distance_matrix = np.zeros((n_recordings*n_components,n_recordings*n_components))

for topo_j in np.arange(n_components *n_recordings ):
    # diagonal set to 1
    whitened_distance_matrix[topo_j, topo_j] = 1
    for topo_i in np.arange(topo_j):
        if topo_i != topo_j:
            # calculate the upper triangular
            whitened_distance_matrix[topo_j, topo_i] = topo_dist(topoplots[:,topo_i], 
                                     topoplots[:,topo_j], 
                                     distance)
            # mirror it to the lower triangular
            whitened_distance_matrix[topo_i, topo_j] = whitened_distance_matrix[topo_j, topo_i] 