import sys
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import metrics as metrics_lib


def euclidean_norm(X,mu):
    return np.linalg.norm(X-mu.T, axis=1)

def KL_divergence(P_X, Q_mu, epsi):
    P_X = P_X + epsi
    Q_mu = Q_mu + epsi
    KLdiv = np.sum(P_X*np.log(P_X/Q_mu),axis=1)
    #print(len(KLdiv[0]))
    return KLdiv

def manhattan_1_norm_distance(X, cl_medians):
    distances = []
    for point in range(0,len(X)):
        manh_distance = np.sum(np.abs(X[point]-cl_medians))
        distances.append(manh_distance)
    return np.asarray(distances)


def manhattan_kmeansplusplus( distributions, k):
    
    float_epsilon = 2.220446049250313e-16
    
    random_id = np.random.choice(len(distributions))
    centers = [distributions[random_id]] 
    
    _distance = np.array(manhattan_1_norm_distance(distributions, 
                                                   np.asarray(centers[0])))

    infidx = np.isinf( _distance )
    idx = np.logical_not( infidx ) 
    _distance[infidx] = _distance[idx].max()
    
    while len(centers) < k:
        p = _distance**2
        p /= p.sum() + float_epsilon

        random_id_wrt_p = np.random.choice( len(distributions), p=p )
        centers.append( distributions[random_id_wrt_p] )

        _distance = np.minimum( _distance, manhattan_1_norm_distance(distributions, 
                                                                     np.asarray(centers[-1])))
    return np.asarray(centers)

def KL_kmeansplusplus( distributions, k):
    
    float_epsilon = 2.220446049250313e-16
    
    random_id = np.random.choice(len(distributions))
    centers = [distributions[random_id]] 
    
    _distance = np.array( KL_divergence(distributions, 
                                        np.asarray(centers[0]), 
                                        float_epsilon) )

    infidx = np.isinf( _distance )
    idx = np.logical_not( infidx ) 
    _distance[infidx] = _distance[idx].max()
    
    while len(centers) < k:
        p = _distance**2
        p /= p.sum() + float_epsilon

        random_id_wrt_p = np.random.choice( len(distributions), p=p )
        centers.append( distributions[random_id_wrt_p] )

        _distance = np.minimum( _distance, KL_divergence(distributions, 
                                                         np.asarray(centers[-1]), 
                                                         float_epsilon) )
    return np.asarray(centers)

def eucl_kmeansplusplus( distributions, k):
    
    float_epsilon = 2.220446049250313e-16
    
    random_id = np.random.choice(len(distributions))
    centers = [distributions[random_id]]
    
    _distance = np.array( euclidean_norm( distributions, np.asarray(centers[0]) ) )
    
    # tackle infinity distance
    infidx = np.isinf( _distance ) # return 0 if not infinite, and 1 if infinite
    idx = np.logical_not( infidx ) # reverses 0 and 1 
    _distance[infidx] = _distance[idx].max()
    
    while len(centers) < k:
        p = _distance**2
        p /= p.sum() + float_epsilon
        
        random_id_wrt_p = np.random.choice( len(distributions), p=p )
        centers.append( distributions[random_id_wrt_p] )
        
        _distance = np.minimum( _distance, 
                               euclidean_norm(distributions, np.asarray(centers[-1])) )
        
    return np.asarray(centers)