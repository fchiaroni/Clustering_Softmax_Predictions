import numpy as np
from scipy.optimize import linear_sum_assignment

def scipy_hung_assign(centroids, hung_distance="eucl"):
    centroids_number = len(centroids)
    centroids_dim = len(centroids[0])
    one_hot_vectors = None
    if centroids_number == centroids_dim:
        one_hot_vectors = np.identity(centroids_dim)
    else:
        print("we must have centroids_dim=centroids_number")
    
    cost = []
    if hung_distance=="eucl":
        for centroid in centroids:
            cost.append(np.linalg.norm(centroid-one_hot_vectors, axis=1))
    elif hung_distance=="KL":
        epsi = 2.220446049250313e-16
        one_hot_vectors = one_hot_vectors + epsi
        centroids = centroids + epsi
        for centroid in centroids:
            KLdiv = np.sum(centroid*np.log(centroid/one_hot_vectors),axis=1)
            cost.append(KLdiv)
    cost = np.asarray(cost)
    row_ind, col_ind = linear_sum_assignment(cost)    
    centroids_labels = col_ind
    return centroids_labels