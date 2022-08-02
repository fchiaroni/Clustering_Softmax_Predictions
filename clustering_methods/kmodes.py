import numpy as np
from clustering_methods import kmeans_plus_plus_init

def kmodes_vertices_init(n_clusters, n_dim):
    all_mus = None
    if n_dim==n_clusters:
        all_mus = np.identity(n_dim)
    else:
        all_mus = False
        print("vertices_init is not possible because n_dim != n_clusters.")
    return all_mus

def euclidean_norm(X,mu):
    return np.linalg.norm(X-mu.T, axis=1)

# Mode estimation using Eucl Meanshift
def eucl_meanshift(X_cl, window_size):
    prev_meanshift_point=0
    eucl_window_size = np.linalg.norm(window_size)
    meanshift_point = X_cl[np.random.choice(len(X_cl))]
    prev_meanshift_point = meanshift_point + window_size
    county=0
    while np.sum(np.isclose(meanshift_point, prev_meanshift_point))!=int(len(meanshift_point)):
        prev_meanshift_point = meanshift_point
        eucl_distances_to_mean = np.linalg.norm(X_cl-meanshift_point, axis=1)
        window_subset = X_cl[eucl_distances_to_mean<eucl_window_size]
        meanshift_point=np.mean(np.asarray(window_subset), axis=0)
        county+= 1
        if county>50:
            break
    return meanshift_point

# Mode estimation using Meanshift (better than eucl_meanshift)
def meanshift(X_cl, window_size):
    denom = np.linalg.norm(window_size)
    meanshift_point = np.mean(X_cl,axis=0)
    prev_meanshift_point = meanshift_point + window_size 
    while np.sum(np.isclose(meanshift_point, prev_meanshift_point))!=int(len(meanshift_point)):
        prev_meanshift_point = meanshift_point
        exp_weights = np.exp(-np.linalg.norm(X_cl-meanshift_point, axis=1)/denom)
        meanshift_point=np.sum(np.transpose(exp_weights*np.transpose(X_cl)),axis=0)/np.sum(exp_weights) 
    return meanshift_point

def clustering(full_set, 
                      iters=25, 
                      number_of_classes=10, 
                      simplex_dim=10,
                      init_strategy="vertices_init",
                      window_size=0.05):
    
    float_epsilon = 2.220446049250313e-16
    labels = None
    estim_weights = np.ones(number_of_classes)/number_of_classes 
    prev_assign = np.zeros(len(full_set))
    
    for it in range(0, iters):
        
        ## Parameters estimation
        all_mus = None
        if it==0:
            if init_strategy == "random_init":
                all_mus = np.array([ full_set[i] for i in np.random.randint( len(full_set), size=number_of_classes ) ])
            elif init_strategy == "vertices_init":
                all_mus = kmodes_vertices_init(number_of_classes, simplex_dim)
            elif init_strategy == "kmeans_plusplus_init":
                all_mus = kmeans_plus_plus_init.eucl_kmeansplusplus( full_set, number_of_classes)
            else: print("init_strategy: ", init_strategy, " does not exist.")
        else:
            all_mus = []
            #all_sigmas = []
            for cl_id in range(0, number_of_classes):
                cl_set = full_set[labels==cl_id]
                #modes = eucl_meanshift(cl_set, window_size) # Euclidean Meanshift Mode
                modes = meanshift(cl_set, window_size) # Meanshift Mode
                ###
                mus = np.asarray(modes)
                all_mus.append(mus)                
        all_mus = np.asarray(all_mus)
        ##
        
        ## Assignment
        all_dist_estims = []
        for cl_id in range(0, number_of_classes):
            cl_pdfs = euclidean_norm(full_set, all_mus[cl_id])
            all_dist_estims.append(cl_pdfs)
        dists = np.transpose(np.asarray(all_dist_estims)) + float_epsilon
        labels = np.argmin(dists, axis=1)
        ##
        
        # Balancing weights estimation
        for cluster_id in range(0, number_of_classes):
            estim_weights[cluster_id] = (np.asarray(labels) == cluster_id).sum()
        estim_weights = estim_weights/np.sum(estim_weights)
        
        ## check convergence
        if np.allclose( labels, prev_assign ) and it>=1:
            #print( 'k-modes converged in %d iterations' % (it+1) )
            break
        prev_assign = labels.copy()
        ##
        
    return labels, dists, estim_weights, all_mus