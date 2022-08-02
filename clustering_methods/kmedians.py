import numpy as np
from clustering_methods import kmeans_plus_plus_init

def vertices_init(n_clusters, n_dim):
    all_meds = None
    if n_dim==n_clusters:
        all_meds = np.identity(n_dim)
    else:
        all_meds = False
        print("vertices_init is not possible because n_dim != n_clusters.")
    return all_meds

def manhattan_1_norm_distance(X, cl_medians):
    distances = []
    for point in range(0,len(X)):
        manh_distance = np.sum(np.abs(X[point]-cl_medians))
        distances.append(manh_distance)
    return np.asarray(distances)

def clustering(full_set, 
                        iters=25, 
                        number_of_classes=10, 
                        simplex_dim=10,
                        init_strategy="vertices_init"):
    
    float_epsilon = 2.220446049250313e-16
    labels = None
    estim_weights = np.ones(number_of_classes)/number_of_classes   
    prev_assign = np.zeros(len(full_set))
    for it in range(0, iters):
        
        ## Parameters estimation
        all_meds = None
        if it==0:
            if init_strategy == "random_init":
                all_meds = np.array([ full_set[i] for i in np.random.randint( len(full_set), size=number_of_classes ) ])
            elif init_strategy == "vertices_init":
                all_meds = vertices_init(number_of_classes, simplex_dim)
            elif init_strategy == "kmeans_plusplus_init":
                all_meds = kmeans_plus_plus_init.manhattan_kmeansplusplus( full_set, number_of_classes)
            else: print("init_strategy: ", init_strategy, " does not exist.")
        else:
            all_meds = []
            for cl_id in range(0, number_of_classes):
                cl_set = full_set[labels==cl_id]
                meds = np.median(cl_set,axis=0)
                all_meds.append(meds)            
        all_meds = np.asarray(all_meds)
        ##
        
        ## Assignment
        all_dist_estims = []
        for cl_id in range(0, number_of_classes):
            cl_pdfs = manhattan_1_norm_distance(full_set, all_meds[cl_id])
            all_dist_estims.append(cl_pdfs)
        dists = np.transpose(np.asarray(all_dist_estims)) + float_epsilon
        labels = np.argmin(dists, axis=1)
        ##
        
        # Weights estimation
        for cluster_id in range(0, number_of_classes):
            estim_weights[cluster_id] = (np.asarray(labels) == cluster_id).sum()
        estim_weights = estim_weights/np.sum(estim_weights)
        
        ## check convergence
        if np.allclose( labels, prev_assign ) and it>=1:
            #print( 'k-medians converged in %d iterations' % (it+1) )
            break
        prev_assign = labels.copy()
        ##
        
    return labels, dists, estim_weights, all_meds