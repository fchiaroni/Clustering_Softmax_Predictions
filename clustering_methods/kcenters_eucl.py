### Solve and finish followings
# add wheighted technique
# add unbiased formulation for imbalanced sets
# what else?
###

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from clustering_methods import kmeans_plus_plus_init

# Solve double and or missing clustering labels
def solve_double_and_or_missing_labels(X_cluster_centers):
    clusters_labels = []
    for cl in range(0,len(X_cluster_centers)):
        clusters_labels.append(np.argmax(X_cluster_centers[cl]))
    #print("Cluster labels (some duplicate and missing labels): ", clusters_labels)  
    for i in range(1, len(X_cluster_centers[0])): 
        for cl in range (0, len(X_cluster_centers)):
            for other_cl in range (0, len(X_cluster_centers)):
                if other_cl != cl:
                    if clusters_labels[cl] == clusters_labels[other_cl]:
                        if X_cluster_centers[cl][clusters_labels[cl]] < X_cluster_centers[other_cl][clusters_labels[other_cl]]:
                            clusters_labels[cl] = -1
                            # Find a relevant label
                            cl_labels_descent_sort = np.argsort(-1*X_cluster_centers[cl], axis=-1, kind='quicksort', order=None)
                            clusters_labels[cl] = cl_labels_descent_sort[i]           
    #print("Cluster labels (each cluster as its own label):     ", clusters_labels)
    return clusters_labels


def asym_var(sample):
    mean = np.mean(sample)
    both_vars = [np.var(sample[sample<mean]),np.var(sample[sample>mean])]
    largest_var = np.max(both_vars)
    return largest_var
def proposed_asym_diag_cov(x):
    asym_cov = np.identity(len(x[0]))
    for dim_coord_i in range(0, len(x[0])):
        for dim_coord_j in range(0, len(x[0])):
            if dim_coord_i==dim_coord_j:
                asym_cov[dim_coord_i][dim_coord_j] = asym_var(x[:,dim_coord_i])
    return asym_cov

def kcenter_vertices_init(n_clusters, n_dim):
    all_mus = None
    if n_dim==n_clusters:
        all_mus = np.identity(n_dim) #* 0.8 + 0.1
    else:
        all_mus = False
        print("vertices_init is not possible because n_dim != n_clusters.")
    return all_mus


def kmeans_init(sample, n_clusters=10, n_dim = 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sample)
    KM_examples_labels = kmeans.labels_
    KM_cluster_centers = kmeans.cluster_centers_
    
    all_mus = []
    for cluster_id in range(0, n_clusters):
        cluster_set = sample[KM_examples_labels==cluster_id]
        all_mus.append(np.mean(cluster_set, axis=0))      
 
    return all_mus, KM_examples_labels 

def multivariate_normal_pdf(X,mu,sigma_diag):
    m = len(mu)
    X = X-mu.T
    constant = 1/((2*np.pi)**(m/2)*np.linalg.det(sigma_diag)**(0.5))
    p = constant*np.exp(-0.5*np.sum(X.dot(np.linalg.pinv(sigma_diag))*X,axis=1))
    return p
def multivariate_Gibbs_prob_euclidean(X,mu):
    return np.exp(-np.linalg.norm(X-mu.T, axis=1))

def euclidean_norm(X,mu):
    return np.linalg.norm(X-mu.T, axis=1)

def KL_divergence(P_X, Q_mu, epsi):
    P_X = P_X + epsi
    Q_mu = Q_mu + epsi
    KLdiv = np.sum(P_X*np.log(P_X/Q_mu),axis=1)
    #print(len(KLdiv[0]))
    return KLdiv

def cluster_center_estim(X_cl, max_center_iters, epsi):
    # Maybe change or at least check the random selection...
    #cluster_center = np.random.exponential( scale = 1.0, 
    #                                       size = len(X_cl[0]) )
    #cluster_center = cluster_center / np.sum(cluster_center)
    cluster_center = X_cl[np.random.choice(len(X_cl))]
    #cluster_center = np.ones(len(X_cl[0]))*(1/len(X_cl[0]))
    for center_iter in range(1, max_center_iters+1):
        far_id = np.argmax(np.linalg.norm(X_cl-cluster_center, axis=1))
        #far_id = np.argmax(np.sum((X_cl+epsi)*np.log((X_cl+epsi)/(cluster_center+epsi)),axis=1))
        
        far_point = X_cl[far_id]
        r = 1/(center_iter+1)
        cluster_center = (1-r)*cluster_center + r*far_point
    return cluster_center

def clustering(full_set, 
                           iters=25, 
                           number_of_classes=10, 
                           simplex_dim=10,
                           max_center_iters = 100,
                           weighted_clustering = False,
                           init_strategy="vertices_init"):
    float_epsilon = 2.220446049250313e-16
    labels = None
    # Balancing weights init
    #estim_weights = np.empty(number_of_classes)
    #estim_weights.fill(1./number_of_classes)
    estim_weights = np.ones(number_of_classes)/number_of_classes
    #    
    prev_assign = np.zeros(len(full_set))
    for it in range(0, iters):
        
        ## Parameters estimation
        all_mus = None
        if it==0:
            if init_strategy == "random_init":
                all_mus = np.array([ full_set[i] for i in np.random.randint( len(full_set), size=number_of_classes ) ])
            elif init_strategy == "vertices_init":
                all_mus = kcenter_vertices_init(number_of_classes, simplex_dim)
            elif init_strategy == "kmeans_plusplus_init":
                all_mus = kmeans_plus_plus_init.eucl_kmeansplusplus( full_set, number_of_classes)
            elif init_strategy == "kmeans_init":
                all_mus, _ = kmeans_init(full_set, number_of_classes, simplex_dim)
            else: print("init_strategy: ", init_strategy, " does not exist.")

        else:
            all_mus = []
            #all_sigmas = []
            for cl_id in range(0, number_of_classes):
                cl_set = full_set[labels==cl_id]
                #mus = np.mean(cl_set,axis=0) 
                mus = cluster_center_estim(cl_set, max_center_iters, float_epsilon)
                all_mus.append(mus)
                #all_sigmas.append(sigmas)                 
        all_mus = np.asarray(all_mus)
        #all_sigmas = np.asarray(all_sigmas)
        ##
        
        ## Assignment
        all_dist_estims = []
        for cl_id in range(0, number_of_classes):
            #cl_pdfs = multivariate_normal.pdf(full_set, mean=all_mus[cl_id], cov=all_sigmas[cl_id])
            ###
            #cl_pdfs = multivariate_normal_pdf(full_set, 
            #                                  all_mus[cl_id],
            #                                  all_sigmas[cl_id]#*np.identity(simplex_dim)
            #                                  )
            #cl_pdfs = multivariate_Gibbs_prob_euclidean(full_set, 
            #                                            all_mus[cl_id])
            cl_pdfs = euclidean_norm(full_set, all_mus[cl_id])
            #cl_pdfs = KL_divergence(full_set, all_mus[cl_id], float_epsilon)
            ###
            # ##
            # if i>0:
                # #mean_softmax = np.mean(full_set[labels==cl_id],axis=0)
                # #print(-np.sum(mean_softmax*np.log(mean_softmax)))
                # #cl_pdfs = cl_pdfs -mean_softmax[cl_id]*np.log(mean_softmax[cl_id])
                
                # #mean_softmax = np.mean(full_set[:,cl_id])
                # mean_softmax = np.mean(full_set[labels==cl_id][:,cl_id])
                # print("1: ", mean_softmax)
                # #print(-mean_softmax*np.log(mean_softmax))
                # #cl_pdfs = cl_pdfs -mean_softmax*np.log(mean_softmax)
                
                # mean_softmax = len(full_set[labels==cl_id])/len(full_set)
                # #print("len(full_set[labels==cl_id]):", len(full_set[labels==cl_id]))
                # #print("len(full_set):", len(full_set))
                # print("2: ", mean_softmax)
                # #cl_pdfs = cl_pdfs -len(full_set[labels==cl_id])*np.log(mean_softmax)
            # ##
            all_dist_estims.append(cl_pdfs)
        all_dist_estims = np.transpose(np.asarray(all_dist_estims)) + float_epsilon
        unbias = -np.log(estim_weights)/np.sum(-np.log(estim_weights))
        dists = all_dist_estims - unbias if weighted_clustering == True else all_dist_estims
        labels = np.argmin(dists, axis=1)
        #
        #print("dists.shape: ", dists.shape)
        #labels = np.argmax(np.log(dists) - mean_probs*np.log(mean_probs), axis=1)
        #
        ##
        
        # Balancing weights estimation
        for cluster_id in range(0, number_of_classes):
            estim_weights[cluster_id] = (np.asarray(labels) == cluster_id).sum()
        estim_weights = estim_weights/np.sum(estim_weights)
        #estim_weights = np.mean(dists, axis=0)
        #if it == (iters-1): 
        #    print("estim_weights: ", estim_weights)
        #
        #estim_weights = (estim_weights+(np.ones(number_of_classes)/number_of_classes))/2
        
        ### check convergence
        if np.allclose( labels, prev_assign ) and it>=1:
            print( 'k-center converged in %d iterations' % (it+1) )
            break
        prev_assign = labels.copy()
        ###        
    return labels, dists, estim_weights, all_mus