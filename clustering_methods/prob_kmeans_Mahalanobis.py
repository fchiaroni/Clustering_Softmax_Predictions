import numpy as np
from scipy.stats import multivariate_normal
from clustering_methods import kmeans_plus_plus_init

def mah_vertices_init(n_clusters, n_dim, sigma_init = 0.1):
    all_mus = None
    all_sigmas = []
    if n_dim==n_clusters:
        all_mus = np.identity(n_dim) #* 0.8 + 0.1
        for cluster_id in range(0,n_clusters):
            all_sigmas.append(np.identity(n_dim)*sigma_init)
        all_sigmas = np.asarray(all_sigmas)
    else:
        all_mus = False
        all_sigmas = False
        print("vertices_init is not possible because n_dim != n_clusters.")
    return all_mus, all_sigmas

def mah_plusplus_init(full_set, n_clusters, n_dim, sigma_init = 0.1):
    all_mus = None
    all_sigmas = []
    all_mus = kmeans_plus_plus_init.eucl_kmeansplusplus( full_set, n_clusters)
    for cluster_id in range(0,n_clusters):
        all_sigmas.append(np.identity(n_dim)*sigma_init)
    all_sigmas = np.asarray(all_sigmas)
    return all_mus, all_sigmas

def multivariate_normal_pdf(X,mu,sigma_diag):
    m = len(mu)
    X = X-mu.T
    constant = 1/((2*np.pi)**(m/2)*np.linalg.det(sigma_diag)**(0.5))
    p = constant*np.exp(-0.5*np.sum(X.dot(np.linalg.pinv(sigma_diag))*X,axis=1))
    return p

def clustering(full_set, 
                           iters=25, 
                           number_of_classes=10, 
                           simplex_dim=10, 
                           weighted_clustering = False,
                           init_strategy="vertices_init"):
    
    float_epsilon = 2.220446049250313e-16
    labels = None
    estim_weights = np.empty(number_of_classes)
    estim_weights.fill(1./number_of_classes)
    prev_assign = np.zeros(len(full_set))
    for it in range(0, iters):
        
        ## Parameters estimation
        all_mus, all_sigmas = None, None
        if it==0:
            if init_strategy == "random_init":                
                sigma_init = 0.1
                all_sigmas = []
                for cluster_id in range(0,number_of_classes):
                    all_sigmas.append(np.identity(simplex_dim)*sigma_init)
                all_sigmas = np.asarray(all_sigmas) 
                all_mus = np.array([ full_set[i] for i in np.random.randint( len(full_set), size=number_of_classes ) ])
            elif init_strategy == "vertices_init":
                all_mus, all_sigmas = mah_vertices_init(number_of_classes, simplex_dim)
            elif init_strategy == "kmeans_plusplus_init":    
                all_mus, all_sigmas = mah_plusplus_init(full_set, number_of_classes, simplex_dim)
            else: print("init_strategy: ", init_strategy, " does not exist.")
        else:
            all_mus = []
            all_sigmas = []
            for cl_id in range(0, number_of_classes):
                cl_set = full_set[labels==cl_id]
                
                mus = np.mean(cl_set,axis=0)
                sigmas = np.cov(np.transpose(cl_set))
                
                all_mus.append(mus)
                all_sigmas.append(sigmas)  
                
        all_mus = np.asarray(all_mus)
        all_sigmas = np.asarray(all_sigmas)
        ##
        
        ## Assignment
        all_pdf_estims = []
        for cl_id in range(0, number_of_classes):
            cl_pdfs = multivariate_normal_pdf(full_set, 
                                              all_mus[cl_id],
                                              all_sigmas[cl_id])
            all_pdf_estims.append(cl_pdfs)
        all_pdf_estims = np.transpose(np.asarray(all_pdf_estims)) + float_epsilon
        weighted_estims = np.multiply(estim_weights, all_pdf_estims) if weighted_clustering == True else all_pdf_estims
        probs = np.transpose(np.divide(np.transpose(weighted_estims), 
                                       np.sum(weighted_estims, axis=1)))
        labels = np.argmax(probs, axis=1)
        ##
        
        # Weights estimation
        for cluster_id in range(0, number_of_classes):
            estim_weights[cluster_id] = (np.asarray(labels) == cluster_id).sum()
        estim_weights = estim_weights/np.sum(estim_weights)
        
        ## check convergence
        if np.allclose( labels, prev_assign ) and it>=1:
            #print( 'Mahalanobis k-means converged in %d iterations' % (it+1) )
            break
        prev_assign = labels.copy()
        ##        
    return labels, probs, estim_weights, all_mus, all_sigmas