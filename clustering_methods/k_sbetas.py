# Requires to install pertdist package: https://pypi.org/project/pertdist/
# Unsupervisedely Weighted beta PERT clustering of univariate probability mixture distributions using posterior assignment

import numpy as np
from scipy.special import gamma
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from clustering_methods import sbeta_mle
import torch
from scipy.optimize import linear_sum_assignment
import time

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

def sBeta_vertices_init(delta, cluster_number, n_dim, a=0., c=1.):
    params =[]
    if n_dim==cluster_number:
        a = a - delta
        c = c + delta
        for cluster_id in range(0, cluster_number):
            node_params = []
            for node_id in range(0, n_dim):
                if node_id==cluster_id:
                    node_params.append([a, 1., c, 4.])
                else:
                    node_params.append([a, 0., c, 4.])
            params.append(node_params)
        params = np.asarray(params)
    else:
        params = False
        print("vertices_init is not possible because n_dim != cluster_number.")
    return params

def sBeta_plusplus_init(full_set, delta, cluster_number, n_dim, a=0., c=1.):
    all_mus = eucl_kmeansplusplus(full_set, cluster_number)
    params =[]
    a = a - delta
    c = c + delta
    for cluster_id in range(0,cluster_number):
        node_params = []
        for node_id in range(0,n_dim):
            node_params.append([a,all_mus[cluster_id][node_id],c, 4.])
        params.append(node_params)
    params = np.asarray(params)
    return params

def sBeta_init(full_set, delta, cluster_number, n_dim, init_strategy = "vertices_init", a=0., c=1.):
    if init_strategy == "vertices_init":
        init_params = sBeta_vertices_init(delta, cluster_number, n_dim, a, c)
    elif init_strategy == "kmeans_plusplus_init":
        init_params = sBeta_plusplus_init(full_set, delta, cluster_number, n_dim, a, c)
    sBeta_params = []
    a = a - delta
    c = c + delta
    for cluster_id in range(0, cluster_number):
        node_params = []
        for node_id in range(0, n_dim):
            mode = init_params[cluster_id][node_id][1]
            lambda_param = init_params[cluster_id][node_id][3]
            alpha = 1 + lambda_param*((mode-a)/(c-a))
            beta = 1 + lambda_param*((c-mode)/(c-a))
               
            node_params.append([alpha, beta, mode])
        sBeta_params.append(node_params)
    return np.asarray(sBeta_params)

def euclidean_norm(X,mu):
    return np.linalg.norm(X-mu.T, axis=1)

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

def joint_sBeta_pdf(sbeta_params, x, delta, n_nodes):
    joint_pdf_y = None
    for node_id in range(0, n_nodes):
        node_pdf_y = log_sBeta_pdf(sbeta_params[node_id][0], 
                                     sbeta_params[node_id][1], 
                                     delta,
                                     x[:,node_id])
        if node_id==0:
            joint_pdf_y = node_pdf_y
        else:
            joint_pdf_y = np.multiply(joint_pdf_y, node_pdf_y)
    return joint_pdf_y

def log_sBeta_pdf(alpha, beta, delta, x):
    a = 0. - delta
    c = 1. + delta   
    constant = (gamma(alpha)*gamma(beta))/gamma(alpha+beta)
    y=-((alpha-1)*np.log(x-a)+(beta-1)*np.log(c-x)-np.log(constant)-(alpha+beta-2)*np.log(c-a))
    return np.exp(-y)+2.220446049250313e-16

def constr(alpha, beta, delta, lambda_constr):
    
    mode = (alpha-1 + delta*(alpha-beta))/(alpha+beta-2)
    lambda_param = alpha + beta - 2
    
    if lambda_param < 1.:
        lambda_param = 1.
    if lambda_param > lambda_constr:
        lambda_param = lambda_constr
    if mode < 0.:
        mode = 0.
    if mode > 1.:
        mode = 1.
       
    alpha = 1 + lambda_param*( (mode+delta)/(1+2*delta) )
    beta  = 1 + lambda_param*( (1+delta-mode)/(1+2*delta) )     
    
    return alpha, beta

def univariate_mom(x_pred_node, delta, estim_method, lambda_constr):
    # Article version
    
    mu_delta = (np.mean(x_pred_node)+delta)/(1+2*delta)
    var = np.var(x_pred_node)
    
    alpha = (((mu_delta*(1-mu_delta)*(1+2*delta)**2)/var)-1)*mu_delta
    beta  = (((mu_delta*(1-mu_delta)*(1+2*delta)**2)/var)-1)*(1-mu_delta)
    
    alpha, beta = constr(alpha, beta, delta, lambda_constr)
    mode  = (alpha-1 + delta*(alpha-beta))/(alpha+beta-2)
        
    return alpha, beta, mode
    

def mom_method(x, delta, n_nodes, estim_method, lambda_constr):
    prms = []
    for node_id in range(0, n_nodes):
        if len(x[:,node_id])>0:
            alpha, beta, mode = univariate_mom(x[:,node_id], delta, estim_method, lambda_constr)
            prms.append([alpha, beta, mode])
    return np.asarray(prms)

def from_numpy_to_pytorch(np_array):
    return torch.from_numpy(np_array)

def from_torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()

def mle_method(x, delta, n_nodes, estim_method, lambda_constr):
    a = 0. - delta
    c = 1. + delta
    alphas, betas = sbeta_mle.multivariate_mle(from_numpy_to_pytorch(x), a, c)
    alphas = from_torch_to_numpy(alphas)
    betas = from_torch_to_numpy(betas)
    
    prms = []
    for node_id in range(0, n_nodes):
        alpha = alphas[node_id]
        beta = betas[node_id]
        
        alpha, beta = constr(alpha, beta, delta, lambda_constr)
        mode = (alpha-1 + delta*(alpha-beta))/(alpha+beta-2)
        
        prms.append([alpha, beta, mode])
        
    return np.asarray(prms)

def clustering(x_pred, 
               n_clusters = 10, 
               n_dim = 10, 
               num_iters = 25, 
               weighted_clustering = False, 
               posterior_assignment = False, 
               delta = 0.15, 
               init_strategy = "vertices_init",  # ["vertices_init", "kmeans_plusplus_init"]
               estim_method = "MoM",  # ["MoM","mle"]
               lambda_constr = 100.,
               oc = False): 
    
    # Best config.
    if oc == True:
        num_iters = 10
        delta = 0.15
        init_strategy = "vertices_init"
        estim_method = "moments_v1"
        lambda_constr = 165.
        
    # Balancing weights init
    estim_weights = np.empty(n_clusters)
    estim_weights.fill(1./n_clusters)
    #
    estim_labels = []
    prev_assign = np.zeros(len(x_pred))
    for it in range(0, num_iters):
        
        if  estim_method == "mle":
            print("it: ", it)
            
        ### Params estim
        if it==0:
            # Initialize first beta distribs parameters using k-means
            all_alphas_betas = None
            if init_strategy == "vertices_init" or init_strategy == "kmeans_plusplus_init":
                all_alphas_betas = sBeta_init(x_pred, delta, n_clusters, n_dim, init_strategy)    
                if init_strategy == "kmeans_plusplus_init" or delta != 0.15: 
                    lambda_constr = 100.
            else: print("init_strategy: ", init_strategy, " does not exist.")
        else:
            # Iterative beta distrib parameters estimation for each cluster
            all_alphas_betas = []
            for cluster_id in range(0, n_clusters):
                # for each cluster set
                np_estim_labels = np.asarray(estim_labels)
                cluster_set = x_pred[np_estim_labels==cluster_id]
                ##
                if estim_method == "MoM":
                    alphas_betas = mom_method(np.asarray(cluster_set), delta, n_dim, estim_method, lambda_constr)
                elif estim_method == "mle":
                    alphas_betas = mle_method(np.asarray(cluster_set), delta, n_dim, estim_method, lambda_constr)
                all_alphas_betas.append(alphas_betas)
                ##
        ###            
            
        ### Iterative cluster association for each example
        y_estim = []
        for cluster_id in range(0, n_clusters):
            y_estim.append(joint_sBeta_pdf(all_alphas_betas[cluster_id], x_pred, delta, n_dim))
        t_y_estim = np.transpose(np.asarray(y_estim))
        weighted_estims = np.multiply(estim_weights, t_y_estim) if weighted_clustering == True else t_y_estim
        
        estim_probs = np.transpose(np.divide(np.transpose(weighted_estims), 
                                             np.sum(weighted_estims, axis=1)))
        
        if posterior_assignment == True:
            estim_labels = list(map(lambda probs:np.random.choice(n_clusters, None, p=probs), 
                                    estim_probs))
        else:
            estim_labels = np.argmax(weighted_estims, axis=1)
        
        # Balancing weights estimation
        for cluster_id in range(0, n_clusters):
            estim_weights[cluster_id] = (np.asarray(estim_labels) == cluster_id).sum()
        estim_weights = estim_weights/np.sum(estim_weights)
        ###
        
        ## check convergence
        #if np.allclose(estim_labels, prev_assign, rtol=2.) and it>=1:
        if np.allclose(estim_labels, prev_assign) and it>=1:
            print( 'k-' + 'sBeta' + 's converged in %d iterations' % (it+1) )
            break
        prev_assign = estim_labels.copy()
        ##
        
    return all_alphas_betas, estim_labels, estim_probs, estim_weights


#############################################
def main():
    print("k-sBetas clustering algorithm")
    
    ## Load the target dataset (points must be defined on the probability simplex domain)
    dataset_name = "VISDA_C"
    print("Dataset: ", dataset_name)
    if (dataset_name == 'SVHN_to_MNIST'):   
        # SVHN -> MNIST softmax predictions
        GT_labels_path = "../softmax_preds_datasets/SVHN_to_MNIST/target_gt_labels.txt"
        softmax_preds_path = "../softmax_preds_datasets/SVHN_to_MNIST/target_softmax_predictions_ep30.txt"
        
    if (dataset_name == 'VISDA_C'):    
        # VISDA-C softmax predictions (pre-trained source model preds)
        GT_labels_path = "../softmax_preds_datasets/VISDA_C/target_gt_labels.txt"
        softmax_preds_path = "../softmax_preds_datasets/VISDA_C/target_softmax_predictions.txt"
    
    gt_labels = np.loadtxt(GT_labels_path)
    init_all_softmax_predictions = np.loadtxt(softmax_preds_path)
    ##
    
    cluster_number = len(init_all_softmax_predictions[0])
    
    start_time = time.time()
    sBeta_params,pred_labels,_,_ = clustering(init_all_softmax_predictions, 
                                              n_clusters = cluster_number, 
                                              n_dim = cluster_number, 
                                              num_iters = 25, 
                                              weighted_clustering = True, 
                                              posterior_assignment = False, 
                                              delta = 0.15,
                                              estim_method = "MoM",
                                              lambda_constr = 165.) 
    total_time = time.time()-start_time   
    print("k-sBetas comp time:", np.round(total_time, 4))  
    
    centroids = []
    for cl in range(0, cluster_number):
        centroids.append([row[2] for row in sBeta_params[cl]])
    centroids = np.asarray(centroids)
    
    # Optimal cluster-to-class assignment
    centroids_labels = scipy_hung_assign(centroids)
    
    reordered_labels = []
    for i in range(0,len(pred_labels)):
        reordered_labels.append(centroids_labels[pred_labels[i]])
    
    #reordered_probs = final_estim_probs.copy()
    #for coord in range(0,cluster_number):
    #    reordered_probs[:,centroids_labels[coord]] = final_estim_probs[:,coord]
    
    NMI = metrics.normalized_mutual_info_score(gt_labels, pred_labels)
    
    Acc = accuracy_score(np.asarray(gt_labels, dtype = float), 
                         np.asarray(reordered_labels, dtype = float))
    
    print("NMI: ", NMI)
    print("Acc: ", Acc)
#############################################

if __name__ == '__main__':
    main()