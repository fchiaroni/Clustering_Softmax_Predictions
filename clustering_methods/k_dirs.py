# Dirichlet package link: https://github.com/ericsuh/dirichlet
# fixed-point MLE developped by Thomas P. Minka


###
# 1. K-means init
# 2. While
#   2.1. optimize parameters
#   2.2. optimize assignments
# 3. Evaluation
# 4. Visualization
###

import dirichlet
import numpy as np
from numpy.linalg import norm
from sklearn import metrics
from sklearn.metrics import accuracy_score
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
    if hung_distance == "eucl":
        for centroid in centroids:
            cost.append(np.linalg.norm(centroid - one_hot_vectors, axis=1))
    elif hung_distance == "KL":
        epsi = 2.220446049250313e-16
        one_hot_vectors = one_hot_vectors + epsi
        centroids = centroids + epsi
        for centroid in centroids:
            KLdiv = np.sum(centroid * np.log(centroid / one_hot_vectors), axis=1)
            cost.append(KLdiv)
    cost = np.asarray(cost)
    row_ind, col_ind = linear_sum_assignment(cost)
    centroids_labels = col_ind
    return centroids_labels

def vertices_init(sample):
    vertices_labels = np.argmax(sample, axis=1)
    return vertices_labels

def modes_from_alphas(estim_alphas):
    K = len(estim_alphas)
    estim_modes = (estim_alphas-1)/(np.sum(estim_alphas)-K)
    return estim_modes

def clustering(x_pred, n_clusters = 2, num_iters = 25):

    for it in range(0, num_iters):
        #print("it: ", it)

        ## optimize parameters
        dirichlet_params = []
        if it==0:
            estim_labels = vertices_init(x_pred) # Parameters intitialization
        for cluster_id in range(0, n_clusters):
            # for each cluster set
            # Maximum Likelihood Estimator
            # using "fixedpoint" or "meanprecision" (faster)
            cluster_set = x_pred[estim_labels==cluster_id]
            alphas_estims = dirichlet.dirichlet.mle(np.asarray(cluster_set),
                                                    method="meanprecision")
            alphas_estims[alphas_estims < 1.] = 1.
            dirichlet_params.append(alphas_estims)
        dirichlet_params = np.asarray(dirichlet_params)
        ##

        ## optimize assignments
        y_estim = []
        for cluster_id in range(0, n_clusters):
            # Define the dirichlet function depending on alpha parameters
            dir_pdf_function_estim = dirichlet.dirichlet.pdf(dirichlet_params[cluster_id])
            y_estim.append(dir_pdf_function_estim(x_pred))
        y_estim = np.asarray(y_estim)
        estim_labels = np.argmax(y_estim.transpose(),1)
        ##

    return dirichlet_params, estim_labels


def main():
    ## Load the target dataset (points must be defined on the probability simplex domain)
    dataset_name = "SVHN_to_MNIST"
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

    n_clusters = len(init_all_softmax_predictions[0])
    num_iters = 25

    ## k-Dirs clustering
    clust_start_time = time.time()
    #
    final_dir_alphas, final_estim_labels = clustering(init_all_softmax_predictions, n_clusters, num_iters)
    #
    clust_total_time = time.time() - clust_start_time
    print("k-Dirs comp time:", np.round(clust_total_time, 4))
    ##
    ### Evaluation
    # Estimate centroids
    centroids = []
    for kcls in range(0,n_clusters):
        centroids.append(modes_from_alphas(final_dir_alphas[kcls]))
    centroids = np.asarray(centroids)

    # Proposed assignment
    centroids_labels = scipy_hung_assign(centroids)
    pred_labels = []
    for i in range(0,len(final_estim_labels)):
        pred_labels.append(centroids_labels[final_estim_labels[i]])
    pred_labels = np.asarray(pred_labels, dtype = float)
    Acc = accuracy_score(np.asarray(gt_labels, dtype = float),
                         np.asarray(pred_labels, dtype = float))
    NMI = metrics.normalized_mutual_info_score(gt_labels, pred_labels)

    print("    NMI: ", NMI)
    print("    Acc: ", Acc)
    ###

if __name__ == '__main__':
    main()