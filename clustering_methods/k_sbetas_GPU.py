# k-sBetas for closet-set prediction adjustment

import numpy as np
from scipy.special import gamma
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import accuracy_score
import torch
from scipy.optimize import linear_sum_assignment
import time
import torch
import sbeta_mle
import os
import argparse

def from_list_of_torch_to_torch(lst):
    return torch.stack(lst)
def from_numpy_to_torch(np_array, torch_device):
    return torch.from_numpy(np_array).to(torch_device)

def from_torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()

def torch_gamma(torch_tensor):
    return torch.exp(torch.lgamma(torch_tensor))

def torch_log_sBeta_pdf(pdf_params, simplices, delta):
    a = 0. - delta
    c = 1. + delta
    alpha_s, beta_s = pdf_params[:, :, 0], pdf_params[:, :, 1]
    constant_s = (torch_gamma(alpha_s) * torch_gamma(beta_s)) / torch_gamma(alpha_s + beta_s)
    y_s = (alpha_s - 1) * torch.log(simplices.unsqueeze(1) - a) + (beta_s - 1) * torch.log(
        c - simplices.unsqueeze(1)) - torch.log(constant_s) - (alpha_s + beta_s - 2) * torch.log(c - a)
    joint_y = torch.sum(-y_s, 2)
    return joint_y

def torch_gamma_(torch_tensor, inplace: bool = False):
    if inplace:
        return torch_tensor.lgamma_().exp_()
    else:
        return torch.lgamma(torch_tensor).exp_()

def torch_log_sBeta_pdf_(pdf_params, simplices, delta):
    a = 0. - delta
    c = 1. + delta
    alpha_s, beta_s = pdf_params[:, :, 0], pdf_params[:, :, 1]
    intermediate = torch_gamma_(beta_s)
    constant_s = alpha_s.lgamma().add_(beta_s.lgamma()).sub_(
        torch.add(alpha_s, beta_s, out=intermediate).lgamma_()).exp_()
    diff_simplices = simplices - a
    y_s = diff_simplices.unsqueeze(1).log_() * torch.sub(alpha_s, 1, out=intermediate)
    y_s.addcmul_(torch.sub(c, simplices, out=diff_simplices).unsqueeze(1).log_(),
                 torch.sub(beta_s, 1, out=intermediate))
    y_s.sub_(constant_s.log_())
    y_s.addcmul_(torch.add(alpha_s, beta_s, out=intermediate).sub_(2), (c - a).log_(), value=-1)
    joint_y = torch.sum(y_s, 2, out=diff_simplices).neg_()
    return joint_y

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

def torch_sBeta_vertices_init(cluster_number, n_dim, torch_device):
    params = []
    if n_dim == cluster_number:
        modes = torch.diag(torch.ones(n_dim, device=torch_device, dtype=torch.double))
        lambda_params = modes * 4. # value 4 chosen arbitrarily
    else:
        params = False
        print("vertices_init is not possible because n_dim != cluster_number.")
    return modes, lambda_params

def eucl_kmeansplusplus(distributions, k):
    float_epsilon = 2.220446049250313e-16

    random_id = np.random.choice(len(distributions))
    centers = [distributions[random_id]]

    _distance = np.array(euclidean_norm(distributions, np.asarray(centers[0])))

    # tackle infinity distance
    infidx = np.isinf(_distance)  # return 0 if not infinite, and 1 if infinite
    idx = np.logical_not(infidx)  # reverses 0 and 1
    _distance[infidx] = _distance[idx].max()

    while len(centers) < k:
        p = _distance ** 2
        p /= p.sum() + float_epsilon

        random_id_wrt_p = np.random.choice(len(distributions), p=p)
        centers.append(distributions[random_id_wrt_p])

        _distance = np.minimum(_distance,
                               euclidean_norm(distributions, np.asarray(centers[-1])))

    return np.asarray(centers)

def euclidean_norm(X, mu):
    return np.linalg.norm(X - mu.T, axis=1)

def torch_opt_sBeta_init(torch_full_set, delta, cluster_number, n_dim, torch_device, init_strategy="vertices_init"):
    if init_strategy == "vertices_init":
        #init_params = sBeta_vertices_init(delta, cluster_number, n_dim)
        mode, lambda_param = torch_sBeta_vertices_init(cluster_number, n_dim, torch_device)
    elif init_strategy == "kmeans_plusplus_init":
        full_set = from_torch_to_numpy(torch_full_set)
        all_mus = eucl_kmeansplusplus(full_set, cluster_number)
        mode = from_numpy_to_torch(all_mus, torch_device)
        lambda_param = mode * 4.
    alpha = 1 + lambda_param * ((mode + delta) / (1. + 2. * delta))
    beta = 1 + lambda_param * ((1. + delta - mode) / (1. + 2. * delta))
    stacked_params = torch.stack((alpha, beta, mode), 2)
    return stacked_params

def torch_constr(alpha, beta, delta, lambda_constr):
    ## torch_constr
    mode = (alpha - 1 + delta * (alpha - beta)) / (alpha + beta - 2)
    lambda_param = alpha + beta - 2

    # Constraints
    lambda_param = torch.clamp(lambda_param, min=1., max=lambda_constr)
    mode = torch.clamp(mode, min=0., max=1.)

    # Update
    constr_alpha = 1 + lambda_param * ((mode + delta) / (1 + 2 * delta))
    constr_beta = 1 + lambda_param * ((1 + delta - mode) / (1 + 2 * delta))
    ##
    return constr_alpha, constr_beta

def torch_opt_mom_method(mu, var, delta, lambda_constr):
    mu_delta = (mu + delta) / (1 + 2 * delta)
    alpha = (((mu_delta * (1 - mu_delta) * (1 + 2 * delta) ** 2) / var) - 1) * mu_delta
    beta = (((mu_delta * (1 - mu_delta) * (1 + 2 * delta) ** 2) / var) - 1) * (1 - mu_delta)

    alpha, beta = torch_constr(alpha, beta, delta, lambda_constr)

    mode = (alpha - 1 + delta * (alpha - beta)) / (alpha + beta - 2)

    return torch.stack((alpha, beta, mode), 1)

def torch_opt_mle_method(torch_set_k, delta, lambda_constr, device_name):
    alpha, beta = sbeta_mle.multivariate_mle(Y=torch_set_k,
                                             a=0. - delta,
                                             c=1. + delta,
                                             device_name=device_name)

    alpha, beta = torch_constr(alpha, beta, delta, lambda_constr)

    mode = (alpha - 1 + delta * (alpha - beta)) / (alpha + beta - 2)

    return torch.stack((alpha, beta, mode), 1)


def clustering(torch_x_pred,
               n_clusters=10,
               n_dim=10,
               num_iters=25,
               weighted_clustering=False,
               posterior_assignment=False,
               delta=0.15,
               init_strategy="vertices_init",  # ["vertices_init", "kmeans_plusplus_init"]
               estim_method="MoM",  # ["MoM","mle"]
               lambda_constr=100.,
               oc=False,
               device_name='cuda:0'):
    epsi = 2.220446049250313e-16
    torch_device = torch.device(device_name)

    # Best config.
    if oc == True:
        num_iters = 10
        delta = 0.15
        init_strategy = "vertices_init"
        estim_method = "moments_v1"
        lambda_constr = 165.

    # Balancing weights init
    torch_estim_weights = torch.ones(n_clusters, device=torch_device) / n_clusters
    #
    estim_labels = []
    prev_assign = torch.zeros(len(torch_x_pred), dtype=torch.long, device=torch_device)
    torch_params = torch.ones((n_clusters, n_clusters, 3), dtype=torch.double, device=torch_device)

    for it in range(0, num_iters):
        if estim_method == "mle":
            print("it: ", it)  # Because mle is slow

        ### Params estimation
        if it == 0:
            # Initialize parameters
            if init_strategy == "vertices_init" or init_strategy == "kmeans_plusplus_init":
                torch_params = torch_opt_sBeta_init(torch_x_pred, delta, n_clusters, n_dim, torch_device, init_strategy)
                # print(torch_params.shape)
                if init_strategy == "kmeans_plusplus_init" or delta != 0.15:
                    lambda_constr = 100.
            else:
                print("init_strategy: ", init_strategy, " does not exist.")
        else:
            # Parameters estimation
            if estim_method == "MoM":
                for cluster_id in range(0, n_clusters):
                    mask = torch_estim_labels.eq(cluster_id)
                    sum_mask = torch.sum(mask)
                    means_k = torch.sum((mask * torch_x_pred.t()), dim=1) / sum_mask
                    vars_k = (torch.sum((mask * (torch_x_pred ** 2).t()), dim=1) / sum_mask) - means_k ** 2

                    means_k = torch.nan_to_num(means_k, 1 / n_dim)  # If no point in the cluster
                    vars_k = torch.nan_to_num(vars_k, 0.5)  # If no point in the cluster

                    if cluster_id == 0:
                        all_means = means_k
                        all_vars = vars_k
                    else:
                        all_means = torch.cat((all_means, means_k), 0)
                        all_vars = torch.cat((all_vars, vars_k), 0)
                stacked_params = torch_opt_mom_method(all_means, all_vars,
                                                      delta, lambda_constr)
                splitted_stacked_params = torch.split(stacked_params,
                                                      n_dim, dim=0)
                torch_params = torch.stack(splitted_stacked_params)

            elif estim_method == "mle":
                for cluster_id in range(0, n_clusters):
                    mask = torch_estim_labels.eq(cluster_id)
                    torch_cluster_set = torch_x_pred[mask]
                    torch_delta = torch.tensor(delta, device=torch_device)
                    torch_cluster_params = torch_opt_mle_method(torch_cluster_set,
                                                                torch_delta,
                                                                lambda_constr,
                                                                device_name)
                    torch_params[cluster_id] = torch_cluster_params
        ###

        ### Iterative cluster association for each example
        y_estim = torch_log_sBeta_pdf(torch_params, torch_x_pred,
                                       torch.tensor(delta, dtype=torch.double, device=torch_device))
        weighted_estims = torch.sub(y_estim, torch.log(torch_estim_weights)) if weighted_clustering == True else y_estim
        torch_estim_probs = torch.softmax(-weighted_estims, dim=1)

        if posterior_assignment == True and (it != (num_iters - 1)):
            estim_probs = from_torch_to_numpy(torch_estim_probs)
            estim_labels = list(map(lambda probs: np.random.choice(n_clusters, None, p=probs),
                                    estim_probs))
        else:
            torch_estim_labels = torch.argmin(weighted_estims, dim=1)

        # Weights estimation
        torch_labels_count = torch.bincount(torch_estim_labels)
        torch_estim_weights = torch.divide(torch_labels_count,
                                           torch.sum(torch_labels_count))

        # check convergence
        # if np.allclose( estim_labels, prev_assign, atol=10.) and it>=1: # use atol=10 for early stop
        if torch.allclose(torch_estim_labels, prev_assign) and it >= 1:
            print('k-' + 'sBeta' + 's converged in %d iterations' % (it + 1))
            break
        prev_assign = torch_estim_labels.clone()
        ###

    return torch_params, torch_estim_labels, torch_estim_probs, torch_estim_weights


def pred_only(torch_x_pred,
              torch_trained_params,
              torch_estim_weights,
              n_clusters=10,
              n_dim=10,
              weighted_clustering=False,
              delta=0.15,
              device_name='cuda:0'):
    torch_device = torch.device(device_name)

    # Inference
    y_estim = torch_log_sBeta_pdf(torch_trained_params, torch_x_pred,
                                   torch.tensor(delta, dtype=torch.double, device=torch_device))
    weighted_estims = torch.sub(y_estim, torch.log(torch_estim_weights)) if weighted_clustering == True else y_estim
    torch_estim_probs = torch.softmax(-weighted_estims, dim=1)
    torch_estim_labels = torch.argmin(weighted_estims, dim=1)

    return torch_estim_labels, torch_estim_probs


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for clustering probability simplex points")
    
    parser.add_argument('--dataset', type=str, default='SVHN_to_MNIST',
                        choices=['SVHN_to_MNIST', 'VISDA_C', 'iVISDA_Cs', 'Simu', 'iSimus'])
    
    parser.add_argument('--device_name', type=str, default='cuda:0',
                        choices=['cpu', 'cuda:0', 'cuda:1'])
    
    parser.add_argument('--clustering_iters', type=int, default=25,
                        help="Clustering iterations.")
    
    parser.add_argument('--unbiased', type=bool, default=True,
                        help="Enable unbiased formulation. " +
                             "We recommend to set True on Large-Scale imbalanced datasets.")

    parser.add_argument('--centroids_init', type=str, default='vertices_init', 
                        choices=['kmeans_plusplus_init', 'vertices_init'],
                        help="Centroids (or parameters) initialization strategy. " +
                             "Use vertices_init only on closet-set challenges.")

    parser.add_argument('--delta', type=float, default=0.15,
                        help="delta value")

    return parser.parse_args()

def main():
    print("k-sBetas clustering algorithm")
    args = get_arguments()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used

    ## Load the target dataset (points must be defined on the probability simplex domain)
    print("Dataset: ", args.dataset)
    if (args.dataset == 'SVHN_to_MNIST'):
        # SVHN -> MNIST softmax predictions
        GT_labels_path = "../softmax_preds_datasets/SVHN_to_MNIST/target_gt_labels.txt"
        softmax_preds_path = "../softmax_preds_datasets/SVHN_to_MNIST/target_softmax_predictions_ep30.txt"
    if (args.dataset == 'VISDA_C'):
        # VISDA-C softmax predictions
        GT_labels_path = "../softmax_preds_datasets/VISDA_C/target_gt_labels.txt"
        softmax_preds_path = "../softmax_preds_datasets/VISDA_C/target_softmax_predictions.txt"
    gt_labels = np.loadtxt(GT_labels_path)
    init_all_softmax_predictions = np.loadtxt(softmax_preds_path)
    ##

    class_number = int(np.max(gt_labels)+1)
    print(init_all_softmax_predictions.shape[0])
    print(init_all_softmax_predictions.shape[1])
    if class_number != init_all_softmax_predictions.shape[1]:
        raise Exception("The number of present classes must be equal to softmax preds dimension! (closed-set setting)")

    ## Conversions npy to pytorch
    torch_device = torch.device(args.device_name)
    torch_x_pred = from_numpy_to_torch(init_all_softmax_predictions, torch_device)
    ##

    ## Clustering
    clust_start_time = time.time()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    #
    torch_sBeta_params, torch_pred_labels, _, torch_estim_weights = clustering(torch_x_pred,
                                                                               n_clusters=class_number,
                                                                               n_dim=class_number,
                                                                               num_iters=args.clustering_iters,
                                                                               weighted_clustering=args.unbiased,
                                                                               delta=args.delta,
                                                                               estim_method="MoM",
                                                                               lambda_constr=165.,
                                                                               device_name=args.device_name,
                                                                               init_strategy=args.centroids_init)
    #
    end.record()
    torch.cuda.synchronize()
    cuda_clust_total_time = (start.elapsed_time(end)) / 1000
    cpu_clust_total_time = time.time() - clust_start_time
    if args.device_name == 'cpu':
        print("CPU clustering comp time:", np.round(cpu_clust_total_time, 4))
    else:
        print("GPU clustering comp time:", np.round(cuda_clust_total_time, 4))
    ##

    ## Inference only
    inf_start_time = time.time()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    #
    torch_pred_labels, _ = pred_only(torch_x_pred,
                                     torch_sBeta_params,
                                     torch_estim_weights,
                                     n_clusters=class_number,
                                     n_dim=class_number,
                                     weighted_clustering=args.unbiased,
                                     delta=args.delta,
                                     device_name=args.device_name)
    #
    end.record()
    torch.cuda.synchronize()
    cuda_inf_total_time = (start.elapsed_time(end)) / 1000
    cpu_inf_total_time = time.time() - inf_start_time
    if args.device_name == 'cpu':
        print("CPU inference comp time:", np.round(cpu_inf_total_time, 4))
    else:
        print("GPU inference comp time:", np.round(cuda_inf_total_time, 4))
    ##

    ##
    pred_labels = from_torch_to_numpy(torch_pred_labels)
    sBeta_params = from_torch_to_numpy(torch_sBeta_params)
    ##

    centroids = []
    for cl in range(0, class_number):
        centroids.append([row[2] for row in sBeta_params[cl]])
    centroids = np.asarray(centroids)

    # Optimal cluster-to-class assignment (closed-set)
    centroids_labels = scipy_hung_assign(centroids)

    reordered_labels = []
    for i in range(0, len(pred_labels)):
        reordered_labels.append(centroids_labels[pred_labels[i]])

    NMI = metrics.normalized_mutual_info_score(gt_labels, pred_labels)

    Acc = accuracy_score(np.asarray(gt_labels, dtype=float),
                         np.asarray(reordered_labels, dtype=float))
    print("NMI: ", NMI)
    print("Acc: ", Acc)

if __name__ == '__main__':
    main()