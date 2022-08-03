import numpy as np
from sklearn import metrics as sk_metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

import metrics as metrics_lib
import hungarian_assignment as hung

from clustering_methods import kmeans as kmeans_lib
from clustering_methods import kl_kmeans as kl_kmeans_lib
from clustering_methods import prob_kmeans_Mahalanobis as prob_kmeans_Mahalanobis_lib
from sklearn.mixture import GaussianMixture
from clustering_methods import kmedians as kmedians_lib
from sklearn_extra.cluster import KMedoids
from clustering_methods.kmedoids import KMedoids as KMed
from clustering_methods import kmodes as kmodes_lib
from clustering_methods import kcenters_eucl as kcenters_eucl_lib
from clustering_methods.k_hsc import Multinomial # HSC Authors library
from clustering_methods import k_dirs as k_dirs_lib
#from clustering_methods import k_sbetas_GPU as k_sbetas_lib
from clustering_methods import k_sbetas as k_sbetas_lib
from clustering_methods import sbeta_mle

import argparse
import yaml

import time

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for clustering probability simplex points")
    
    parser.add_argument('--dataset', type=str, default='SVHN_to_MNIST',
                        choices=['SVHN_to_MNIST', 'VISDA_C', 'iVISDA_Cs', 'Simu', 'iSimus'])
    
    parser.add_argument('--clustering_iters', type=int, default=25,
                        help="Clustering iterations.")

    parser.add_argument('--use_hung_assign', type=bool, default=True,
                        help="Enable Hungarian assignment. Set True only on closet-set challenges.")
    
    parser.add_argument('--hung_distance', type=str, default='eucl', 
                        choices=['eucl', 'KL'],
                        help="Centroids (or parameters) initialization strategy.")
    
    parser.add_argument('--unbiased', type=bool, default=False,
                        help="Enable unbiased formulation. " +
                             "We recommend to set True on Large-Scale imbalanced datasets.")

    parser.add_argument('--use_posterior_assignment', type=bool, default=False,
                        help="Posterior assignment.")
    
    parser.add_argument('--max_center_iters', type=int, default=100,
                        help="Iterations for estimating center (i.e. Minimum Enclosing Ball) centroids.")

    parser.add_argument('--centroids_init', type=str, default='vertices_init', 
                        choices=['kmeans_plusplus_init', 'vertices_init'],
                        help="Centroids (or parameters) initialization strategy. " +
                             "Use vertices_init only on closet-set challenges.")

    parser.add_argument('--delta', type=float, default=0.15,
                        help="delta value")

    parser.add_argument('--simu_mixt_size', type=int, default=10000,
                        help="Size of the Simulated mixture of dirichlet distributions in Simu or iSimus.")

    parser.add_argument('--cfg', type=str, default="./configs/select_methods_to_compare.yml",
                        help="optional config file")

    return parser.parse_args()


def simul_simplex_mixt(mixture_size, alphas, props):
    sample_sizes = props*mixture_size 
    for cluster_id in range(0,len(alphas)):
        diri_sample = np.random.dirichlet((alphas[cluster_id]), int(sample_sizes[cluster_id]))
        labels_diri_sample = np.ones(len(diri_sample))*cluster_id
        if cluster_id == 0:
            mixture_sample = diri_sample
            mixture_gt_labels = labels_diri_sample
        else:
            mixture_sample = np.concatenate((mixture_sample, diri_sample), axis=0)
            mixture_gt_labels = np.concatenate((mixture_gt_labels, labels_diri_sample), axis=0)
    return mixture_sample, mixture_gt_labels


def main():

    print("This code is only designed for closet-set challenges, with the", 
          " probability simplex points dimension equal to the number of", 
          " present classes.")

    # LOAD ARGS
    args = get_arguments()
    with open(args.cfg, "r") as stream:
        try:
            cfg_infos = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ## Hyper-parameters
    clustering_iters = args.clustering_iters

    use_opt = args.use_hung_assign
    hung_distance = args.hung_distance
    
    posterior_assignment = args.use_posterior_assignment # Only tested on sBeta 
    is_unbiased = args.unbiased
    
    clustering_init = args.centroids_init
    
    total_run = 1 # Number of run for each method
    if clustering_init == "kmeans_plusplus_init":
        total_run = 10
        
    max_center_iters = args.max_center_iters
    
    np.random.seed(0)
    ##    
    
    ## Load the target dataset (points must be defined on the probability simplex domain)
    print("Dataset: ", args.dataset)
    if args.dataset == 'SVHN_to_MNIST':   
        # SVHN -> MNIST softmax predictions
        GT_labels_path = "softmax_preds_datasets/SVHN_to_MNIST/target_gt_labels.txt"
        softmax_preds_path = "softmax_preds_datasets/SVHN_to_MNIST/target_softmax_predictions_ep30.txt"
        
    if args.dataset == 'VISDA_C':    
        # VISDA-C softmax predictions (pre-trained source model preds)
        GT_labels_path = "softmax_preds_datasets/VISDA_C/target_gt_labels.txt"
        softmax_preds_path = "softmax_preds_datasets/VISDA_C/target_softmax_predictions.txt"

    imb_config = 0    
    if args.dataset == 'iVISDA_Cs':    
        # iVISDA-Cs softmax predictions (pre-trained source model preds)
        total_run = 10
        GT_labels_path = "softmax_preds_datasets/iVISDA_Cs/imb_config_" + str(imb_config) + "/target_gt_labels.txt"
        softmax_preds_path = "softmax_preds_datasets/iVISDA_Cs/imb_config_" + str(imb_config) + "/target_softmax_predictions.txt"
    
    if (args.dataset == 'Simu') or (args.dataset == 'iSimus'):
        # Simulate mixture(s) of three 3-dimensional Dirichlet distributions
        total_run = 6
        alphas = np.array([[25, 5, 5],
                           [5, 7, 5],
                           [1, 1, 5]])
        
        if args.dataset == 'Simu':
            # Balanced mixture
            props = np.ones(len(alphas))/len(alphas) # Uniform (i.e. balanced)
            softmax_predictions, gt_labels = simul_simplex_mixt(args.simu_mixt_size, alphas, props)
            
        if args.dataset == 'iSimus':
            # Imbalanced mixtures
            prop1 = 0.75
            prop2 = 0.20
            prop3 = 0.05
            props = np.array([[prop1, prop2, prop3],
                              [prop1, prop3, prop2],
                              [prop2, prop1, prop3],
                              [prop2, prop3, prop1],
                              [prop3, prop1, prop2],
                              [prop3, prop2, prop1]])
            softmax_predictions, gt_labels = simul_simplex_mixt(args.simu_mixt_size, alphas, props[imb_config])
    else:    
        gt_labels = np.loadtxt(GT_labels_path)
        softmax_predictions = np.loadtxt(softmax_preds_path)
    #softmax_predictions = softmax_predictions.astype(np.single) 
    ##
    
    method_selection = {"argmax": True, 
                        "K-means": cfg_infos["METHODS"]["K_MEANS"],
                        "KL K-means": cfg_infos["METHODS"]["KL_K_MEANS"],
                        "Prob K-means Mahalanobis": cfg_infos["METHODS"]["Mah_K_MEANS"], # Not working on Simus and iSimus
                        "GMM (scikit)": cfg_infos["METHODS"]["GMM"], 
                        "K-medians": cfg_infos["METHODS"]["K_MEDIANS"], # k-medians is slow.
                        "K-medoids": cfg_infos["METHODS"]["K_MEDOIDS"], # k-medoids is slow.
                        "K-modes": cfg_infos["METHODS"]["K_MODES"],
                        "K-center": cfg_infos["METHODS"]["K_CENTERS"],
                        "HSC": cfg_infos["METHODS"]["HSC"], # HSC is extremely slow.
                        "K-Dirs": cfg_infos["METHODS"]["K_DIRS"],
                        "K-Betas": cfg_infos["METHODS"]["K_BETAS"],
                        "K-sBetas": cfg_infos["METHODS"]["K_SBETAS"],
                        "K-sBetas W": cfg_infos["METHODS"]["K_SBETAS_W"]}
    
    all_names = []
    all_scores = []
    for name, _ in method_selection.items():
        all_names.append(name)
        approach_scores = {
          "Name": name,
          "NMI": np.zeros(total_run),
          "AMI": np.zeros(total_run),
          "Acc": np.zeros(total_run),
          "BA": np.zeros(total_run),
          "CBA": np.zeros(total_run),
          "mean G-mean": np.zeros(total_run),
          "mean AUC": np.zeros(total_run),
          "mean IoU": np.zeros(total_run),
        }
        all_scores.append(approach_scores)
    
    ###
    prop_id = 0
    for run in range(0, total_run):
        print(" ")
        print("run: ", run)
        
        ## Datasets realizations
        if args.dataset == 'Simu':
            softmax_predictions, gt_labels = simul_simplex_mixt(args.simu_mixt_size, alphas, props)
            
        if imb_config >=1:
            
            if args.dataset == 'iVISDA_Cs':
                if imb_config >= 10:
                    imb_config = 0
                GT_labels_path = "softmax_preds_datasets/iVISDA_Cs/imb_config_" + str(imb_config) + "/target_gt_labels.txt"
                softmax_preds_path = "softmax_preds_datasets/iVISDA_Cs/imb_config_" + str(imb_config) + "/target_softmax_predictions.txt"
                gt_labels = np.loadtxt(GT_labels_path)
                softmax_predictions = np.loadtxt(softmax_preds_path)
            
            if args.dataset == 'iSimus':
                if imb_config >= len(props):
                    imb_config = 0
                softmax_predictions, gt_labels = simul_simplex_mixt(args.simu_mixt_size, alphas, props[imb_config])
        
        imb_config +=1
        ##
        
        number_of_classes = len(softmax_predictions[0])
        simplex_dim = len(softmax_predictions[0])           
        
        for approach_name in all_names:
            if method_selection[approach_name] == True:
                print(" ")
                print(approach_name, "...")
            method_id = np.argmax([all_scores[m]["Name"]==approach_name for m in range(0,len(all_scores))])     
            
            
            #pred_labels = np.argmax(softmax_predictions,axis=1)
            
            if method_selection[approach_name] == True and approach_name=="argmax":
                ## argmax
                pred_labels = np.argmax(softmax_predictions,axis=1)
                clustering_labels = np.argmax(softmax_predictions,axis=1)
                centroids = np.identity(simplex_dim)
                ##
            
            elif method_selection[approach_name] == True and approach_name=="K-means":    
                ## K-means
                start_time = time.time()
                clustering_labels, kmeans_dists, final_kmeans_Weights, centroids = kmeans_lib.clustering(softmax_predictions, 
                                                                            iters=clustering_iters, 
                                                                            number_of_classes=number_of_classes, 
                                                                            simplex_dim=simplex_dim,
                                                                            init_strategy=clustering_init)    
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4))
                ##
                
            elif method_selection[approach_name] == True and approach_name=="KL K-means":         
                ## KL K-means
                start_time = time.time()
                clustering_labels, kl_kmeans_dists, final_kl_kmeans_Weights, centroids = kl_kmeans_lib.clustering(softmax_predictions, 
                                                                            iters=clustering_iters, 
                                                                            number_of_classes=number_of_classes, 
                                                                            simplex_dim=simplex_dim,
                                                                            init_strategy=clustering_init)  
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4)) 
                ##
        
            elif method_selection[approach_name] == True and approach_name=="Prob K-means Mahalanobis":             
                ## Prob K-means Mahalanobis
                start_time = time.time()
                clustering_labels, pk_probs, final_pk_Weights, centroids, pk_all_sigmas = prob_kmeans_Mahalanobis_lib.clustering(softmax_predictions, 
                                                                            iters=clustering_iters, 
                                                                            number_of_classes=number_of_classes, 
                                                                            simplex_dim=simplex_dim,
                                                                            weighted_clustering=is_unbiased,
                                                                            init_strategy=clustering_init)
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4)) 
                ##  

            elif method_selection[approach_name] == True and approach_name=="GMM (scikit)":             
                ## GMM (scikit)
                start_time = time.time()
                gmm = GaussianMixture(n_components=number_of_classes, random_state=None, init_params='kmeans').fit(np.asarray(softmax_predictions))
                clustering_labels = gmm.predict(np.asarray(softmax_predictions))
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4)) 
                
                GMM_probs = gmm.predict_proba(np.asarray(softmax_predictions))
                centroids = gmm.means_ 
                ##
                
            elif method_selection[approach_name] == True and approach_name=="K-medians":                   
                ## K-medians
                start_time = time.time()
                clustering_labels, kmedians_dists, final_kmedians_Weights, centroids = kmedians_lib.clustering(softmax_predictions, 
                                                                            iters=clustering_iters, 
                                                                            number_of_classes=number_of_classes, 
                                                                            simplex_dim=simplex_dim,
                                                                            init_strategy=clustering_init)
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4))     
                ##
        
            elif method_selection[approach_name] == True and approach_name=="K-medoids":           
                ## K-medoids
                
                ### Scikit-learn implementation of k-medoids does not allow to perform vertices initialization.
                # #kmedoids = KMedoids(n_clusters=number_of_classes, random_state=0).fit(np.asarray(softmax_predictions))
                # kmedoids = KMedoids(n_clusters=number_of_classes, method='pam', init='k-medoids++', max_iter=clustering_iters, random_state=0).fit(np.asarray(softmax_predictions))
                # clustering_labels = kmedoids.labels_
                # centroids = kmedoids.cluster_centers_
                # #centroids_labels = np.asarray(hung.hungarian_labels_association(centroids, number_of_classes)).astype(int)
                # centroids_labels = hung.scipy_hung_assign(centroids, hung_distance)
                # reordered_labels = []
                # for i in range(0,len(clustering_labels)):
                #     reordered_labels.append(centroids_labels[clustering_labels[i]])
                # reordered_labels = np.asarray(reordered_labels)
                # pred_labels = reordered_labels
                ###
                
                start_time = time.time()
                kmedoids = KMed(n_cluster=number_of_classes, max_iter=clustering_iters, init_strat=clustering_init)
                kmedoids.fit(softmax_predictions.tolist())
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4))
                
                kmedoids_clusters_dict = kmedoids.clusters
                kmedoids_clusters_np = np.array(list(kmedoids_clusters_dict.items()), dtype=object)
                clustering_labels = np.ones(len(softmax_predictions))
                for k in range(0,number_of_classes):
                    clustering_labels[kmedoids_clusters_np[k][1]] = k
                clustering_labels = clustering_labels.astype(int)
                kmedoids_dict = kmedoids.medoids
                centroids = [] 
                for medoid in kmedoids_dict:
                    centroids.append(softmax_predictions[medoid])
                ##
            
            elif method_selection[approach_name] == True and approach_name=="K-modes": 
                ## K-modes
                start_time = time.time()
                clustering_labels, kmeans_dists, final_kmeans_Weights, centroids = kmodes_lib.clustering(softmax_predictions, 
                                                                            iters=clustering_iters, 
                                                                            number_of_classes=number_of_classes, 
                                                                            simplex_dim=simplex_dim,
                                                                            init_strategy=clustering_init)
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4))  
                ##
            
            elif method_selection[approach_name] == True and approach_name=="K-center":      
                ## K-center
                start_time = time.time()
                clustering_labels, kcenter_dists, final_kcenter_Weights, centroids = kcenters_eucl_lib.clustering(softmax_predictions, 
                                                                            iters=clustering_iters, 
                                                                            number_of_classes=number_of_classes, 
                                                                            simplex_dim=simplex_dim,
                                                                            max_center_iters=max_center_iters,
                                                                            weighted_clustering=is_unbiased,
                                                                            init_strategy=clustering_init)
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4))   
                ##
            
            elif method_selection[approach_name] == True and approach_name=="HSC":    
                ## HSC
                distributions = [ Multinomial(p) for p in softmax_predictions ]
                magic_number = 2020 # chosen arbitrarily
                start_time = time.time()
                clustering_labels, Hilbert_kcenter_dists, Hilbert_centroids = Multinomial.hilbert_kcenters(distributions, 
                                                                                                           number_of_classes, 
                                                                                                           seed=magic_number,
                                                                                                           max_itrs = clustering_iters,
                                                                                                           max_center_itrs=max_center_iters,
                                                                                                           init_strategy = clustering_init)
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4))  
                
                centroids = np.array( [ _d.p for _d in Hilbert_centroids ] )
                ##

            elif method_selection[approach_name] == True and approach_name == "K-Dirs":
                ## K-Dirs
                start_time = time.time()
                dir_params, clustering_labels = k_dirs_lib.clustering(softmax_predictions, number_of_classes, clustering_iters)

                total_time = time.time() - start_time
                print(approach_name, "comp time:", np.round(total_time, 4))
                centroids = []
                for cl in range(0, number_of_classes):
                    centroids.append(k_dirs_lib.modes_from_alphas(dir_params[cl]))
                ##

            elif method_selection[approach_name] == True and approach_name=="K-Betas": 
                ## K-Betas
                start_time = time.time()
                clustering_params, clustering_labels, _, _ = k_sbetas_lib.clustering(softmax_predictions, 
                                                                                 number_of_classes, 
                                                                                 simplex_dim, 
                                                                                 clustering_iters, 
                                                                                 weighted_clustering = False,
                                                                                 posterior_assignment = posterior_assignment,
                                                                                 delta = 0.,
                                                                                 init_strategy = clustering_init,
                                                                                 estim_method = "MoM")
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4))  
                
                centroids = []
                for cl in range(0, number_of_classes):
                    centroids.append([row[2] for row in clustering_params[cl]])
                ##

            elif method_selection[approach_name] == True and approach_name=="K-sBetas":     
                ## K-sBetas (0.15)
                start_time = time.time()
                clustering_params, clustering_labels, _, _ = k_sbetas_lib.clustering(softmax_predictions, 
                                                                                 number_of_classes, 
                                                                                 simplex_dim, 
                                                                                 clustering_iters, 
                                                                                 weighted_clustering = False,
                                                                                 posterior_assignment = posterior_assignment,
                                                                                 delta = args.delta,
                                                                                 init_strategy = clustering_init,
                                                                                 estim_method = "MoM",
                                                                                 lambda_constr = 165.)
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4))  
                
                centroids = []
                for cl in range(0, number_of_classes):
                    centroids.append([row[2] for row in clustering_params[cl]])
                ##
                
            elif method_selection[approach_name] == True and approach_name=="K-sBetas W":      
                ## K-sBetas (0.15) W
                start_time = time.time()
                clustering_params, clustering_labels, _, _ = k_sbetas_lib.clustering(softmax_predictions, 
                                                                                 number_of_classes, 
                                                                                 simplex_dim, 
                                                                                 clustering_iters, 
                                                                                 weighted_clustering = True,
                                                                                 posterior_assignment = posterior_assignment,
                                                                                 delta = args.delta,
                                                                                 init_strategy = clustering_init,
                                                                                 estim_method = "MoM",
                                                                                 lambda_constr = 165.)
                total_time = time.time()-start_time   
                print(approach_name, "comp time:", np.round(total_time, 4))  
                
                centroids = []
                for cl in range(0, number_of_classes):
                    centroids.append([row[2] for row in clustering_params[cl]])
                ##

            if method_selection[approach_name] == True:   
                
                ## Points assignment with optimal classes
                centroids = np.asarray(centroids)
                centroids_labels = hung.scipy_hung_assign(centroids, hung_distance)
                reordered_labels = []
                for i in range(0,len(softmax_predictions)):
                    if use_opt == True:
                        reordered_labels.append(centroids_labels[clustering_labels[i]])
                    else:
                        reordered_labels.append(np.argmax(centroids[clustering_labels[i]])) 
                if method_id!=0:
                    pred_labels = reordered_labels 
                ##                

                # Evaluation                
                all_scores[method_id]["Acc"][run] = accuracy_score(np.asarray(gt_labels, dtype = float), 
                                                                        np.asarray(pred_labels, dtype = float))
                all_scores[method_id]["NMI"][run] = sk_metrics.normalized_mutual_info_score(np.asarray(gt_labels, dtype = float), 
                                                                                              np.asarray(pred_labels, dtype = float))
                all_scores[method_id]["mean IoU"][run] = metrics_lib.compute_mean_IoU(number_of_classes, pred_labels, gt_labels)
            
    ## Display scores using latex format
    print("")
    print("Method & NMI \\\ ")
    for appraoch_id in range(0,len(all_scores)):
        
        All_Accs = 100.*all_scores[appraoch_id]["NMI"]
        meanAcc = np.mean(All_Accs)
        plus_minus = np.max(np.abs(All_Accs - meanAcc))
        
        if meanAcc!=0.:
            print(all_scores[appraoch_id]["Name"], " & ", 
                  str(np.round(meanAcc,1)),
                  "$\pm$",
                  str(np.round(plus_minus,1)),
                  " \\\ ")
        # else:
        #     print(all_scores[appraoch_id]["Name "], " & ", 
        #          "-",
        #           "$\pm$",
        #          "-", 
        #          " \\\ ")
        
    print("")
    print("Method & Acc \\\ ")
    for appraoch_id in range(0,len(all_scores)):
        
        All_Accs = 100.*all_scores[appraoch_id]["Acc"]
        meanAcc = np.mean(All_Accs)
        plus_minus = np.max(np.abs(All_Accs - meanAcc))
        
        if meanAcc!=0.:
            print(all_scores[appraoch_id]["Name"], " & ", 
                  str(np.round(meanAcc,1)),
                  "$\pm$",
                  str(np.round(plus_minus,1)),
                  " \\\ ")
        # else:
        #     print(all_scores[appraoch_id]["Name "], " & ", 
        #          "-",
        #           "$\pm$",
        #          "-", 
        #          " \\\ ")
        
    print("")
    print("Method & mean IoU \\\ ")
    for appraoch_id in range(0,len(all_scores)):
        
        All_Accs = 100.*all_scores[appraoch_id]["mean IoU"]
        meanAcc = np.mean(All_Accs)
        plus_minus = np.max(np.abs(All_Accs - meanAcc))
        
        if meanAcc!=0.:
            print(all_scores[appraoch_id]["Name"], " & ", 
                  str(np.round(meanAcc,1)),
                  "$\pm$",
                  str(np.round(plus_minus,1)),
                  " \\\ ")
        # else:
        #     print(all_scores[appraoch_id]["Name "], " & ", 
        #          "-",
        #           "$\pm$",
        #          "-", 
        #          " \\\ ")
    ##

if __name__ == '__main__':
    main()

