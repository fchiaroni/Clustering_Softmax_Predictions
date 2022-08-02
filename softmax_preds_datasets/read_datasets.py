import numpy as np
import argparse


def dataset_caracteristics(ground_truth_labels, predictions):
    number_of_classes = len(predictions[0])
    imb_Ns = np.zeros(number_of_classes, dtype=int)
    for k in range(0,number_of_classes):
        imb_Ns[k] = np.sum([ground_truth_labels==k])
    classes_props = imb_Ns/np.sum(imb_Ns)
    print("    -Dataset length: ", len(predictions))
    print("    -Number of classes: ", number_of_classes)
    print("    -Softmax predictions shape: ", np.shape(predictions))
    print("    -Examples per class: ", imb_Ns)
    print("    -Classes proportions: ", np.round(classes_props, 3))

# def unison_shuffled(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     return a[p], b[p]

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--dataset_name', type=str, default='all', 
                        choices=['SVHN_to_MNIST', 'VISDA_C', 'iVISDA_Cs', 'all'])
    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args.dataset_name)
    
    
    if (args.dataset_name == 'SVHN_to_MNIST') or (args.dataset_name == 'all'):
        ## SVHN_to_MNIST
        SVHN_to_MNIST_GT_labels_path = "SVHN_to_MNIST/target_gt_labels.txt"
        SVHN_to_MNIST_softmax_preds_path = "SVHN_to_MNIST/target_softmax_predictions_ep30.txt"
        gt_labels = np.loadtxt(SVHN_to_MNIST_GT_labels_path)
        softmax_predictions = np.loadtxt(SVHN_to_MNIST_softmax_preds_path)
        print(" ")
        print("SVHN_to_MNIST")
        dataset_caracteristics(gt_labels, softmax_predictions)
        ##
    
    if (args.dataset_name == 'VISDA_C') or (args.dataset_name == 'all'):
        ## VISDA-C
        VISDA_C_GT_labels_path = "VISDA_C/target_gt_labels.txt"
        VISDA_C_softmax_preds_path = "VISDA_C/target_softmax_predictions.txt"
        gt_labels = np.loadtxt(VISDA_C_GT_labels_path)
        softmax_predictions = np.loadtxt(VISDA_C_softmax_preds_path)
        print(" ")
        print("VISDA-C")
        dataset_caracteristics(gt_labels, softmax_predictions)
        ##

    if (args.dataset_name == 'iVISDA_Cs') or (args.dataset_name == 'all'):        
        ## iVISDA-Cs
        for imb_config in range(0,10):
            iVISDA_Cs_GT_labels_path = "iVISDA_Cs/imb_config_" + str(imb_config) + "/target_gt_labels.txt"
            iVISDA_Cs_softmax_preds_path = "iVISDA_Cs/imb_config_" + str(imb_config) + "/target_softmax_predictions.txt"
            gt_labels = np.loadtxt(iVISDA_Cs_GT_labels_path)
            softmax_predictions = np.loadtxt(iVISDA_Cs_softmax_preds_path)
            # softmax_predictions, gt_labels = unison_shuffled(softmax_predictions, gt_labels)    
            print(" ")
            print(str(imb_config), "iVISDA-Cs")
            dataset_caracteristics(gt_labels, softmax_predictions)
        ##


if __name__ == '__main__':
    main()
