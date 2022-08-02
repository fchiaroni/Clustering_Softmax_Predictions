import sys
import numpy as np
from sklearn.metrics import roc_auc_score

eps = sys.float_info.epsilon


def compute_IoU(positive_class, argmax_preds, gtLabels):

    # Convert as binary classification (i.e. One vs Rest)
    bin_preds = []
    for id_preds in range(0, len(argmax_preds)):
        if argmax_preds[id_preds] == positive_class:
            bin_preds.append(1.) 
        else:
            bin_preds.append(0.)
    bin_preds = np.asarray(bin_preds)
               
    bin_gt_labels = []
    for id_preds in range(0, len(gtLabels)):
        if gtLabels[id_preds] == positive_class:
            bin_gt_labels.append(1.) 
        else:
            bin_gt_labels.append(0.)
    bin_gt_labels = np.asarray(bin_gt_labels)
    
    All_P = np.sum(bin_gt_labels)
    TP = np.sum(bin_preds*bin_gt_labels)
    FN = np.sum((1-bin_preds)*bin_gt_labels)
    FP = np.sum(bin_preds*(1-bin_gt_labels))
    TN = np.sum((1-bin_preds)*(1-bin_gt_labels))
    
    IoU = TP/(TP+FN+FP)
    
    return IoU


def compute_mean_IoU(class_number, argmax_preds, gtLabels):

    all_IoU = []
    for pos_class in range(0, class_number):
        pos_class_IoU = compute_IoU(pos_class, argmax_preds, gtLabels)
        all_IoU.append(pos_class_IoU)
    mean_IoU = np.mean(np.asarray(all_IoU))

    return mean_IoU

def compute_AUC(positive_class, softmax_preds, gtLabels):

    bin_gt_labels = []
    for id_preds in range(0, len(gtLabels)):
        if gtLabels[id_preds] == positive_class:
            bin_gt_labels.append(1.) 
        else:
            bin_gt_labels.append(0.)
    bin_gt_labels = np.asarray(bin_gt_labels)
    
    AUC_score = roc_auc_score(bin_gt_labels, softmax_preds[:,positive_class])

    return AUC_score

def compute_mean_AUC(class_number, softmax_preds, gtLabels):

    all_AUC_scores = []
    for pos_class in range(0, class_number):
        pos_class_AUC = compute_AUC(pos_class, softmax_preds, gtLabels)
        all_AUC_scores.append(pos_class_AUC)
    mean_AUC_score = np.mean(np.asarray(all_AUC_scores))

    return mean_AUC_score

def compute_gmean(positive_class, argmax_preds, gtLabels):

    # Convert as binary classification (i.e. One vs Rest)
    bin_preds = []
    for id_preds in range(0, len(argmax_preds)):
        if argmax_preds[id_preds] == positive_class:
            bin_preds.append(1.) 
        else:
            bin_preds.append(0.)
    bin_preds = np.asarray(bin_preds)
               
    bin_gt_labels = []
    for id_preds in range(0, len(gtLabels)):
        if gtLabels[id_preds] == positive_class:
            bin_gt_labels.append(1.) 
        else:
            bin_gt_labels.append(0.)
    bin_gt_labels = np.asarray(bin_gt_labels)
    
    #All_P = np.sum(bin_gt_labels)
    TP = np.sum(bin_preds*bin_gt_labels)
    FN = np.sum((1-bin_preds)*bin_gt_labels)
    FP = np.sum(bin_preds*(1-bin_gt_labels))
    TN = np.sum((1-bin_preds)*(1-bin_gt_labels))
    
    TPR = TP/(TP+FN)
    TPN = TN/(TN+FP)
    # g-mean
    g_mean = np.sqrt(TPR*TPN)

    return g_mean

def compute_mean_gmean(class_number, argmax_preds, gtLabels):

    all_gmean = []
    for pos_class in range(0, class_number):
        pos_class_gmean = compute_gmean(pos_class, argmax_preds, gtLabels)
        all_gmean.append(pos_class_gmean)
    mean_gmean = np.mean(np.asarray(all_gmean))

    return mean_gmean

def compute_CBA(positive_class, argmax_preds, gtLabels):

    # Convert as binary classification (i.e. One vs Rest)
    bin_preds = []
    for id_preds in range(0, len(argmax_preds)):
        if argmax_preds[id_preds] == positive_class:
            bin_preds.append(1.) 
        else:
            bin_preds.append(0.)
    bin_preds = np.asarray(bin_preds)
               
    bin_gt_labels = []
    for id_preds in range(0, len(gtLabels)):
        if gtLabels[id_preds] == positive_class:
            bin_gt_labels.append(1.) 
        else:
            bin_gt_labels.append(0.)
    bin_gt_labels = np.asarray(bin_gt_labels)
    
    #All_P = np.sum(bin_gt_labels)
    TP = np.sum(bin_preds*bin_gt_labels)
    FN = np.sum((1-bin_preds)*bin_gt_labels)
    FP = np.sum(bin_preds*(1-bin_gt_labels))
    #TN = np.sum((1-bin_preds)*(1-bin_gt_labels))
    pos_cls_CBA = None
    if TP+FN >= TP+FP:
        pos_cls_CBA = TP/(TP+FN)
    else:
        pos_cls_CBA = TP/(TP+FP)
    #pos_cls_CBA = int(TP)/int(np.max(TP+FN,TP+FP))

    return pos_cls_CBA

def compute_mean_CBA(class_number, argmax_preds, gtLabels):

    all_CBA = []
    for pos_class in range(0, class_number):
        pos_class_CBA = compute_CBA(pos_class, argmax_preds, gtLabels)
        all_CBA.append(pos_class_CBA)
    mean_CBA = np.mean(np.asarray(all_CBA))

    return mean_CBA