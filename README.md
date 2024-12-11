# Clustering Softmax Predictions

## Updates

## Paper
### [**Simplex Clustering via sBeta with Applications to Online Adjustment of Black-Box Predictions**](https://arxiv.org/pdf/2208.00287.pdf)

If you find this code useful for your research, please cite our [paper](https://arxiv.org/pdf/2208.00287.pdf):
```
@ARTICLE{10571603,
  author={Chiaroni, Florent and Boudiaf, Malik and Mitiche, Amar and Ayed, Ismail Ben},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Simplex Clustering via sBeta With Applications to Online Adjustment of Black-Box Predictions}, 
  year={2024},
  volume={46},
  number={12},
  pages={9123-9138},
  keywords={Predictive models;Adaptation models;Closed box;Standards;Distortion measurement;Computational modeling;Histograms;Probability simplex clustering;softmax predictions;deep black-box models;pre-trained;unsupervised adaptation},
  doi={10.1109/TPAMI.2024.3418776}}
```

## Abstract
<p align="justify">
  We explore clustering the softmax predictions of deep neural networks and introduce a novel probabilistic clustering method, referred to as k-sBetas. In the general context of clustering discrete distributions, the existing methods focused on exploring distortion measures tailored to simplex data, such as the KL divergence, as alternatives to the standard Euclidean distance. We provide a general maximum a posteriori (MAP) perspective of clustering distributions, which emphasizes that the statistical models underlying the existing distortion-based methods may not be descriptive enough. Instead, we optimize a mixed-variable objective measuring the conformity of data within each cluster to the introduced sBeta density function, whose parameters are constrained and estimated jointly with binary assignment variables. Our versatile formulation approximates a variety of parametric densities for modeling simplex data, and enables to control the cluster-balance bias. This yields highly competitive performances for unsupervised adjustments of black-box model predictions in a variety of scenarios. Our code and comparisons with the existing simplex-clustering approaches along with our introduced softmax-prediction benchmarks are publicly available: https://github.com/fchiaroni/Clustering_Softmax_Predictions.
</p>
<p align="center">
  <img src="./code_illustrations/real_time_UDA_road_seg.PNG" width="450">
</p>

### Pre-requisites
* Python 3.9.4
* numpy 1.22.0
* scikit-learn 0.24.1
* scikit-learn-extra 0.2.0 (for k-medoids only)
* Pytorch 1.11.0 (for GPU-based k-sBetas only)
* CUDA 11.3 (for GPU-based k-sBetas only)

You can install all the pre-requisites using 
```bash
$ cd <root_dir>
$ pip install -r requirements.txt
```

### Datasets
The comparisons are performed on the following datasets:
- Artificial datasets on the probability simplex domain
  - Simu
  - iSimus
- Real-world softmax predictions ([`softmax_preds_datasets/`](./softmax_preds_datasets))
  - SVHN -> MNIST ([`SVHN_to_MNIST/`](./softmax_preds_datasets/SVHN_to_MNIST))
  - VISDA-C ([`VISDA_C/`](./softmax_preds_datasets/VISDA_C))
  - iVISDA-Cs ([`sVISDA_Cs/`](./softmax_preds_datasets/iVISDA_Cs))

Note that we used the source models implemented in this code https://github.com/tim-learn/SHOT to generate these real-world softmax prediction datasets.

### Implemented clustering models
The script compare_softmax_preds_clustering.py compares the following clustering alogithms:
- k-means
- GMM (scikit-learn)
- k-medians
- k-medoids
- k-modes
- KL k-means
- HSC (Hilbert Simplex Clustering)
- k-Dirs (pip install git+https://github.com/ericsuh/dirichlet.git)
- k-sBetas (proposed)

### Running the code
You can select the methods to compare by setting the config file [`./configs/select_methods_to_compare.py`](./configs/select_methods_to_compare.yml) .

Compare clustering approaches on SVHN to MNIST dataset:
```bash
$ cd <root_dir>
$ python compare_softmax_preds_clustering.py --dataset SVHN_to_MNIST
```

Compare clustering approaches on VISDA-C dataset:
```bash
$ cd <root_dir>
$ python compare_softmax_preds_clustering.py --dataset VISDA_C
```

Compare clustering approaches on highly imbalanced iVISDA-Cs datasets:
```bash
$ cd <root_dir>
$ python compare_softmax_preds_clustering.py --dataset iVISDA_Cs
```

Run only k-sBetas (GPU-based):
```bash
$ cd <root_dir>/clustering_methods
$ python k_sbetas_GPU.py --dataset SVHN_to_MNIST
$ python k_sbetas_GPU.py --dataset VISDA_C
$ python k_sbetas_GPU.py --dataset iVISDA_Cs
```

### Results

<table>
<tr><th>Table 1: Accuracy scores</th><th>Table 2: mean IoU scores</th></tr>
<tr><td>

|   (Acc)    | SVHN to MNIST | VISDA-C | iVISDA-Cs |
|------------|:-------------:|:-------:|:---------:|
|argmax | 69.8 | 53.1 | 44.2 |
|K-means | 68.9 | 47.9 | 39.3 |
|KL K-means | 75.5 | 51.2 | 41.8 |
|GMM | 67.6 | 45.7 | 37.0 |
|K-medians | 68.8 | 40.0 | 36.9 |
|K-medoids | 71.3 | 46.8 | 40.4 |
|K-modes | 71.3 | 31.1 | 29.9 |
|K-Betas | 41.2 | 24.9 | 27.2 |
| **k-sBetas** <br> (proposed) | **76.5** | **56.0** | **46.8** |

</td><td>

|   (mIoU)    | SVHN to MNIST | VISDA-C | iVISDA-Cs |
|------------|:-------------:|:-------:|:---------:|
|argmax | 54.3 | 32.5 | 22.7 |
|K-means | 55.7 | 34.6 | 24.2 |
|KL K-means | 62.1 | 37.3 | 24.9 |
|GMM | 55.0 | 30.1 | 20.4 |
|K-medians | 56.0 | 29.6 | 22.4 |
|K-medoids | 57.5 | 33.7 | 22.5 |
|K-modes | 56.2 | 24.3 | 18.4 |
|K-Betas | 25.4 | 14.0 | 14.1 |
| **k-sBetas** <br> (proposed) | **63.6** | **39.0** | **26.9** |

</td></tr> </table>

### Recommendations
- The most appropriate value for the "delta" parameter of k-sBetas may change depending on the datasets distributions. We recommend to select delta using a validation set.
- On small-scale datasets, the biased formulation for k-sBetas could be more stable.
- On large-scale imbalanced datasets, the unbiased formulation provides better results.
