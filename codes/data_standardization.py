"""

This is an module containing several pre-processing methods for ML4DC studies on CERN data set.
 These studies are devoted to the task of abnormality detection on CERN's experimental results.
In these studies, Fedor Ratnikov is the Research Supervisor and Soroosh Shalileh is the Research Assistant.


- Prepared by: Soroosh Shalileh.

- Contact Info:
    In the case of having comments please feel free to send an email to sr.shalileh@gmail.com

"""


import numpy as np
import networkx as nx
from copy import deepcopy


def preprocess_Y(Yin, data_type):

    """
    :param Yin: numpy array, the Entity-to-feature
    :param nscf:  Dict. the dict-key is the index of categorical variable V_l in Y, and dict-value is the number of
                    sub-categorie b_v (|V_l|) in categorical feature V_l.
            Apply Z-scoring, normalization by range, and Prof. Mirkin's 3-stage normalization method,
            constant features elimination normalization.
    :return: Original entity-to-feature data matrix, Z-scored preprocessed matrix, 2-stages preprocessed matrix,
            3-stages preprocessed matrix and their corresponding relative contribution
    """

    nscf = {}
    Y_std = np.std(Yin, axis=0)
    # cnst_features = np.where(Y_std == 0)[0].tolist()  # detecting constant features, I.e std = rng(MaxMin) = 0
    # Yin_cnst_free = np.delete(Yin, obj=cnst_features, axis=1)  # New Yin s.t all constant features are removed4
    Yin_cnst_free = Yin
    print("Yin.shape:", Yin_cnst_free.shape)

    TY = np.sum(np.multiply(Yin_cnst_free, Yin_cnst_free))  # data scatter, T stands for data scatter
    TY_v = np.sum(np.multiply(Yin_cnst_free, Yin_cnst_free), axis=0)  # feature scatter
    Y_rel_cntr = TY_v / TY  # relative contribution

    mean_Y = np.mean(Yin_cnst_free, axis=0)
    std_Y = np.std(Yin_cnst_free, axis=0)

    Yz = np.divide(np.subtract(Yin_cnst_free, mean_Y), std_Y)  # Z-score

    TYz = np.sum(np.multiply(Yz, Yz))
    TYz_v = np.sum(np.multiply(Yz, Yz), axis=0)
    Yz_rel_cntr = TYz_v / TYz

    scale_min_Y = np.min(Yin_cnst_free, axis=0)
    scale_max_Y = np.max(Yin_cnst_free, axis=0)
    rng_Y = scale_max_Y - scale_min_Y

    # 3 steps normalization (Range-without follow-up division)
    Yrng = np.divide(np.subtract(Yin_cnst_free, mean_Y), rng_Y)
    TYrng = np.sum(np.multiply(Yrng, Yrng))
    TYrng_v = np.sum(np.multiply(Yrng, Yrng), axis=0)
    Yrng_rel_cntr = TYrng_v / TYrng

    # For the combination of quantitative and categorical feature  with one hot vector encoding
    # this section should be modified such that the removed columns are taken into account.
    Yrng_rs = deepcopy(Yrng)  # 3 steps normalization (Range-with follow-up division)
    for k, v in nscf.items():
        Yrng_rs[:, int(k)] = Yrng_rs[:, int(k)] / np.sqrt(int(v))  # : int(k)+ int(v)

    #     Yrng_rs = (Y_rescale - Y_mean)/ rng_Y
    TYrng_rs = np.sum(np.multiply(Yrng_rs, Yrng_rs))
    TYrng_v_rs = np.sum(np.multiply(Yrng_rs, Yrng_rs), axis=0)
    Yrng_rel_cntr_rs = TYrng_v_rs / TYrng_rs

    return Yin, Y_rel_cntr, Yz, Yz_rel_cntr, Yrng, Yrng_rel_cntr  # Yrng_rs, Yrng_rel_cntr_rs


def preprocess_P(P):

    """
    input: Adjacency matrix
    Apply Uniform, Modularity, Lapin preprocessing methods.
    return: Original Adjanceny matrix, Uniform preprocessed matrix, Modularity preprocessed matrix, and
    Lapin preprocessed matrix and their coresponding relative contribution
    """
    N, V = P.shape
    P_sum_sim = np.sum(P)
    P_ave_sim = np.sum(P) / N * (V - 1)
    cnt_rnd_interact = np.mean(P, axis=1)  # constant random interaction

    # Uniform method
    Pu = P - cnt_rnd_interact
    Pu_sum_sim = np.sum(Pu)
    Pu_ave_sim = np.sum(Pu) / N * (V - 1)

    # Modularity method (random interaction)
    P_row = np.sum(P, axis=0)
    P_col = np.sum(P, axis=1)
    P_tot = np.sum(P)
    rnd_interact = np.multiply(P_row, P_col) / P_tot  # random interaction formula
    Pm = P - rnd_interact
    Pm_sum_sim = np.sum(Pm)
    Pm_ave_sim = np.sum(Pm) / N * (V - 1)

    # Lapin (Laplacian Inverse Transform)
    # Laplacian
    """
    r, c = P.shape
    P = (P + P.T) / 2  # to warrant the symmetry
    Pr = np.sum(P, axis=1)
    D = np.diag(Pr)
    D = np.sqrt(D)
    Di = LA.pinv(D)
    L = eye(r) - Di @ P @ Di

    # pseudo-inverse transformation
    L = (L + L.T) / 2
    M, Z = LA.eig(L)  # eig-val, eig-vect
    ee = np.diag(M)
    print("ee:", ee)
    ind = list(np.nonzero(ee > 0)[0])  # indices of non-zero eigenvalues
    Zn = Z[ind, ind]
    print("Z:", Z)
    print("M:")
    print(M)
    print("ind:", ind)
    Mn = np.diag(M[ind])  # previously: Mn =  np.asarray(M[ind])
    print("Mn:", Mn)
    Mi = LA.inv(Mn)
    Pl = Zn@Mi@Zn.T
    """
    G = nx.from_numpy_array(P)
    GPl = nx.laplacian_matrix(G)
    Pl = np.asarray(GPl.todense())
    Pl_sum_sim = np.sum(Pl)
    Pl_ave_sim = np.sum(Pl) / N * (V - 1)

    return P, P_sum_sim, P_ave_sim, Pu, Pu_sum_sim, Pu_ave_sim, Pm, Pm_sum_sim, Pm_ave_sim, Pl, Pl_sum_sim, Pl_ave_sim
