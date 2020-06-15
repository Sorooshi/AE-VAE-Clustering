import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

CASES = ['original', 'reconstructed', 'latent']


def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return metrics.roc_auc_score(y_test, y_pred, average=average)


def evaluation_with_clustering_metrics(kmeans_ms, agg_ms, gmm_ms, GT_ms, case):

    recm_k, recm_a, recm_g = {}, {}, {}  # Results Evaluation with Clustering Metrics

    case_ = CASES[case]

    print(" ")
    print("Results on the " + case_ + " Variable")

    print("\t", "  p", "  q", " a/e", "\t",
          "  K-ARI  ", "  K-NMI ", "\t",
          "  A-ARI  ", "  A-NMI ", "\t",
          "  G-ARI  ", "  G-NMI  ",
          )

    print(" \t", " \t", " \t",
          " Ave ", " std  ", " Ave ", "std ",
          " Ave ", " std  ", " Ave ", "std ",
          " Ave ", " std  ", " Ave ", "std"
          )

    for setting, results in kmeans_ms.items():

        ARI_k, NMI_k, ARI_a, NMI_a, ARI_g, NMI_g = [], [], [], [], [], []
        for repeat, result in results.items():
            gt = GT_ms[setting][repeat]
            lp_k = result[case]
            lp_a = agg_ms[setting][repeat][case]
            lp_g = gmm_ms[setting][repeat][case]
            ARI_k.append(metrics.adjusted_rand_score(gt, lp_k))
            NMI_k.append(metrics.normalized_mutual_info_score(gt, lp_k, average_method='max'))

            ARI_a.append(metrics.adjusted_rand_score(gt, lp_a))
            NMI_a.append(metrics.normalized_mutual_info_score(gt, lp_a, average_method='max'))

            ARI_g.append(metrics.adjusted_rand_score(gt, lp_g))
            NMI_g.append(metrics.normalized_mutual_info_score(gt, lp_g, average_method='max'))

        ari_ave_k = np.mean(np.asarray(ARI_k), axis=0)
        ari_std_k = np.std(np.asarray(ARI_k), axis=0)
        nmi_ave_k = np.mean(np.asarray(NMI_k), axis=0)
        nmi_std_k = np.std(np.asarray(NMI_k), axis=0)
        recm_k[setting] = [ari_ave_k, ari_std_k, nmi_ave_k, nmi_std_k]  # Evaluation Results Clustering Kmeans
        # recm_k = [ari_ave_k, ari_std_k, nmi_ave_k, nmi_std_k]  # Evaluation Results Clustering Kmeans

        ari_ave_a = np.mean(np.asarray(ARI_a), axis=0)
        ari_std_a = np.std(np.asarray(ARI_a), axis=0)
        nmi_ave_a = np.mean(np.asarray(NMI_a), axis=0)
        nmi_std_a = np.std(np.asarray(NMI_a), axis=0)
        recm_a[setting] = [ari_ave_a, ari_std_a, nmi_ave_a, nmi_std_a]  # Evaluation Results Clustering Agglomerative
        # recm_a = [ari_ave_a, ari_std_a, nmi_ave_a, nmi_std_a]  # Evaluation Results Clustering Agglomerative

        ari_ave_g = np.mean(np.asarray(ARI_g), axis=0)
        ari_std_g = np.std(np.asarray(ARI_g), axis=0)
        nmi_ave_g = np.mean(np.asarray(NMI_g), axis=0)
        nmi_std_g = np.std(np.asarray(NMI_g), axis=0)
        recm_g[setting] = [ari_ave_g, ari_std_g, nmi_ave_g, nmi_std_g]  # Evaluation Results Clustering GMM
        # recm_g = [ari_ave_g, ari_std_g, nmi_ave_g, nmi_std_g]  # Evaluation Results Clustering GMM

    return recm_k, recm_a, recm_g


def evaluation_with_classification_metric(kmeans_ms, agg_ms, gmm_ms, GT_ms, case):

    recm_k, recm_a, recm_g = {}, {}, {}  # Results Evaluation with Classification Metrics
    case_ = CASES[case]

    print(" ")
    print("Results on the " + case_ + " Variable")
    print(" ")

    print("\t", "  p", "  q", " a/e   ",
          "K-roc_auc", "K-prec", "K-rec", "K-fsc",
          "A-roc_auc", "A-prec", "A-rec", "A-fsc",
          "G-roc_auc", "G-prec", "G-rec", "G-fsc"
          )

    print(" \t", " \t",
          "   Ave", " std", " Ave", " std", " Ave", " std", " Ave", " std",
          "Ave", " std", " Ave", " std", " Ave", " std", " Ave", " std",
          "Ave", " std", " Ave", " std", " Ave", " std", " Ave", " std",
          )

    for setting, results in kmeans_ms.items():

        precision_k, recall_k, fscore_k, roc_auc_k = [], [], [], []
        precision_a, recall_a, fscore_a, roc_auc_a = [], [], [], []
        precision_g, recall_g, fscore_g, roc_auc_g = [], [], [], []

        for repeat, result in results.items():

            gt = GT_ms[setting][repeat]
            lp_k = result[case]
            lp_a = agg_ms[setting][repeat][case]
            lp_g = gmm_ms[setting][repeat][case]

            tmp_k = metrics.precision_recall_fscore_support(gt, lp_k, average='weighted')
            roc_auc_k.append(multiclass_roc_auc_score(gt, lp_k, average='weighted'))

            tmp_a = metrics.precision_recall_fscore_support(gt, lp_a, average='weighted')
            roc_auc_a.append(multiclass_roc_auc_score(gt, lp_a, average='weighted'))

            tmp_g = metrics.precision_recall_fscore_support(gt, lp_g, average='weighted')
            roc_auc_g.append(multiclass_roc_auc_score(gt, lp_g, average='weighted'))

            precision_k.append(tmp_k[0])
            recall_k.append(tmp_k[1])
            fscore_k.append(tmp_k[2])

            precision_a.append(tmp_a[0])
            recall_a.append(tmp_a[1])
            fscore_a.append(tmp_a[2])

            precision_g.append(tmp_g[0])
            recall_g.append(tmp_g[1])
            fscore_g.append(tmp_g[2])

        # K-means stats
        precision_ave_k = np.mean(np.asarray(precision_k), axis=0)
        precision_std_k = np.std(np.asarray(precision_k), axis=0)

        recall_ave_k = np.mean(np.asarray(recall_k), axis=0)
        recall_std_k = np.std(np.asarray(recall_k), axis=0)

        fscore_ave_k = np.mean(np.asarray(fscore_k), axis=0)
        fscore_std_k = np.std(np.asarray(fscore_k), axis=0)

        roc_auc_ave_k = np.mean(np.asarray(roc_auc_k), axis=0)
        roc_auc_std_k = np.std(np.asarray(roc_auc_k), axis=0)
        recm_k[setting] = [roc_auc_ave_k, roc_auc_std_k,
                           precision_ave_k, precision_std_k,
                           recall_ave_k, recall_std_k,
                           fscore_ave_k, fscore_std_k
                           ]

        # Agglomerative stats
        precision_ave_a = np.mean(np.asarray(precision_a), axis=0)
        precision_std_a = np.std(np.asarray(precision_a), axis=0)

        recall_ave_a = np.mean(np.asarray(recall_a), axis=0)
        recall_std_a = np.std(np.asarray(recall_a), axis=0)

        fscore_ave_a = np.mean(np.asarray(fscore_a), axis=0)
        fscore_std_a = np.std(np.asarray(fscore_a), axis=0)

        roc_auc_ave_a = np.mean(np.asarray(roc_auc_a), axis=0)
        roc_auc_std_a = np.std(np.asarray(roc_auc_a), axis=0)
        recm_a[setting] = [roc_auc_ave_a, roc_auc_std_a,
                           precision_ave_a, precision_std_a,
                           recall_ave_a, recall_std_a,
                           fscore_ave_a, fscore_std_a
                           ]

        # GMM stats
        precision_ave_g = np.mean(np.asarray(precision_g), axis=0)
        precision_std_g = np.std(np.asarray(precision_g), axis=0)

        recall_ave_g = np.mean(np.asarray(recall_g), axis=0)
        recall_std_g = np.std(np.asarray(recall_g), axis=0)

        fscore_ave_g = np.mean(np.asarray(fscore_g), axis=0)
        fscore_std_g = np.std(np.asarray(fscore_g), axis=0)

        roc_auc_ave_g = np.mean(np.asarray(roc_auc_g), axis=0)
        roc_auc_std_g = np.std(np.asarray(roc_auc_g), axis=0)
        recm_g[setting] = [roc_auc_ave_g, roc_auc_std_g,
                           precision_ave_g, precision_std_g,
                           recall_ave_g, recall_std_g,
                           fscore_ave_g, fscore_std_g
                           ]

    return recm_k, recm_a, recm_g