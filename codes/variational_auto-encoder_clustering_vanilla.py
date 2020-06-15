import os
import time
import pickle
import warnings
import argparse
import tempfile
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn import mixture
import data_standardization as ds
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split


CASES = ['original', 'reconstructed', 'latent']
MAX_ITERS = 1000
BATCH_SIZE = 100
MVB = False

warnings.filterwarnings('ignore')

tf.keras.backend.set_floatx('float32')


def args_parser(args):
    path = args.Path
    name = args.Name
    run = args.Run
    with_noise = args.With_noise
    pp = args.PreProcessing
    setting = args.Setting
    latent_dim_ratio = args.Latent_dim_ratio
    n_epochs = args.N_epochs

    return path, name, run, with_noise, pp, setting, latent_dim_ratio, n_epochs


def flat_ground_truth(ground_truth):
    """
    :param ground_truth: the clusters/communities cardinality
                        (output of cluster cardinality from synthetic data generator)
    :return: two flat lists, the first one is the list of labels in an appropriate format
             for applying sklearn metrics. And the second list is the list of lists of
              containing indices of nodes in the corresponding cluster.
    """
    k = 1
    interval = 1
    labels_true, labels_true_indices = [], []
    for v in ground_truth:
        tmp_indices = []
        for vv in range(v):
            labels_true.append(k)
            tmp_indices.append(interval + vv)

        k += 1
        interval += v
        labels_true_indices += tmp_indices

    return labels_true, labels_true_indices


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
    print("Results on the" + 'f' + "Variable")
    print("\t", "  p", "  q", " a/e   ", "K-roc_auc_score", "A-roc_auc_score", "G-roc_auc_score", )
    print(" \t", " \t", " \t", "Ave", " std", " Ave", " std", " Ave", " std")

    for setting, results in kmeans_ms.items():

        precision_k, recmall_k, fscore_k, roc_auc_k = [], [], [], []
        precision_a, recmall_a, fscore_a, roc_auc_a = [], [], [], []
        precision_g, recmall_g, fscore_g, roc_auc_g = [], [], [], []

        for repeat, result in results.items():
            gt = GT_ms[setting][repeat]
            lp_k = result[case]
            lp_a = agg_ms[setting][repeat][case]
            lp_g = gmm_ms[setting][repeat][case]

            tmp_k = metrics.precision_recall_fscore_support(gt, lp_k, average='weighted')
            roc_auc_k.append(metrics.roc_auc_score(gt, lp_k, average='weighted'))

            tmp_a = metrics.precision_recall_fscore_support(gt, lp_a, average='weighted')
            roc_auc_a.append(metrics.roc_auc_score(gt, lp_a, average='weighted'))

            tmp_g = metrics.precision_recall_fscore_support(gt, lp_g, average='weighted')
            roc_auc_g.append(metrics.roc_auc_score(gt, lp_g, average='weighted'))

            precision_k.append(tmp_k[0])
            recmall_k.append(tmp_k[1])
            fscore_k.append(tmp_k[2])

            precision_a.append(tmp_a[0])
            recmall_a.append(tmp_a[1])
            fscore_a.append(tmp_a[2])

            precision_g.append(tmp_g[0])
            recmall_g.append(tmp_g[1])
            fscore_g.append(tmp_g[2])

        # K-means stats
        precision_ave_k = np.mean(np.asarray(precision_k), axis=0)
        precision_std_k = np.std(np.asarray(precision_k), axis=0)

        recmall_ave_k = np.mean(np.asarray(recmall_k), axis=0)
        recmall_std_k = np.std(np.asarray(recmall_k), axis=0)

        fscore_ave_k = np.mean(np.asarray(fscore_k), axis=0)
        fscore_std_k = np.std(np.asarray(fscore_k), axis=0)

        roc_auc_ave_k = np.mean(np.asarray(roc_auc_k), axis=0)
        roc_auc_std_k = np.std(np.asarray(roc_auc_k), axis=0)
        recm_k[setting] = [roc_auc_ave_k, roc_auc_std_k]

        # Agglomerative stats
        precision_ave_a = np.mean(np.asarray(precision_a), axis=0)
        precision_std_a = np.std(np.asarray(precision_a), axis=0)

        recall_ave_a = np.mean(np.asarray(recmall_a), axis=0)
        recmall_std_a = np.std(np.asarray(recmall_a), axis=0)

        fscore_ave_a = np.mean(np.asarray(fscore_a), axis=0)
        fscore_std_a = np.std(np.asarray(fscore_a), axis=0)

        roc_auc_ave_a = np.mean(np.asarray(roc_auc_a), axis=0)
        roc_auc_std_a = np.std(np.asarray(roc_auc_a), axis=0)
        recm_a[setting] = [roc_auc_ave_a, roc_auc_std_a]

        # GMM stats
        precision_ave_g = np.mean(np.asarray(precision_g), axis=0)
        precision_std_g = np.std(np.asarray(precision_g), axis=0)

        recmall_ave_g = np.mean(np.asarray(recmall_g), axis=0)
        recmall_std_g = np.std(np.asarray(recmall_g), axis=0)

        fscore_ave_g = np.mean(np.asarray(fscore_g), axis=0)
        fscore_std_g = np.std(np.asarray(fscore_g), axis=0)

        roc_auc_ave_g = np.mean(np.asarray(roc_auc_g), axis=0)
        roc_auc_std_g = np.std(np.asarray(roc_auc_g), axis=0)
        recm_g[setting] = [roc_auc_ave_g, roc_auc_std_g]

    return recm_k, recm_a, recm_g


class Sampling(tf.keras.layers.Layer):

    """Uses (z_mean, z_log_var) to sample z, the vector encoding a datapoint"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):

    """Maps a datapoint vector to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim, intermediate_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = tf.keras.layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):

    """Converts z, the encoded datapoint vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = tf.keras.layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = tf.keras.layers.Dense(original_dim, activation='sigmoid')  # probably linear is needed! !?!?!

    def call(self, inputs):
        z = self.dense_proj(inputs)
        return self.dense_output(z)


class VariationalAutoEncoder(tf.keras.Model):

    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, original_dim, intermediate_dim, latent_dim, name='v-auto-encoder', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim, )
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed, z_mean, z_log_var, z


def run_cluster_latents(Y, P, GT, n_epochs, latent_dim_ratio, repeat, name, setting):

    X = np.concatenate((Y, P), axis=1)

    # Initialization
    latent_dim = int(X.shape[1] / latent_dim_ratio)  # Dimension of latent variables
    original_dim = X.shape[1]  # Dimension of original datapoints

    vae = VariationalAutoEncoder(original_dim=original_dim, intermediate_dim=latent_dim*2, latent_dim=latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    mse_loss_fn_ = tf.keras.losses.MeanSquaredError()
    loss_metric = tf.keras.metrics.Mean()
    loss_metric_ = tf.keras.metrics.Mean()

    n_clusters = len(GT)
    y, _ = flat_ground_truth(GT)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60,
                                                        random_state=42, shuffle=True)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,
                                                    random_state=42, shuffle=True)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    spec = str(n_epochs) + "-" + str(latent_dim) + "-" + name + "-" + str(setting) + "-" + str(repeat)

    print("latent dim:", latent_dim)

    step = 0

    for epoch in range(1, n_epochs + 1):

        step += 1
        # print("epoch:", epoch)

        train_Z = np.array([]).reshape(latent_dim, 0)  # Latent variables (train set)
        train_X_hat = np.array([]).reshape(original_dim, 0)  # reconstructed data points (train set)
        train_loss_epoch = []  # Loss values of Mini-batches per each epoch (train set)

        Z_val = np.array([]).reshape(int(latent_dim), 0)  # Latent variables (validation set)
        val_X_hat = np.array([]).reshape(int(original_dim), 0)  # reconstructed data points (validation set)
        val_loss_epoch = []  # Loss values of Mini-batches per each epoch (validation set)

        # Training the model by iterating over the batches of dataset
        for x_batch_train, _ in train_ds:
            with tf.GradientTape() as tape:
                reconstructed, z_mean, z_log_var, z = vae(x_batch_train)
                # compute reconstruction loss
                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += sum(vae.losses)  # Add KL Divergence regularization loss

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)

            # if step % 100 == 0:
            #     print('step %s: mean loss = %s' % (step, loss_metric.result()))

        # Evaluating the performance of the model is done on validation set.
        # To stop the training procedure we used early stop condition on validation/dev set.
        for x_batch_val, _ in val_ds:
            with tf.GradientTape() as tape:
                reconstructed_, z_mean_, z_log_var_, z_ = vae(x_batch_val)
                Z_val = np.c_[Z_val, z_.numpy().T]
                val_X_hat = np.c_[val_X_hat, reconstructed_.numpy().T]  # concatenating reconstructed data points

                # compute reconstruction loss
                loss_ = mse_loss_fn(x_batch_val, reconstructed_)
                loss_ += sum(vae.losses)  # Add KL Divergence regularization loss

            grads_ = tape.gradient(loss_, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads_, vae.trainable_weights))

            loss_metric_(loss_)

            # if step % 100 == 0:
            #     print('step %s: mean loss = %s' % (step, loss_metric_.result()))

        Z_val = Z_val.T

    # K-menas
    kmeans_Z_val = KMeans(n_clusters=len(set(y_val)), n_jobs=-2).fit(Z_val)
    kmeans_val_labels = kmeans_Z_val.labels_

    # Agglomerative (merge)
    agglomerative_val = AgglomerativeClustering(n_clusters=len(set(y_val))).fit(Z_val)
    agg_val_labels = agglomerative_val.labels_

    # Gaussian Mixture Model
    gmm_val = mixture.GaussianMixture(n_components=len(set(y_val), )).fit(Z_val)
    gmm_val_labels = gmm_val.predict(Z_val)

    print("Dev set ARI: ")
    print("K-means:", metrics.adjusted_rand_score(labels_true=y_val, labels_pred=kmeans_val_labels),
          "Agg: ", metrics.adjusted_rand_score(labels_true=y_val, labels_pred=agg_val_labels),
          "GMM: ", metrics.adjusted_rand_score(labels_true=y_val, labels_pred=gmm_val_labels)
          )

    print("Dev set NMI: ")
    print("K-means: ", metrics.normalized_mutual_info_score(labels_true=y_val,
                                                            labels_pred=kmeans_val_labels,
                                                            average_method='max'),

          "Agg: ", metrics.normalized_mutual_info_score(labels_true=y_val,
                                                        labels_pred=agg_val_labels,
                                                        average_method='max'),

          "GMM: ", metrics.normalized_mutual_info_score(labels_true=y_val,
                                                        labels_pred=gmm_val_labels,
                                                        average_method='max'),
          )

    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     autoencoder.save_weights(os.path.join(tmpdirname, "AE-" + str(repeat) + spec + ".h5"))
    #     print("Training finished!")
    #     autoencoder.load_weights(os.path.join(tmpdirname, "AE-" + str(repeat) + spec + ".h5"))

    Z_test = np.array([]).reshape(int(latent_dim), 0)  # Latent variables (test set)
    X_test_hat = np.array([]).reshape(int(original_dim), 0)  # reconstructed data points (test set)

    for x_batch_ts, _ in test_ds:
        with tf.GradientTape() as tape:
            reconstructed__, z_mean__, z_log_var__, z__ = vae(x_batch_ts)
            Z_test = np.c_[Z_test, z__.numpy().T]
            X_test_hat = np.c_[X_test_hat, reconstructed__.numpy().T]  # concatenating reconstructed data points

    Z_test = Z_test.T
    X_test_hat = X_test_hat.T

    # K-means
    kmeans_X_test = KMeans(n_clusters=len(set(y_test)), n_jobs=-2).fit(X_test)
    kmeans_X_test_labels = kmeans_X_test.labels_

    kmeans_X_test_hat = KMeans(n_clusters=len(set(y_test)), n_jobs=-2).fit(X_test_hat)
    kmeans_X_test_hat_labels = kmeans_X_test_hat.labels_

    kmeans_Z_test = KMeans(n_clusters=len(set(y_test)), n_jobs=-2).fit(Z_test)
    kmeans_Z_test_labels = kmeans_Z_test.labels_
    kmeans_test_labels = [kmeans_X_test_labels, kmeans_X_test_hat_labels, kmeans_Z_test_labels]

    # Agglomerative (merge)
    agglomerative_X_test = AgglomerativeClustering(n_clusters=len(set(y_test))).fit(X_test)
    agg_X_test_labels = agglomerative_X_test.labels_

    agglomerative_X_test_hat = AgglomerativeClustering(n_clusters=len(set(y_test))).fit(X_test_hat)
    agg_X_test_hat_labels = agglomerative_X_test_hat.labels_

    agglomerative_Z_test = AgglomerativeClustering(n_clusters=len(set(y_test))).fit(Z_test)
    agg_Z_test_labels = agglomerative_Z_test.labels_
    agg_test_labels = [agg_X_test_labels, agg_X_test_hat_labels, agg_Z_test_labels]

    # Gaussian Mixture Model
    gmm_X_test = mixture.GaussianMixture(n_components=len(set(y_test), )).fit(X_test)
    gmm_X_test_labels = gmm_X_test.predict(X_test)

    gmm_X_test_hat = mixture.GaussianMixture(n_components=len(set(y_test), )).fit(X_test_hat)
    gmm_X_test_hat_labels = gmm_X_test_hat.predict(X_test_hat)

    gmm_Z_test = mixture.GaussianMixture(n_components=len(set(y_test), )).fit(Z_test)
    gmm_Z_test_labels = gmm_Z_test.predict(Z_test)
    gmm_test_labels = [gmm_X_test_labels, gmm_X_test_hat_labels, gmm_Z_test_labels]

    return kmeans_test_labels, agg_test_labels, gmm_test_labels, y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--Path', type=str, default='/home/soroosh/gps1/NNs4clustering/synthetic_data/',
                        help='Path to load the data sets')

    parser.add_argument('--Name', type=str, default='--',
                        help='Name of the Experiment')

    parser.add_argument('--Run', type=int, default=0,
                        help='Whether to run the program or to evaluate the results')

    parser.add_argument('--With_noise', type=int, default=0,
                        help='With noisy features or without')

    parser.add_argument('--PreProcessing', type=str, default='z-m',
                        help='string determining which pre processing method should be applied.'
                             'The first letter determines Y pre processing and the third determines P pre processing. '
                             'Separated with "-".')

    parser.add_argument('--Setting', type=str, default='all')

    parser.add_argument('--Latent_dim_ratio', type=float, default=16.,
                        help='A float denoting the ratio between original data dimension and the latent dimension')

    parser.add_argument('--N_epochs', type=int, default=500,
                        help='An int. denoting the number of epochs')

    args = parser.parse_args()
    path, name, run, with_noise, pp, setting_, latent_dim_ratio, n_epochs = args_parser(args)

    start = time.time()

    if run == 1:

        with open(os.path.join(path, name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        data_name = name.split('(')[0][-2:]
        if with_noise == 1:
            data_name = data_name + "-N"
        type_of_data = name.split('(')[0][-1]
        print("run:", run, name, pp, with_noise, setting_, data_name, type_of_data)

        def apply_aec(data_type, with_noise):  # auto-encode cluster

            # Global initialization
            kmeans_ms = {}  # K-means results
            agg_ms = {}  # Agglomerative results
            gmm_ms = {}  # Gaussian Mixture Model
            GT_ms = {}  # Ground Truth

            if setting_ != 'all':
                for setting, repeats in DATA.items():
                    if str(setting) == setting_:
                        print("setting:", setting, )

                        kmeans_ms[setting] = {}
                        agg_ms[setting] = {}
                        gmm_ms[setting] = {}
                        GT_ms[setting] = {}

                        for repeat, matrices in repeats.items():
                            print("repeat:", repeat)
                            GT = matrices['GT']
                            Y = matrices['Y'].astype("float32")
                            P = matrices['P'].astype("float32")
                            Yn = matrices['Yn']
                            if len(Yn) != 0:
                                Yn = Yn.astype('float32')
                            N, V = Y.shape

                            # Quantitative case
                            if type_of_data == 'Q':
                                _, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y, data_type='Q')

                                if with_noise == 1:
                                    Yn, _, Ynz, _, Ynrng, _, = ds.preprocess_Y(Yin=Yn, data_type='Q')

                            # Because there is no Yn in the case of categorical features.
                            if type_of_data == 'C':
                                enc = OneHotEncoder(sparse=False, categories='auto')
                                Y_oneHot = enc.fit_transform(Y).astype("float32")  # oneHot encoding

                                # for WITHOUT follow-up rescale Y_oneHot and for WITH follow-up
                                # Y_oneHot should be replaced with Y
                                Y, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y_oneHot, data_type='C')

                            if type_of_data == 'M':
                                Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                Y_q, _, Yz_q, _, Yrng_q, _, = ds.preprocess_Y(Yin=Y[:, :Vq], data_type='Q')
                                enc = OneHotEncoder(sparse=False, categories='auto', )
                                Y_oneHot = enc.fit_transform(Y[:, Vq:]).astype("float32")  # oneHot encoding

                                # for WITHOUT follow-up rescale Y_oneHot and for WITH follow-up
                                # Y_oneHot should be replaced with Y
                                Y_c, _, Yz_c, _, Yrng_c, _, = ds.preprocess_Y(Yin=Y_oneHot, data_type='C')

                                Y = np.concatenate([Y[:, :Vq], Y_oneHot], axis=1)
                                Yrng = np.concatenate([Yrng_q, Yrng_c], axis=1)
                                Yz = np.concatenate([Yz_q, Yz_c], axis=1)

                                if with_noise == 1:
                                    Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                    Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                    Vqn = (Vq + Vc)  # the column index of which noise model1 starts

                                    _, _, Ynz_q, _, Ynrng_q, _, = ds.preprocess_Y(Yin=Yn[:, :Vq], data_type='Q')

                                    enc = OneHotEncoder(sparse=False, categories='auto', )
                                    Yn_oneHot = enc.fit_transform(Yn[:, Vq:Vqn]).astype("float32")  # oneHot encoding
                                    # for WITHOUT follow-up rescale Yn_oneHot and for WITH
                                    # follow-up Yn_oneHot should be replaced with Y
                                    Yn_c, _, Ynz_c, _, Ynrng_c, _, = ds.preprocess_Y(Yin=Yn_oneHot, data_type='C')

                                    Y_ = np.concatenate([Yn[:, :Vq], Yn_c], axis=1)
                                    Yrng = np.concatenate([Ynrng_q, Ynrng_c], axis=1)
                                    Yz = np.concatenate([Ynz_q, Ynz_c], axis=1)

                                    _, _, Ynz_, _, Ynrng_, _, = ds.preprocess_Y(Yin=Yn[:, Vqn:], data_type='Q')
                                    Yn_ = np.concatenate([Y_, Yn[:, Vqn:]], axis=1)
                                    Ynrng = np.concatenate([Yrng, Ynrng_], axis=1)
                                    Ynz = np.concatenate([Yz, Ynz_], axis=1)

                            P, _, _, Pu, _, _, Pm, _, _, Pl, _, _ = ds.preprocess_P(P=P)

                            # Pre-processing - Without Noise
                            if data_type == "NP".lower() and with_noise == 0:
                                print("NP")

                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y, P, GT, n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-u".lower() and with_noise == 0:
                                print("z-u")
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yz, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-m".lower() and with_noise == 0:

                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yz, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-l".lower() and with_noise == 0:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yz, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "rng-u".lower() and with_noise == 0:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yrng, P=Pu, GT=GT, n_epochs=n_epochs,
                                    latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name,
                                    setting=setting)

                            elif data_type == "rng-m".lower() and with_noise == 0:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yrng, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "rng-l".lower() and with_noise == 0:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yrng, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            # Pre-processing - With Noise
                            if data_type == "NP".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Yn, P=P, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-u".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynz, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "z-m".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynz, P=Pm, GT=GT, n_epochs=n_epochs,
                                    latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=data_name, setting=setting)

                            elif data_type == "z-l".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynz, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "rng-u".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynrng, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            elif data_type == "rng-m".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynrng, P=Pm, GT=GT, n_epochs=n_epochs,
                                    latent_dim_ratio=latent_dim_ratio, repeat=repeat,
                                    name=data_name, setting=setting)

                            elif data_type == "rng-l".lower() and with_noise == 1:
                                kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                    Y=Ynrng, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                    repeat=repeat, name=data_name, setting=setting)

                            kmeans_ms[setting][repeat] = kmeans_labels
                            agg_ms[setting][repeat] = agg_labels
                            gmm_ms[setting][repeat] = gmm_labels
                            GT_ms[setting][repeat] = y_test

                    print("Algorithm is applied on the" + setting_ + "data set!")

            if setting_ == 'all':

                for setting, repeats in DATA.items():

                    print("setting:", setting, )

                    kmeans_ms[setting] = {}
                    agg_ms[setting] = {}
                    gmm_ms[setting] = {}
                    GT_ms[setting] = {}

                    for repeat, matrices in repeats.items():
                        print("repeat:", repeat)
                        GT = matrices['GT']
                        Y = matrices['Y'].astype('float32')
                        P = matrices['P'].astype('float32')
                        Yn = matrices['Yn']
                        if len(Yn) != 0:
                            Yn = Yn.astype('float32')
                        N, V = Y.shape

                        # Quantitative case
                        if type_of_data == 'Q':
                            _, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y, data_type='Q')
                            if with_noise == 1:
                                Yn, _, Ynz, _, Ynrng, _, = ds.preprocess_Y(Yin=Yn, data_type='Q')

                        # Because there is no Yn in the case of categorical features.
                        if type_of_data == 'C':
                            enc = OneHotEncoder()  # categories='auto')
                            Y = enc.fit_transform(Y).astype('float32')  # oneHot encoding
                            Y = Y.toarray()
                            # Boris's Theory
                            Y, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y, data_type='C')

                        if type_of_data == 'M':
                            Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                            Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                            Y_, _, Yz_, _, Yrng_, _, = ds.preprocess_Y(Yin=Y[:, :Vq], data_type='M')
                            enc = OneHotEncoder(sparse=False, )  # categories='auto', )
                            Y_oneHot = enc.fit_transform(Y[:, Vq:]).astype('float32')  # oneHot encoding
                            Y = np.concatenate([Y_oneHot, Y[:, :Vq]], axis=1)
                            Yrng = np.concatenate([Y_oneHot, Yrng_], axis=1)
                            Yz = np.concatenate([Y_oneHot, Yz_], axis=1)

                            if with_noise == 1:
                                Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                Vqn = (Vq + Vc)  # the column index of which noise model1 starts

                                _, _, Yz_, _, Yrng_, _, = ds.preprocess_Y(Yin=Yn[:, :Vq], data_type='M')
                                enc = OneHotEncoder(sparse=False, )  # categories='auto',)
                                Yn_oneHot = enc.fit_transform(Yn[:, Vq:Vqn]).astype('float32')  # oneHot encoding
                                Y_ = np.concatenate([Yn_oneHot, Yn[:, :Vq]], axis=1)
                                Yrng = np.concatenate([Yn_oneHot, Yrng_], axis=1)
                                Yz = np.concatenate([Yn_oneHot, Yz_], axis=1)

                                _, _, Ynz_, _, Ynrng_, _, = ds.preprocess_Y(Yin=Yn[:, Vqn:], data_type='M')
                                Yn_ = np.concatenate([Y_, Yn[:, Vqn:]], axis=1)
                                Ynrng = np.concatenate([Yrng, Ynrng_], axis=1)
                                Ynz = np.concatenate([Yz, Ynz_], axis=1)

                        P, _, _, Pu, _, _, Pm, _, _, Pl, _, _ = ds.preprocess_P(P=P)

                        # Pre-processing - Without Noise
                        if data_type == "NP".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Y, P=P, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-u".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yz, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-m".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yz, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-l".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yz, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-u".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yrng, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-m".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yrng, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-l".lower() and with_noise == 0:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yrng, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        # Pre-processing - With Noise
                        if data_type == "NP".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Yn, P=P, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-u".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynz, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-m".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynz, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "z-l".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynz, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-u".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynrng, P=Pu, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-m".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynrng, P=Pm, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        elif data_type == "rng-l".lower() and with_noise == 1:
                            kmeans_labels, agg_labels, gmm_labels, y_test = run_cluster_latents(
                                Y=Ynrng, P=Pl, GT=GT, n_epochs=n_epochs, latent_dim_ratio=latent_dim_ratio,
                                repeat=repeat, name=data_name, setting=setting)

                        kmeans_ms[setting][repeat] = kmeans_labels
                        agg_ms[setting][repeat] = agg_labels
                        gmm_ms[setting][repeat] = gmm_labels
                        GT_ms[setting][repeat] = y_test

                print("Algorithm is applied on the entire data set!")

            return kmeans_ms, agg_ms, gmm_ms, GT_ms


        kmeans_ms, agg_ms, gmm_ms, GT_ms = apply_aec(data_type=pp.lower(), with_noise=with_noise)

        end = time.time()
        print("Time:", end - start)

        if with_noise == 1:
            name = name + '-N'

        if setting_ != 'all':
            # Saving K-Means results
            with open(os.path.join('VAE-KC_computation', "kmeans_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(kmeans_ms, fp)

            # Saving Agglomerative results
            with open(os.path.join('VAE-KC_computation', "agg_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(agg_ms, fp)

            # Saving Gaussian Mixture Model
            with open(os.path.join('VAE-KC_computation', "gmm_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(gmm_ms, fp)

            # Saving the corresponding Ground Truth
            with open(os.path.join('VAE-KC_computation', "GT_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        if setting_ == 'all':
            with open(os.path.join('VAE-KC_computation', "kmeans_ms_" +
                                                        name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(kmeans_ms, fp)

            # Saving Agglomerative results
            with open(os.path.join('VAE-KC_computation', "agg_ms_" +
                                                        name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(agg_ms, fp)

            # Saving Gaussian Mixture Model
            with open(os.path.join('VAE-KC_computation', "gmm_ms_" +
                                                        name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(gmm_ms, fp)

            with open(os.path.join('VAE-KC_computation', "GT_ms_" + name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        print("Results are saved!")

        for case in range(len(CASES)):

            res_clt_m_k, re_clt_m_a, re_clt_m_g = evaluation_with_clustering_metrics(kmeans_ms=kmeans_ms,
                                                                                     agg_ms=agg_ms,
                                                                                     gmm_ms=gmm_ms,
                                                                                     GT_ms=GT_ms,
                                                                                     case=-case)

            for setting, eval_k in res_clt_m_k.items():
                eval_a = re_clt_m_a[setting]
                eval_g = re_clt_m_g[setting]

                print("setting:", setting,
                      "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
                      "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
                      )

            # res_clf_m_k, re_clf_m_a, re_clf_m_g = evaluation_with_classification_metric(kmeans_ms=kmeans_ms,
            #                                                                             agg_ms=agg_ms,
            #                                                                             gmm_ms=gmm_ms,
            #                                                                             GT_ms=GT_ms,
            #                                                                             case=case)
            #
            # for setting, eval_k in res_clf_m_k.items():
            #     eval_a = re_clf_m_a[setting]
            #     eval_g = re_clf_m_g[setting]
            #
            #     print("setting:", setting,
            #           "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
            #           "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
            #           "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
            #           )

    if run == 0:

        print(" \t", " \t", "name:", name)

        if with_noise == 1:
            name = name + '-N'

        with open(os.path.join('VAE-KC_computation', "kmeans_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            kmeans_ms = pickle.load(fp)

        with open(os.path.join('VAE-KC_computation', "agg_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            agg_ms = pickle.load(fp)

        with open(os.path.join('VAE-KC_computation', "gmm_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            gmm_ms = pickle.load(fp)

        with open(os.path.join('VAE-KC_computation', "GT_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            GT_ms = pickle.load(fp)

        for case in range(len(CASES)):

            res_clt_m_k, re_clt_m_a, re_clt_m_g = evaluation_with_clustering_metrics(kmeans_ms=kmeans_ms,
                                                                                     agg_ms=agg_ms,
                                                                                     gmm_ms=gmm_ms,
                                                                                     GT_ms=GT_ms,
                                                                                     case=case)

            for setting, eval_k in res_clt_m_k.items():
                eval_a = re_clt_m_a[setting]
                eval_g = re_clt_m_g[setting]

                print("setting:", setting,
                      "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
                      "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
                      )

            # res_clf_m_k, re_clf_m_a, re_clf_m_g = evaluation_with_classification_metric(kmeans_ms=kmeans_ms,
            #                                                                             agg_ms=agg_ms,
            #                                                                             gmm_ms=gmm_ms,
            #                                                                             GT_ms=GT_ms,
            #                                                                             case=case)
            #
            # for setting, eval_k in res_clf_m_k.items():
            #     eval_a = re_clf_m_a[setting]
            #     eval_g = re_clf_m_g[setting]
            #
            #     print("setting:", setting,
            #           "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
            #           "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
            #           "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3]
            #           )