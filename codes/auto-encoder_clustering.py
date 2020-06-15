import os
import time
import pickle
import warnings
import argparse
import tempfile
import numpy as np
import tensorflow as tf
import CDmetrics as cdm
from sklearn import metrics
from sklearn import mixture
import data_standardization as ds
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split


BATCH_SIZE = 100
warnings.filterwarnings('ignore')
tf.keras.backend.set_floatx('float32')
CASES = ['original', 'reconstructed', 'latent']


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
            tmp_indices.append(interval+vv)

        k += 1
        interval += v
        labels_true_indices += tmp_indices

    return labels_true, labels_true_indices


class Encoder(tf.keras.layers.Layer):
    def __init__(self, original_dim, latent_dim):
        super(Encoder, self).__init__()
        self.h1 = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.relu,
                                        input_shape=(original_dim,))
        self.h2 = tf.keras.layers.Dense(units=int(original_dim/2), activation=tf.nn.relu,)
        self.h3 = tf.keras.layers.Dense(units=int(original_dim/2), activation=tf.nn.relu,)
        self.h4 = tf.keras.layers.Dense(units=int(original_dim/4), activation=tf.nn.relu,)
        self.h5 = tf.keras.layers.Dense(units=int(original_dim/8), activation=tf.nn.relu,)
        self.z = tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu)

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        x = self.h5(x)
        z = self.z(x)
        return z


class Decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, original_dim):
        super(Decoder, self).__init__()
        self.h1 = tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu,
                                        input_shape=(latent_dim,))
        self.h2 = tf.keras.layers.Dense(units=int(original_dim/8), activation=tf.nn.relu, )
        self.h3 = tf.keras.layers.Dense(units=int(original_dim/4), activation=tf.nn.relu,)
        self.h4 = tf.keras.layers.Dense(units=int(original_dim/2), activation=tf.nn.relu, )
        self.h5 = tf.keras.layers.Dense(units=int(original_dim/2), activation=tf.nn.relu,)
        self.x_hat = tf.keras.layers.Dense(units=original_dim, )

    def call(self, z):
        z = self.h1(z)
        z = self.h2(z)
        z = self.h3(z)
        z = self.h4(z)
        z = self.h5(z)
        x = self.x_hat(z)
        return x


class AutoEncoder(tf.keras.Model):
    def __init__(self, latent_dim, original_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(original_dim=original_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, original_dim=original_dim)

    def call(self, x):
        coded = self.encoder(x)  # z/latent features/bottle neck
        decoded = self.decoder(coded)  # recmonstructed
        return coded, decoded


def computation(model, original):
    latent_variables = model.encoder(original)
    recmonstructed = model.decoder(latent_variables)
    recmonstruction_error = tf.losses.mean_squared_error(recmonstructed, original)
    return latent_variables, recmonstructed, recmonstruction_error


def train(computation, model, opt, original):
    with tf.GradientTape() as tape:
        latent_variables, recmonstructed, recmonstruction_error = computation(model, original)
        gradients = tape.gradient(recmonstruction_error, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))


def run_cluster_latents(Y, P, GT, n_epochs, latent_dim_ratio, repeat, name, setting):

    X = np.concatenate((Y, P), axis=1)
    N, V = X.shape
    n_clusters = len(GT)
    if len(name) == 2:  # because the length of the real-world datasets are all larger than two strings
        y, _ = flat_ground_truth(GT)
    else:
        y = GT

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60,
                                                        random_state=42, shuffle=True)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,
                                                    random_state=42, shuffle=True)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    # Initialization
    latent_dim = int(X.shape[1] / latent_dim_ratio)  # Dimension of latent variables
    original_dim = X.shape[1]  # Dimension of original datapoints
    spec = str(n_epochs) + "-" + str(latent_dim) + "-" + name + "-" + str(setting) + "-" + str(repeat)

    autoencoder = AutoEncoder(latent_dim=latent_dim, original_dim=original_dim)
    opt = tf.optimizers.Adam(learning_rate=1e-6)

    train_loss_miniBatche = []  # Total loss per Mini-batches (train set)
    train_loss_total_ave = []  # Average loss per each epoch (train set)
    val_loss_miniBatche = []  # Total loss per Mini-batches (validation set)
    val_loss_total_ave = []  # Average loss per each epoch (validation set)
    delta = 1e-4  # The difference between two consecutive validation losses for early stop
    early_stop = False
    early_stop_counter = 0
    patience = 40  # Number of epochs to calculate the differences between consecutive epochs
    
    print("latent dim:", latent_dim)
    
    for epoch in range(1, n_epochs + 1):

        train_Z = np.array([]).reshape(latent_dim, 0)  # Latent variables (train set)
        train_X_hat = np.array([]).reshape(original_dim, 0)  # recmonstructed data points (train set)
        train_loss_epoch = []  # Loss values of Mini-batches per each epoch (train set)

        Z_val = np.array([]).reshape(int(latent_dim), 0)  # Latent variables (validation set)
        val_X_hat = np.array([]).reshape(int(original_dim), 0)  # recmonstructed data points (validation set)
        val_loss_epoch = []  # Loss values of Mini-batches per each epoch (validation set)

        # Training the model
        for X_tr, _ in train_ds:
            train(computation, autoencoder, opt, X_tr)
            codes, decodes, loss_values = computation(autoencoder, X_tr)
            train_Z = np.c_[train_Z, codes.numpy().T]  # concatenating latent variables
            train_X_hat = np.c_[
                train_X_hat, decodes.numpy().T]  # concatenating recmonstructed data points
            train_loss_epoch += loss_values.numpy().tolist()  # appending loss values

        train_loss_total_ave.append(np.mean(np.array(train_loss_epoch)))
        train_loss_miniBatche += [i for i in train_loss_epoch]

        # Evaluating the performance of the model is done on validation set.
        # To stop the training procedure we used early stop condition on validation/dev set.
        for X_vl, _ in val_ds:
            train(computation, autoencoder, opt, X_vl)
            codes_, decodes_, loss_values_ = computation(autoencoder, X_vl)
            Z_val = np.c_[Z_val, codes_.numpy().T]  # concatenating latent variables
            val_X_hat = np.c_[val_X_hat, decodes_.numpy().T]  # concatenating recmonstructed data points
            val_loss_epoch += loss_values_.numpy().tolist()  # appending loss values

        val_loss_total_ave.append(np.mean(np.array(val_loss_epoch)))
        val_loss_miniBatche += [i for i in val_loss_epoch]

        # if epoch % 10 == 0:
            # print("epoch:", epoch)
            # autoencoder.save_weights("AE" + spec + "-" + str(epoch) + ".h5")

        # if epoch >= patience and epoch % patience == 0:
        #     history = val_loss_total_ave[-patience:]
        #     for i in range(len(history)):
        #         if i < len(history) - 1 and i >= 1:
        #             if history[i - 1] - history[i] < delta:
        #                 early_stop_counter += 1
        #         if early_stop_counter >= patience - 2:
        #             early_stop = True
        #
        #     early_stop_counter = 0
        # if early_stop is True:
        #     break

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

    with tempfile.TemporaryDirectory() as tmpdirname:
        autoencoder.save_weights(os.path.join(tmpdirname, "AE-" + str(repeat) + spec + ".h5"))
        print("Training finished!")
        autoencoder.load_weights(os.path.join(tmpdirname, "AE-" + str(repeat) + spec + ".h5"))

    Z_test = autoencoder.encoder(X_test).numpy()  # Latent Variables
    X_test_hat = autoencoder.decoder(
        Z_test).numpy()  # recmonstructed data points

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

    gmm_Z_test = mixture.GaussianMixture(n_components=len(set(y_test),)).fit(Z_test)
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

        data_name = name.split('(')[0]
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
                            if type_of_data == 'Q' or name.split('(')[-1] == 'r':
                                _, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y, data_type='Q')
                                if with_noise == 1:
                                    Yn, _, Ynz, _, Ynrng, _, = ds.preprocess_Y(Yin=Yn, data_type='Q')

                            # Because there is no Yn in the case of categorical features.
                            if type_of_data == 'C':
                                enc = OneHotEncoder(sparse=False, categories='auto')
                                Y_oneHot = enc.fit_transform(Y)  # .astype("float32")  # oneHot encoding

                                # for WITHOUT follow-up rescale Y_oneHot and for WITH follow-up
                                # Y_oneHot should be replaced with Y
                                Y, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y_oneHot, data_type='C')

                            if type_of_data == 'M':
                                Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                Y_q, _, Yz_q, _, Yrng_q, _, = ds.preprocess_Y(Yin=Y[:, :Vq], data_type='Q')
                                enc = OneHotEncoder(sparse=False, categories='auto',)
                                Y_oneHot = enc.fit_transform(Y[:, Vq:])  # oneHot encoding

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

                                    enc = OneHotEncoder(sparse=False, categories='auto',)
                                    Yn_oneHot = enc.fit_transform(Yn[:, Vq:Vqn])  # oneHot encoding
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
                        if type_of_data == 'Q' or name.split('(')[-1] == 'r':
                            _, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y, data_type='Q')
                            if with_noise == 1:
                                Yn, _, Ynz, _, Ynrng, _, = ds.preprocess_Y(Yin=Yn, data_type='Q')

                        # Because there is no Yn in the case of categorical features.
                        if type_of_data == 'C':
                            enc = OneHotEncoder()  # categories='auto')
                            Y = enc.fit_transform(Y)  # oneHot encoding
                            Y = Y.toarray()
                            # Boris's Theory
                            Y, _, Yz, _, Yrng, _, = ds.preprocess_Y(Yin=Y, data_type='C')

                        if type_of_data == 'M':
                            Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                            Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                            Y_, _, Yz_, _, Yrng_, _, = ds.preprocess_Y(Yin=Y[:, :Vq], data_type='M')
                            enc = OneHotEncoder(sparse=False, )  # categories='auto', )
                            Y_oneHot = enc.fit_transform(Y[:, Vq:])  # oneHot encoding
                            Y = np.concatenate([Y_oneHot, Y[:, :Vq]], axis=1)
                            Yrng = np.concatenate([Y_oneHot, Yrng_], axis=1)
                            Yz = np.concatenate([Y_oneHot, Yz_], axis=1)

                            if with_noise == 1:
                                Vq = int(np.ceil(V / 2))  # number of quantitative features -- Y[:, :Vq]
                                Vc = int(np.floor(V / 2))  # number of categorical features  -- Y[:, Vq:]
                                Vqn = (Vq + Vc)  # the column index of which noise model1 starts

                                _, _, Yz_, _, Yrng_, _, = ds.preprocess_Y(Yin=Yn[:, :Vq], data_type='M')
                                enc = OneHotEncoder(sparse=False, )  # categories='auto',)
                                Yn_oneHot = enc.fit_transform(Yn[:, Vq:Vqn])  # oneHot encoding
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
            with open(os.path.join('AE-KC_computation', "kmeans_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(kmeans_ms, fp)

            # Saving Agglomerative results
            with open(os.path.join('AE-KC_computation', "agg_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(agg_ms, fp)

            # Saving Gaussian Mixture Model
            with open(os.path.join('AE-KC_computation', "gmm_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(gmm_ms, fp)

            # Saving the corresponding Ground Truth
            with open(os.path.join('AE-KC_computation', "GT_ms_" +
                                                        name + "-" + pp + "-" + setting_ +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        if setting_ == 'all':

            with open(os.path.join('AE-KC_computation', "kmeans_ms_" +
                                                        name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(kmeans_ms, fp)

            # Saving Agglomerative results
            with open(os.path.join('AE-KC_computation', "agg_ms_" +
                                                        name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(agg_ms, fp)

            # Saving Gaussian Mixture Model
            with open(os.path.join('AE-KC_computation', "gmm_ms_" +
                                                        name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(gmm_ms, fp)

            with open(os.path.join('AE-KC_computation', "GT_ms_" + name + "-" + pp +
                                                        "-" + str(n_epochs) + ".pickle"), 'wb') as fp:
                pickle.dump(GT_ms, fp)

        print("Results are saved!")

        for case in range(len(CASES)):

            res_clt_m_k, re_clt_m_a, re_clt_m_g = cdm.evaluation_with_clustering_metrics(kmeans_ms=kmeans_ms,
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

            # res_clf_m_k, re_clf_m_a, re_clf_m_g = cdm.evaluation_with_classification_metric(kmeans_ms=kmeans_ms,
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

        with open(os.path.join('AE-KC_computation', "kmeans_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            kmeans_ms = pickle.load(fp)

        with open(os.path.join('AE-KC_computation', "agg_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            agg_ms = pickle.load(fp)

        with open(os.path.join('AE-KC_computation', "gmm_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            gmm_ms = pickle.load(fp)

        with open(os.path.join('AE-KC_computation', "GT_ms_" + name + "-" + pp +
                                                    "-" + str(n_epochs) + ".pickle"), 'rb') as fp:
            GT_ms = pickle.load(fp)

        for case in range(len(CASES)):

            res_clt_m_k, re_clt_m_a, re_clt_m_g = cdm.evaluation_with_clustering_metrics(kmeans_ms=kmeans_ms,
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

            res_clf_m_k, re_clf_m_a, re_clf_m_g = cdm.evaluation_with_classification_metric(kmeans_ms=kmeans_ms,
                                                                                            agg_ms=agg_ms,
                                                                                            gmm_ms=gmm_ms,
                                                                                            GT_ms=GT_ms,
                                                                                            case=case)

            for setting, eval_k in res_clf_m_k.items():
                eval_a = re_clf_m_a[setting]
                eval_g = re_clf_m_g[setting]

                print("setting:", setting,
                      "%.3f" % eval_k[0], "%.3f" % eval_k[1], "%.3f" % eval_k[2], "%.3f" % eval_k[3],
                      "%.3f" % eval_k[4], "%.3f" % eval_k[5], "%.3f" % eval_k[6], "%.3f" % eval_k[7],
                      "%.3f" % eval_a[0], "%.3f" % eval_a[1], "%.3f" % eval_a[2], "%.3f" % eval_a[3],
                      "%.3f" % eval_a[4], "%.3f" % eval_a[5], "%.3f" % eval_a[6], "%.3f" % eval_a[7],
                      "%.3f" % eval_g[0], "%.3f" % eval_g[1], "%.3f" % eval_g[2], "%.3f" % eval_g[3],
                      "%.3f" % eval_g[4], "%.3f" % eval_g[5], "%.3f" % eval_g[6], "%.3f" % eval_g[7]
                      )