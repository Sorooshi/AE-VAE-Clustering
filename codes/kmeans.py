import numpy as np
from copy import deepcopy


MAX_ITERS = 1000


# K-means clustering algorithm
class Kmeans:

    def __init__(self, data, n_clusters):
        super(Kmeans, self).__init__()
        self.data = data
        self.n_clusters = n_clusters
        self.outputs = {}
        self.N = data.shape[0]
        self.V = data.shape[1]
        self.centroids = np.array([]).reshape(self.V, 0)
        self.clusters_labels = np.array([]).reshape(self.N, 0)
        self.seeds = np.random.choice(np.arange(0, self.N), size=n_clusters, replace=False)

        for seed in self.seeds:
            self.centroids = np.c_[self.centroids, self.data[seed, :]]

    def kmeans_plus_plus(self):
        initial_centroids = []
        return initial_centroids

    def apply_kmeans(self):

        f_iter = True
        cntr_iter = 0

        while f_iter is True:

            # previous_outputs = deepcopy(self.outputs)
            # computing the Euclidean distances between entities and centroids.
            # Note that centroids are store in the columns of centroids' matrix.

            distances = np.array([]).reshape(self.data.shape[0], 0)  # Euclidean distances
            for k in range(self.n_clusters):
                tmp_distances = np.sum(np.power(self.data - self.centroids[:, k], 2), axis=1)
                distances = np.c_[distances, tmp_distances]

            self.clusters_labels = np.argmin(distances, axis=1)

            for k in range(self.n_clusters):
                self.outputs[k] = np.array([]).reshape(self.V, 0)

            for i in range(self.N):
                self.outputs[self.clusters_labels[i]] = np.c_[self.outputs[self.clusters_labels[i]], self.data[i, :]]

            for k in range(self.n_clusters):
                self.outputs[k] = self.outputs[k].T

            # Test whether all the new centroids are coincide with previous ones
            previous_centroids = deepcopy(self.centroids)

            for k in range(self.n_clusters):
                self.centroids[:, k] = np.mean(self.outputs[k], axis=0)

            # print("previous centroids:", previous_centroids.shape)
            # print("new centroids:", self.centroids.shape)

            cntr = 0  # number of coincidence for centroids
            for k in range(self.n_clusters):
                if np.array_equal(self.centroids[:, k], previous_centroids[:, k]):
                    cntr += 1

            cntr_iter += 1

            if cntr == self.n_clusters or cntr_iter >= MAX_ITERS:
                f_iter = False

        # print("Kmeans is applied")
        return self.outputs, self.clusters_labels

