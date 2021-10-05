import numpy as np
from tqdm import tqdm

class Kmeans(object):
    def __init__(self, data: np.ndarray, n_cluster: int, iter: int) -> None:
        """K-Means Algorithm for hard clustering.

        Args:
        =====
        data: numpy.ndarray
            Target data
        n_cluster: int
            Number of clusters
        iter: int
            Max trial number.

        """
        self.n_cluster = n_cluster
        if self.n_cluster < 1:
            print("You need to set cluster number bigger than 1.")
            print("n_cluster automatically set to 2.")
            self.n_cluster = 2
        self.data = data
        self.iter = iter
        self.dim_data = len(self.data[0])
        self.n_data = len(self.data)
        self.idx = None
        self.centroid = None

    def fit(self) -> None:
        """ The K-means algorithm
        1. Pick n data points that will act as the initial centroids.
        2. Calculate the Euclidean distance of each data point from each of the centroid points selected in step 1.
        3. Form data clusters by assigning every data point to whichever centroid it has the smallest distance from.
        4. Take the average of each formed cluster. The mean points are our new centroids. """
        self.__init_centroid()
        idx = np.zeros(self.n_data)
        idx_new = np.zeros(self.n_data)
        count = 0
        data_re = self.data.reshape(self.n_data, 1, 1, self.dim_data)
        for _ in tqdm(range(self.iter), desc="K-Means Fit Progress", ncols=0, unit='iter'):
            dist = np.sum((data_re - self.centroid.reshape(1, self.n_cluster, self.dim_data))**2, axis=3)
            idx = np.argmin(dist, axis=2).flatten()
            for i in range(self.n_cluster):
                self.centroid[i] = self.data[idx==i].mean(axis=0)
            if np.all(idx==idx_new) and count == 2:
                self.idx = idx
                break
            elif np.all(idx==idx_new) and count < 2:
                count += 1
                idx_new = idx
            else:
                idx_new = idx

    def __init_centroid(self) -> np.ndarray:
        """The K-means++ algorithm
        1. The first centroid is selected randomly.
        2. Calculate the Euclidean distance between the centroid and every other data point in the dataset. The point farthest away will become our next centroid.
        3. Create clusters around these centroids by associating every point with its nearest centroid.
        4. The point which has the farthest distance from its centroid will be our next centroid.
        5. Repeat steps 3 and 4 until n number of centroids are located. """
        cent = np.zeros((self.n_cluster, self.dim_data))
        dist = np.zeros((self.n_data, self.n_cluster))
        for i, row in enumerate(cent):
            if i == 0:
                pr = np.repeat(1/self.n_data, self.n_data)
            else:
                pr = np.sum(dist, axis=1) / np.sum(dist)
            cent_point_idx = np.random.choice(self.n_data, 1, replace=False, p=pr)
            cent[i] = self.data[cent_point_idx]
            dist[:, i] = np.sum((self.data - row)**2, axis=1)
        self.centroid = cent
        return cent

    def __calc_sse(self):
        """SSE(Sum of Squared Errors)

        Calculate error of prediction.
        """
        sse = 0
        for i in range(self.n_cluster):
            sse += np.sum((self.data[self.idx==i] - self.centroid[i])**2)
        return sse

    def predict(self, data: np.ndarray) -> int:
        """Predict class of given data from centroids.

        Args:
        =====
        data: numpy.ndarray
            Given data.

        Return:
        =======
        res: int
            Predicted cluster index.
        """
        pred_dim = len(data[0])
        if pred_dim != self.dim_data:
            print("Cannot predict class of different dimension data. Return -1")
            return -1
        pred_n = len(data)
        data_re = data.reshape(pred_n, 1, 1, pred_dim)

        dist = np.sum((data_re - self.centroid.reshape(1, self.n_cluster, self.dim_data))**2, axis=3)
        res = np.argmin(dist, axis=2).flatten()
        return res

class FCM(object):
    def __init__(self, data: np.ndarray, n_cluster: int, m_fuzzy: float, iter: int) -> None:
        """Fuzzy C-Means Algorithm for soft clustering(under construction).

        Args:
        =====
        data: numpy.ndarray
            Target data
        n_cluster: int
            Number of clusters
        m_fuzzy: float
            Fuzziness
        iter: int
            Max trial number.
        """
        self.n_cluster = n_cluster
        if self.n_cluster < 1:
            print("You need to set cluster number bigger than 1.")
            print("n_cluster automatically set to 2.")
            self.n_cluster = 2
        self.data = data
        self.m_fuzzy = m_fuzzy
        self.iter = iter
        self.dim_data = len(self.data[0])
        self.n_data = len(self.data)

    def fit(self):
        """Fuzzy C-Means Algorithm(https://en.wikipedia.org/wiki/Fuzzy_clustering)
        1. Choose a number of clusters.
        2. Assign coefficients randomly to each data point for being in the clusters.
        3. Repeat until the algorithm has converged.
            3-1. Compute the centroid for each cluster.
            3-2. For each data point, compute its coefficients of being in the clusters.
        """
        w = np.random.rand(self.n_data, self.n_cluster)
        mu = self.__calc_centroid(w)

        data_re = self.data.reshape(self.n_data, 1, 1, self.dim_data)
        for _ in tqdm(range(self.iter), desc="Fuzzy C-Means Fit Progress", ncols=0, unit='iter'):
            d_sq_list = np.sum((data_re - mu.reshape(1, self.n_cluster, self.dim_data))**2, axis=3)
            d_sq_list_a = d_sq_list.copy().reshape(self.n_data, self.n_cluster, 1)
            w = 1/np.sum(np.power(d_sq_list / d_sq_list_a, 1/(self.m_fuzzy-1)), axis=1)
            mu_next = self.__calc_centroid(w)
            mu = mu_next
        self.cluster = w
        self.mu = mu

    def __calc_centroid(self, w): 
        res = np.sum(w**self.m_fuzzy * self.data.T.reshape(self.dim_data,self.n_data,1), axis=1)/np.sum(w**self.m_fuzzy, axis=0)
        return res.T
    
    def __calc_sse(self):
        """SSE(Sum of Squared Errors of prediction)

        Calculate error of prediction.
        """
        sse = 0
        for i in range(self.n_data):
            for j in range(self.n_cluster):
                sse += self.cluster[i][j] * np.sum((self.data[i] - self.mu[j])**2)
        return sse

    def predict(self, data):
        pred_dim = len(data[0])
        if pred_dim != self.dim_data:
            print("Cannot predict class of different dimension data. Return -1")
            return -1
        pred_n = len(data)
        data_re = data.reshape(pred_n, 1, 1, pred_dim)
        d_sq_list = np.sum((data_re - self.mu.reshape(1, self.n_cluster, self.dim_data))**2, axis=3)
        d_sq_list_a = d_sq_list.copy().reshape(pred_n, self.n_cluster, 1)
        w = 1/np.sum(np.power(d_sq_list / d_sq_list_a, 1/(self.m_fuzzy-1)), axis=1)
        return w


def gen_random_data(n_cluster: int, size: int, dimension: int) -> np.ndarray:
    """Generate random data for clustering.

    Args:
    =====
    n_cluster: int
        number of clusters.
    size: int
        number of data of each cluster.
    dimension: int
        dimension of each data.

    Return:
    =======
    data: numpy.ndarray
        random data points shaped (size, dim).
    """
    def create_data(size, dimension):
        data = np.random.randn(size, dimension)
        m = np.random.randint(-400, 400, dimension)
        s = np.random.randint(50, 100, dimension)
        for d in data:
            for i in range(dimension):
                d[i] = d[i] * s[i] + m[i]
        return data
    for i in range(n_cluster):
        if i == 0:
            data = create_data(size, dimension)
        else:
            data = np.concatenate([data, create_data(size, dimension)], axis=0)
    np.random.shuffle(data)
    return data
