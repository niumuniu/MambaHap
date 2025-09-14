import torch
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import time

class SpectralClustering():
    def __init__(self, n_clusters: int, max_iter=300, tol=1e-3, verbose=False, device=torch.device("cpu")):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.device = device

    def fit(self, affinity: torch.Tensor):
        n = affinity.size(0)
        affinity = affinity.to(self.device)

        degree = torch.sum(affinity, dim=1)
        degree[degree < 1e-10] = 1e-10

        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree))
        L = torch.eye(n, device=self.device) - D_inv_sqrt @ affinity @ D_inv_sqrt

        L = (L + L.T) / 2

        eps = torch.linalg.norm(L, ord='fro') * 1e-5
        L = L + eps * torch.eye(n, device=self.device)

        try:
            eigvals, eigvecs = torch.linalg.eigh(L)
        except RuntimeError:
            print("⚠️ GPU eigh failed, switching to CPU.")
            from scipy.linalg import eigh
            L_cpu = L.detach().cpu().numpy()
            eigvals_np, eigvecs_np = eigh(L_cpu)
            eigvecs = torch.from_numpy(eigvecs_np).float().to(self.device)
            eigvals = torch.from_numpy(eigvals_np).float().to(self.device)

        U = eigvecs[:, :self.n_clusters]
        U_norm = U / (torch.norm(U, dim=1, keepdim=True) + 1e-10)

        U_norm_cpu = U_norm.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol, n_init=10)
        self.labels_ = kmeans.fit_predict(U_norm_cpu)

        return self


if __name__ == '__main__':
    n_clusters = 3

    X, y = make_blobs(n_samples=1000, centers=n_clusters, random_state=42)
    X = torch.from_numpy(X).float()

    sigma = 1.0
    pairwise_dist = torch.cdist(X, X, p=2)
    affinity = torch.exp(-pairwise_dist**2 / (2 * sigma ** 2))

    sc = SpectralClustering(n_clusters=n_clusters, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    t0 = time.time()
    sc.fit(affinity)
    t1 = time.time()
    print(f"谱聚类耗时: {t1-t0:.3f}秒")

    labels = sc.labels_
    plt.figure()
    for i in range(n_clusters):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i}')
    plt.legend()
    plt.title('Spectral Clustering Result')
    plt.show()
