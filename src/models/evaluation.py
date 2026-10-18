"""Các chỉ số đánh giá phân cụm + cài đặt Fuzzy C-Means (không cần thư viện ngoài)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             pairwise_distances, silhouette_score)


def cluster_metrics(X: np.ndarray, labels: np.ndarray, sil_sample: int | None = None,
                    random_state: int = 42) -> dict:
    """Silhouette (↑), Davies-Bouldin (↓), Calinski-Harabasz (↑). Bỏ điểm nhiễu (-1)."""
    mask = labels != -1
    X, labels = X[mask], labels[mask]
    if len(set(labels)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    kw = {}
    if sil_sample and len(X) > sil_sample:
        kw = {"sample_size": sil_sample, "random_state": random_state}
    return {
        "silhouette": float(silhouette_score(X, labels, **kw)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Khoảng cách nhỏ nhất giữa 2 cụm / đường kính lớn nhất của 1 cụm (↑). Nên gọi trên mẫu nhỏ."""
    mask = labels != -1
    X, labels = X[mask], labels[mask]
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.nan
    idx = {lb: np.where(labels == lb)[0] for lb in uniq}
    max_intra = max((pairwise_distances(X[i]).max() for i in idx.values() if len(i) > 1), default=0.0)
    min_inter = min(pairwise_distances(X[idx[a]], X[idx[b]]).min()
                    for n, a in enumerate(uniq) for b in uniq[n + 1:])
    return float(min_inter / max_intra) if max_intra > 0 else np.nan


def fuzzy_c_means(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 150,
                  tol: float = 1e-5, random_state: int = 42) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    u = rng.rand(X.shape[0], c)
    u /= u.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        um = u ** m
        centers = (um.T @ X) / um.sum(axis=0)[:, None]
        dist = np.fmax(np.linalg.norm(X[:, None, :] - centers[None], axis=2), 1e-10)
        new_u = 1.0 / dist ** (2.0 / (m - 1))
        new_u /= new_u.sum(axis=1, keepdims=True)
        done = np.linalg.norm(new_u - u) < tol
        u = new_u
        if done:
            break
    return np.argmax(u, axis=1)
