"""Kiểm thử phân cụm và thuật toán."""
import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import cluster_metrics, dunn_index
from src.models.segmentation import SegmentationModel, search_k


def test_cluster_metrics():
    X = np.array([[1.0, 1.0], [1.1, 1.1], [10.0, 10.0], [10.1, 10.1]])
    labels = np.array([0, 0, 1, 1])
    m = cluster_metrics(X, labels)
    assert m["silhouette"] > 0.8
    assert m["davies_bouldin"] < 0.2


def test_dunn_index():
    X = np.array([[1.0, 1.0], [1.1, 1.1], [10.0, 10.0], [10.1, 10.1]])
    labels = np.array([0, 0, 1, 1])
    d = dunn_index(X, labels)
    assert d > 1.0


def test_segmentation_model():
    rfm = pd.DataFrame({
        "Recency": [10, 15, 200, 300],
        "Frequency": [5, 4, 1, 1],
        "Monetary": [500, 450, 50, 40],
    })
    seg = SegmentationModel(k=2, random_state=42)
    labels = seg.fit(rfm)
    assert len(labels) == 4
    assert len(seg.label_map) == 2

    # Dự đoán cho khách hàng mới
    preds = seg.predict(rfm)
    assert "Segment" in preds.columns
    assert len(preds) == 4
