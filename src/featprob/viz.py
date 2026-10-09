"""2-D projection and plotting helpers with fixed seeds."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEFAULT_SEED = 0


def project_pca(
    X: ArrayLike, n_components: int = 2, seed: int = DEFAULT_SEED
) -> NDArray[np.float64]:
    """Project with PCA.

    Args:
        X: Features of shape (n_samples, n_features).
        n_components: Output dimensionality.
        seed: Random seed (affects the randomized SVD solver only).

    Returns:
        Embedding of shape (n_samples, n_components).
    """
    return PCA(n_components=n_components, random_state=seed).fit_transform(np.asarray(X))


def project_tsne(
    X: ArrayLike,
    n_components: int = 2,
    perplexity: float = 30.0,
    seed: int = DEFAULT_SEED,
) -> NDArray[np.float64]:
    """Project with t-SNE (scikit-learn), PCA-initialised and seeded.

    Args:
        X: Features of shape (n_samples, n_features).
        n_components: Output dimensionality.
        perplexity: Requested perplexity; clipped to ``n_samples - 1`` when too large.
        seed: Random seed.

    Returns:
        Embedding of shape (n_samples, n_components).

    Raises:
        ValueError: If fewer than 3 samples are given.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.shape[0] < 3:
        raise ValueError("t-SNE needs at least 3 samples")
    perp = float(min(perplexity, x.shape[0] - 1))
    return TSNE(
        n_components=n_components, perplexity=perp, init="pca", random_state=seed
    ).fit_transform(x)


def project_umap(
    X: ArrayLike,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> NDArray[np.float64]:
    """Project with UMAP (requires the ``viz`` extra). Seeded, so single-threaded.

    Args:
        X: Features of shape (n_samples, n_features).
        n_components: Output dimensionality.
        n_neighbors: UMAP neighbourhood size.
        min_dist: UMAP minimum distance.
        seed: Random seed.

    Returns:
        Embedding of shape (n_samples, n_components).

    Raises:
        ImportError: If ``umap-learn`` is not installed.
    """
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("umap-learn is required; install with `uv sync --extra viz`") from exc
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed
    )
    return np.asarray(reducer.fit_transform(np.asarray(X)))


def scatter_embedding(
    emb: ArrayLike, labels: ArrayLike | None = None, ax: Any = None, s: float = 8.0
) -> Any:
    """Scatter a 2-D embedding coloured by label.

    Args:
        emb: Array of shape (n_samples, 2).
        labels: Optional labels used for colouring and legend.
        ax: Optional matplotlib axes; a new one is created if omitted.
        s: Marker size.

    Returns:
        The matplotlib axes.
    """
    import matplotlib.pyplot as plt

    e = np.asarray(emb)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if labels is None:
        ax.scatter(e[:, 0], e[:, 1], s=s)
    else:
        lab = np.asarray(labels)
        for u in np.unique(lab):
            m = lab == u
            ax.scatter(e[m, 0], e[m, 1], s=s, label=str(u))
        ax.legend(markerscale=2, fontsize="small")
    return ax
