import numpy as np
import pytest


def make_mixture(
    n_per_class: int = 300, n_classes: int = 3, dim: int = 8, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated Gaussian classes (class means 6 apart along random directions)."""
    centers = np.random.default_rng(1234).normal(size=(n_classes, dim)) * 6.0  # shared
    rng = np.random.default_rng(seed)
    x = np.concatenate([c + rng.normal(size=(n_per_class, dim)) for c in centers])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return x, y


@pytest.fixture(scope="session")
def data() -> dict[str, np.ndarray]:
    x_train, y_train = make_mixture(seed=0)
    x_cal, _ = make_mixture(seed=1)
    x_test, y_test = make_mixture(seed=2)
    # OOD: a cluster far from every class (and offset in all dims)
    rng = np.random.default_rng(3)
    x_ood = rng.normal(size=(300, 8)) * 1.5 + 25.0
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_cal": x_cal,
        "x_test": x_test,
        "y_test": y_test,
        "x_ood": x_ood,
    }
