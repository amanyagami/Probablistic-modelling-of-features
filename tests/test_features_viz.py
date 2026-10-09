import importlib
import subprocess
import sys

import numpy as np
import pytest

from featprob.viz import project_pca, project_tsne, scatter_embedding


def test_features_imports_without_torch() -> None:
    code = """
import sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("blocked")

sys.meta_path.insert(0, Block())
import featprob
import featprob.features as f
import featprob.viz

try:
    f.FeatureExtractor(object())
except ImportError:
    pass
else:
    raise SystemExit(1)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_feature_extractor_penultimate() -> None:
    torch = pytest.importorskip("torch")
    from featprob.features import FeatureExtractor

    net = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3))
    x = torch.randn(5, 4)
    with FeatureExtractor(net) as ext:
        f = ext.extract_batch(x)
    expected = torch.relu(net[0](x)).detach().numpy()
    np.testing.assert_allclose(f, expected, rtol=1e-5)
    assert importlib.import_module("featprob.features")


def test_tsne_reproducible_and_perplexity_clipped() -> None:
    x = np.random.default_rng(0).normal(size=(40, 5))
    a = project_tsne(x, perplexity=500, seed=1)
    b = project_tsne(x, perplexity=500, seed=1)
    assert a.shape == (40, 2)
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError):
        project_tsne(x[:2])


def test_pca_and_scatter() -> None:
    import matplotlib

    matplotlib.use("Agg")
    x = np.random.default_rng(0).normal(size=(30, 5))
    emb = project_pca(x)
    ax = scatter_embedding(emb, np.arange(30) % 3)
    assert ax is not None
