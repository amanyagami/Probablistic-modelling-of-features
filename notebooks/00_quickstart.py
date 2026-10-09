# ---
# jupytext:
#   text_representation:
#     extension: .py
#     format_name: percent
# ---

# %% [markdown]
# # featprob quickstart
# End-to-end on synthetic data: fit a class-conditional GMM, calibrate on held-out
# in-distribution data, and measure OOD detection. No torch or datasets needed.

# %%
import matplotlib

matplotlib.use("Agg")
import numpy as np

from featprob import ClassConditionalGMM
from featprob.viz import project_tsne, scatter_embedding

rng = np.random.default_rng(0)
centers = rng.normal(size=(3, 8)) * 6.0


def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([c + rng.normal(size=(n, 8)) for c in centers])
    return x, np.repeat(np.arange(3), n)


x_train, y_train = sample(300)
x_cal, _ = sample(200)  # held-out ID data used only for calibration
x_test, _ = sample(200)
x_ood = rng.normal(size=(300, 8)) * 1.5 + 20.0

# %% [markdown]
# ## Fit, calibrate, evaluate

# %%
model = ClassConditionalGMM(max_components=3, random_state=0).fit(x_train, y_train)
model.calibrate(x_cal)
print("components per class:", model.n_components_)
for method in ("loglik", "mahalanobis"):
    print(method, model.evaluate(x_test, x_ood, method=method))
print("flag threshold (95% ID kept):", model.threshold_at_tpr(0.95))
print("median calibrated score  ID:", np.median(model.anomaly_score(x_test)))
print("median calibrated score OOD:", np.median(model.anomaly_score(x_ood)))

# %% [markdown]
# ## Visualise

# %%
emb = project_tsne(np.vstack([x_test, x_ood]), perplexity=30, seed=0)
labels = np.r_[np.repeat("ID", len(x_test)), np.repeat("OOD", len(x_ood))]
ax = scatter_embedding(emb, labels)
ax.figure.savefig("quickstart_tsne.png", dpi=100)
