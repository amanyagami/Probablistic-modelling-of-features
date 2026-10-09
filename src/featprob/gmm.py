"""Class-conditional Gaussian mixture models for feature-space OOD scoring.

One GMM is fitted per class on (penultimate-layer) features. The number of
components per class is chosen by BIC and covariances are regularised. Samples
are scored by class-conditional log-likelihood or by the squared Mahalanobis
distance to the nearest mixture component.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture

FloatArray = NDArray[np.float64]
Method = Literal["loglik", "mahalanobis"]

_SUPPORTED_COV = ("full", "diag", "spherical")


def _as_2d(x: ArrayLike, name: str = "X") -> FloatArray:
    """Convert input to a finite 2-D float64 array.

    Args:
        x: Array-like of shape (n_samples, n_features).
        name: Name used in error messages.

    Returns:
        The validated array.

    Raises:
        ValueError: If the array is not 2-D, is empty, or contains non-finite values.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _min_sq_mahalanobis(gmm: GaussianMixture, x: FloatArray) -> FloatArray:
    """Squared Mahalanobis distance from each sample to its nearest component.

    Args:
        gmm: A fitted ``GaussianMixture`` (``full``, ``diag`` or ``spherical``).
        x: Samples of shape (n_samples, n_features).

    Returns:
        Array of shape (n_samples,) with the minimum over components.
    """
    chol = gmm.precisions_cholesky_
    cov_type = gmm.covariance_type
    dists = np.empty((x.shape[0], gmm.n_components))
    for k in range(gmm.n_components):
        diff = x - gmm.means_[k]
        if cov_type == "full":
            y = diff @ chol[k]
            dists[:, k] = np.sum(y * y, axis=1)
        elif cov_type == "diag":
            dists[:, k] = np.sum((diff * chol[k]) ** 2, axis=1)
        else:  # spherical
            dists[:, k] = np.sum(diff * diff, axis=1) * chol[k] ** 2
    return dists.min(axis=1)


class ClassConditionalGMM:
    """Per-class Gaussian mixtures with BIC model selection.

    Attributes:
        classes_: Sorted array of classes that received a model.
        models_: Mapping from class label to the fitted ``GaussianMixture``.
        n_components_: Mapping from class label to the BIC-selected component count.
        skipped_classes_: Classes dropped for having too few samples.
        log_priors_: Log class priors (over fitted classes) estimated from counts.
    """

    def __init__(
        self,
        max_components: int = 4,
        covariance_type: str = "full",
        reg_covar: float = 1e-3,
        min_samples_per_class: int = 5,
        min_samples_per_component: int = 5,
        random_state: int = 0,
        n_jobs: int | None = None,
    ) -> None:
        """Initialise the model.

        Args:
            max_components: Largest number of components tried per class.
            covariance_type: One of ``full``, ``diag`` or ``spherical``.
            reg_covar: Non-negative ridge added to covariance diagonals. Keeps
                covariances invertible when features are collinear or constant.
            min_samples_per_class: Classes with fewer samples are skipped (with a warning).
            min_samples_per_component: Upper bound on components is
                ``n_class_samples // min_samples_per_component``.
            random_state: Seed controlling all GMM initialisations.
            n_jobs: Processes used to fit classes in parallel (joblib semantics; results
                are identical to ``None`` because each class uses the same seed).

        Raises:
            ValueError: If an argument is out of range.
        """
        if covariance_type not in _SUPPORTED_COV:
            raise ValueError(f"covariance_type must be one of {_SUPPORTED_COV}")
        if max_components < 1:
            raise ValueError("max_components must be >= 1")
        if reg_covar <= 0:
            raise ValueError("reg_covar must be > 0 for numerical stability")
        if min_samples_per_class < 2:
            raise ValueError("min_samples_per_class must be >= 2")
        self.max_components = max_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.min_samples_per_class = min_samples_per_class
        self.min_samples_per_component = max(1, min_samples_per_component)
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.classes_: NDArray[np.generic] = np.array([])
        self.models_: dict[int, GaussianMixture] = {}
        self.n_components_: dict[int, int] = {}
        self.skipped_classes_: list[int] = []
        self.log_priors_: FloatArray = np.array([])
        self._cal_raw: dict[str, FloatArray] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: ArrayLike, y: ArrayLike) -> ClassConditionalGMM:
        """Fit one BIC-selected GMM per class.

        Args:
            X: Features of shape (n_samples, n_features).
            y: Integer class labels of shape (n_samples,).

        Returns:
            ``self``.

        Raises:
            ValueError: On shape mismatch or if no class has enough samples.
        """
        x = _as_2d(X)
        labels = np.asarray(y)
        if labels.shape != (x.shape[0],):
            raise ValueError("y must have shape (n_samples,) matching X")
        self.models_, self.n_components_, self.skipped_classes_ = {}, {}, []
        fit_classes: list[int] = []
        blocks: list[FloatArray] = []
        for cls in np.unique(labels):
            xc = x[labels == cls]
            if xc.shape[0] < self.min_samples_per_class:
                warnings.warn(
                    f"class {cls} has {xc.shape[0]} samples "
                    f"(< {self.min_samples_per_class}); skipping",
                    stacklevel=2,
                )
                self.skipped_classes_.append(int(cls))
                continue
            fit_classes.append(int(cls))
            blocks.append(xc)
        fitted_models = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_class)(b) for b in blocks)
        counts: list[int] = []
        for cls, model, xc in zip(fit_classes, fitted_models, blocks, strict=True):
            self.models_[cls] = model
            self.n_components_[cls] = model.n_components
            counts.append(xc.shape[0])
        if not self.models_:
            raise ValueError("no class has enough samples to fit a mixture")
        self.classes_ = np.array(sorted(self.models_))
        c = np.array([counts[i] for i in np.argsort(list(self.models_))], dtype=float)
        self.log_priors_ = np.log(c / c.sum())
        self._cal_raw = {}
        return self

    def _fit_class(self, xc: FloatArray) -> GaussianMixture:
        """Fit candidate mixtures on one class and return the lowest-BIC one."""
        k_max = max(1, min(self.max_components, xc.shape[0] // self.min_samples_per_component))
        best: GaussianMixture | None = None
        best_bic = np.inf
        for k in range(1, k_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                n_init=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ConvergenceWarning on tiny classes
                gmm.fit(xc)
            bic = gmm.bic(xc)
            if bic < best_bic:
                best, best_bic = gmm, bic
        assert best is not None
        return best

    def _check_fitted(self) -> None:
        if not self.models_:
            raise RuntimeError("model is not fitted; call fit() first")

    # --------------------------------------------------------------- scoring
    def class_log_likelihood(self, X: ArrayLike) -> FloatArray:
        """Class-conditional log-likelihood ``log p(x | c)``.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes), columns ordered as ``classes_``.
        """
        self._check_fitted()
        x = _as_2d(X)
        return np.column_stack([self.models_[int(c)].score_samples(x) for c in self.classes_])

    def predict(self, X: ArrayLike) -> NDArray[np.generic]:
        """Predict the class maximising ``log p(x|c) + log p(c)``.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        joint = self.class_log_likelihood(X) + self.log_priors_
        return self.classes_[np.argmax(joint, axis=1)]

    def mahalanobis_to_nearest(self, X: ArrayLike) -> FloatArray:
        """Squared Mahalanobis distance to the nearest component of each class.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes).
        """
        self._check_fitted()
        x = _as_2d(X)
        return np.column_stack(
            [_min_sq_mahalanobis(self.models_[int(c)], x) for c in self.classes_]
        )

    def raw_anomaly_score(self, X: ArrayLike, method: Method = "loglik") -> FloatArray:
        """Uncalibrated anomaly score; larger means more anomalous.

        Args:
            X: Features of shape (n_samples, n_features).
            method: ``loglik`` uses ``-max_c log p(x|c)``; ``mahalanobis`` uses
                ``min_c`` of the nearest-component squared Mahalanobis distance.

        Returns:
            Array of shape (n_samples,).

        Raises:
            ValueError: If ``method`` is unknown.
        """
        if method == "loglik":
            return -self.class_log_likelihood(X).max(axis=1)
        if method == "mahalanobis":
            return self.mahalanobis_to_nearest(X).min(axis=1)
        raise ValueError(f"unknown method {method!r}")

    def calibrate(self, X_cal: ArrayLike, method: Method | None = None) -> ClassConditionalGMM:
        """Store raw scores of held-out in-distribution data for calibration.

        Calibration data must not be the data used in :meth:`fit`, otherwise the
        in-distribution scores are optimistically low.

        Args:
            X_cal: Held-out in-distribution features.
            method: Score to calibrate; ``None`` calibrates both.

        Returns:
            ``self``.
        """
        methods: tuple[Method, ...] = ("loglik", "mahalanobis") if method is None else (method,)
        for m in methods:
            self._cal_raw[m] = np.sort(self.raw_anomaly_score(X_cal, m))
        return self

    def _cal(self, method: Method) -> FloatArray:
        if method not in self._cal_raw:
            raise RuntimeError(f"call calibrate() for method {method!r} first")
        return self._cal_raw[method]

    def anomaly_score(self, X: ArrayLike, method: Method = "loglik") -> FloatArray:
        """Calibrated anomaly score: robust z-score against held-out ID scores.

        ``(raw - median) / (1.4826 * MAD)`` of the calibration scores. It is a
        strictly increasing affine map of the raw score, so rankings (and AUROC)
        are unchanged, but values are comparable across methods and datasets.

        Args:
            X: Features of shape (n_samples, n_features).
            method: See :meth:`raw_anomaly_score`.

        Returns:
            Array of shape (n_samples,); ~0 for typical ID data, large when anomalous.
        """
        cal = self._cal(method)
        med = np.median(cal)
        scale = 1.4826 * np.median(np.abs(cal - med))
        scale = max(scale, 1e-12)
        return (self.raw_anomaly_score(X, method) - med) / scale

    def p_value(self, X: ArrayLike, method: Method = "loglik") -> FloatArray:
        """Empirical p-value: fraction of ID calibration scores at least as anomalous.

        Args:
            X: Features of shape (n_samples, n_features).
            method: See :meth:`raw_anomaly_score`.

        Returns:
            Array in (0, 1]; small values indicate OOD. Saturates at ``1/(n+1)``.
        """
        cal = self._cal(method)
        raw = self.raw_anomaly_score(X, method)
        n_ge = cal.size - np.searchsorted(cal, raw, side="left")
        return (n_ge + 1.0) / (cal.size + 1.0)

    def threshold_at_tpr(self, tpr: float = 0.95, method: Method = "loglik") -> float:
        """Raw-score threshold that keeps ``tpr`` of calibration ID data.

        Args:
            tpr: Target in-distribution true-positive rate in (0, 1).
            method: See :meth:`raw_anomaly_score`.

        Returns:
            Samples with raw anomaly score above this value are flagged OOD.
        """
        if not 0.0 < tpr < 1.0:
            raise ValueError("tpr must be in (0, 1)")
        return float(np.quantile(self._cal(method), tpr))

    def evaluate(
        self, X_id: ArrayLike, X_ood: ArrayLike, method: Method = "loglik"
    ) -> dict[str, float]:
        """Evaluate OOD detection of held-out ID versus OOD features.

        Args:
            X_id: Held-out in-distribution features.
            X_ood: Out-of-distribution (or anomalous) features.
            method: See :meth:`raw_anomaly_score`.

        Returns:
            Dict from :func:`evaluate_ood`.
        """
        return evaluate_ood(
            self.raw_anomaly_score(X_id, method), self.raw_anomaly_score(X_ood, method)
        )


def evaluate_ood(
    scores_id: ArrayLike, scores_ood: ArrayLike, tpr_level: float = 0.95
) -> dict[str, float]:
    """Compute threshold-free OOD metrics from anomaly scores.

    Convention: higher score means more anomalous. For ``auroc`` and ``aupr_out``
    OOD is the positive class; for ``aupr_in`` and ``fpr_at_95tpr`` in-distribution
    is positive (as in the Mahalanobis/ODIN literature), so ``fpr_at_95tpr`` is the
    fraction of OOD samples accepted when 95% of ID samples are accepted.

    Args:
        scores_id: Anomaly scores of in-distribution samples.
        scores_ood: Anomaly scores of OOD samples.
        tpr_level: ID true-positive rate at which FPR is measured.

    Returns:
        Dict with keys ``auroc``, ``aupr_in``, ``aupr_out`` and ``fpr_at_95tpr``
        (key name is fixed even if ``tpr_level`` differs).

    Raises:
        ValueError: If either score set is empty or contains NaN.
    """
    s_id = np.asarray(scores_id, dtype=float).ravel()
    s_ood = np.asarray(scores_ood, dtype=float).ravel()
    if s_id.size == 0 or s_ood.size == 0:
        raise ValueError("scores_id and scores_ood must be non-empty")
    if np.isnan(s_id).any() or np.isnan(s_ood).any():
        raise ValueError("scores contain NaN")
    scores = np.concatenate([s_id, s_ood])
    is_ood = np.concatenate([np.zeros(s_id.size), np.ones(s_ood.size)])
    auroc = float(roc_auc_score(is_ood, scores))
    aupr_out = float(average_precision_score(is_ood, scores))
    aupr_in = float(average_precision_score(1 - is_ood, -scores))
    fpr, tpr, _ = roc_curve(1 - is_ood, -scores)  # ID positive
    fpr_at = float(fpr[np.searchsorted(tpr, tpr_level, side="left")])
    return {"auroc": auroc, "aupr_in": aupr_in, "aupr_out": aupr_out, "fpr_at_95tpr": fpr_at}
