import warnings

import numpy as np
import pytest

from featprob import ClassConditionalGMM, evaluate_ood


def fitted(data: dict[str, np.ndarray], **kw: object) -> ClassConditionalGMM:
    return ClassConditionalGMM(**kw).fit(data["x_train"], data["y_train"])  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["loglik", "mahalanobis"])
def test_ood_auroc_high(data: dict[str, np.ndarray], method: str) -> None:
    model = fitted(data)
    res = model.evaluate(data["x_test"], data["x_ood"], method=method)  # type: ignore[arg-type]
    assert res["auroc"] > 0.9
    assert res["aupr_out"] > 0.9
    assert res["fpr_at_95tpr"] < 0.1


def test_classification_accuracy(data: dict[str, np.ndarray]) -> None:
    model = fitted(data)
    assert (model.predict(data["x_test"]) == data["y_test"]).mean() > 0.95


def test_bic_selects_multiple_components_for_bimodal_class() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-8, 1, (200, 2)), rng.normal(8, 1, (200, 2))])
    model = ClassConditionalGMM(max_components=4).fit(x, np.zeros(400, dtype=int))
    assert model.n_components_[0] == 2


def test_shapes_and_nearest_component_mahalanobis(data: dict[str, np.ndarray]) -> None:
    model = fitted(data)
    ll = model.class_log_likelihood(data["x_test"])
    md = model.mahalanobis_to_nearest(data["x_test"])
    assert ll.shape == md.shape == (len(data["x_test"]), 3)
    # True class of a test point should typically have the smallest distance.
    assert (md.argmin(axis=1) == data["y_test"]).mean() > 0.95
    assert np.all(md >= 0)


def test_calibration_is_rank_invariant(data: dict[str, np.ndarray]) -> None:
    model = fitted(data).calibrate(data["x_cal"])
    x = np.vstack([data["x_test"], data["x_ood"]])
    for method in ("loglik", "mahalanobis"):
        raw = model.raw_anomaly_score(x, method)  # type: ignore[arg-type]
        cal = model.anomaly_score(x, method)  # type: ignore[arg-type]
        assert np.array_equal(np.argsort(raw, kind="stable"), np.argsort(cal, kind="stable"))
    # ID data should be centred near 0 after calibration, OOD far above.
    assert abs(np.median(model.anomaly_score(data["x_test"]))) < 1.0
    assert np.median(model.anomaly_score(data["x_ood"])) > 5.0


def test_p_value_and_threshold(data: dict[str, np.ndarray]) -> None:
    model = fitted(data).calibrate(data["x_cal"], "loglik")
    assert model.p_value(data["x_ood"]).max() <= 2.0 / (len(data["x_cal"]) + 1)
    thr = model.threshold_at_tpr(0.95)
    kept = (model.raw_anomaly_score(data["x_test"]) <= thr).mean()
    assert 0.9 < kept < 0.99
    with pytest.raises(ValueError):
        model.threshold_at_tpr(1.0)


def test_sample_order_and_label_permutation_invariance(data: dict[str, np.ndarray]) -> None:
    model = fitted(data)
    x = data["x_test"]
    perm = np.random.default_rng(0).permutation(len(x))
    np.testing.assert_allclose(
        model.raw_anomaly_score(x)[perm], model.raw_anomaly_score(x[perm]), rtol=1e-9
    )
    # Relabelling classes must not change class-agnostic scores.
    relabeled = ClassConditionalGMM().fit(data["x_train"], (data["y_train"] + 7) * 3)
    np.testing.assert_allclose(model.raw_anomaly_score(x), relabeled.raw_anomaly_score(x))


def test_reproducible_with_seed(data: dict[str, np.ndarray]) -> None:
    a = fitted(data, random_state=5).raw_anomaly_score(data["x_ood"])
    b = fitted(data, random_state=5).raw_anomaly_score(data["x_ood"])
    np.testing.assert_array_equal(a, b)


def test_class_with_too_few_samples_is_skipped(data: dict[str, np.ndarray]) -> None:
    x = np.vstack([data["x_train"], data["x_ood"][:2]])
    y = np.concatenate([data["y_train"], [99, 99]])
    with pytest.warns(UserWarning, match="skipping"):
        model = ClassConditionalGMM().fit(x, y)
    assert model.skipped_classes_ == [99]
    assert list(model.classes_) == [0, 1, 2]


def test_all_classes_too_small_raises() -> None:
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="enough samples"):
        ClassConditionalGMM(min_samples_per_class=10).fit(np.zeros((4, 2)), np.array([0, 0, 1, 1]))


def test_singular_covariance_is_regularised() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(100, 3))
    # Duplicated and constant columns make the sample covariance exactly singular.
    x = np.column_stack([base, base[:, 0], np.ones(100)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ClassConditionalGMM(max_components=2).fit(x, np.zeros(100, dtype=int))
    score = model.raw_anomaly_score(x)
    assert np.all(np.isfinite(score))
    assert np.all(np.isfinite(model.mahalanobis_to_nearest(x)))


def test_input_validation(data: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError):
        ClassConditionalGMM().fit(np.array([[np.nan, 1.0]] * 10), np.zeros(10))
    with pytest.raises(ValueError):
        ClassConditionalGMM().fit(data["x_train"], data["y_train"][:-1])
    with pytest.raises(RuntimeError):
        ClassConditionalGMM().class_log_likelihood(data["x_test"])
    with pytest.raises(RuntimeError):
        fitted(data).anomaly_score(data["x_test"])  # not calibrated
    with pytest.raises(ValueError):
        ClassConditionalGMM(covariance_type="tied")
    with pytest.raises(ValueError):
        fitted(data).raw_anomaly_score(data["x_test"], "bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("cov", ["diag", "spherical"])
def test_other_covariance_types(data: dict[str, np.ndarray], cov: str) -> None:
    res = fitted(data, covariance_type=cov).evaluate(data["x_test"], data["x_ood"], "mahalanobis")
    assert res["auroc"] > 0.9


def test_evaluate_ood_known_values() -> None:
    res = evaluate_ood([0.0, 1.0, 2.0, 3.0], [10.0, 11.0])
    assert res["auroc"] == 1.0
    assert res["aupr_out"] == 1.0
    assert res["fpr_at_95tpr"] == 0.0
    inverted = evaluate_ood([10.0, 11.0], [0.0, 1.0])
    assert inverted["auroc"] == 0.0
    tied = evaluate_ood([1.0, 1.0], [1.0, 1.0])
    assert tied["auroc"] == 0.5
    with pytest.raises(ValueError):
        evaluate_ood([], [1.0])
