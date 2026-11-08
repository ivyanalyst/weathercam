import numpy as np
import pytest

from weathercam.retrieval.metric_head import MetricHead, fit, loss_and_grads, project


def _numeric(function, array, eps=1e-6):
    grad = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + eps
        plus = function()
        array[index] = original - eps
        minus = function()
        array[index] = original
        grad[index] = (plus - minus) / (2 * eps)
    return grad


def _clusters():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(3, 8)) * 3.0
    x = np.repeat(centers, 20, axis=0) + rng.normal(size=(60, 8)) * 0.3
    y = np.repeat(np.arange(3), 20)
    return x, y


def _fit(x, y, **overrides):
    args = dict(n_classes=3, dim=4, scale=10.0, lr=0.5, epochs=30, batch_size=16, seed=0)
    args.update(overrides)
    return fit(x, y, **args)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 4))
    y = np.array([0, 1, 2, 0, 1, 2])
    weights = rng.normal(size=(4, 3))
    prototypes = rng.normal(size=(3, 3))
    scale = 5.0
    l2 = 0.1
    _, d_weights, d_prototypes = loss_and_grads(weights, prototypes, x, y, scale, l2)
    numeric_w = _numeric(lambda: loss_and_grads(weights, prototypes, x, y, scale, l2)[0], weights)
    numeric_p = _numeric(lambda: loss_and_grads(weights, prototypes, x, y, scale, l2)[0], prototypes)
    np.testing.assert_allclose(d_weights, numeric_w, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(d_prototypes, numeric_p, rtol=1e-5, atol=1e-7)


def test_project_shape_and_unit_norm():
    x, y = _clusters()
    model = _fit(x, y, epochs=2)
    assert isinstance(model, MetricHead)
    z = project(model, x)
    assert z.shape == (60, 4)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-9)


def test_training_lowers_the_loss():
    x, y = _clusters()
    short = _fit(x, y, epochs=1)
    long = _fit(x, y, epochs=40)
    loss_short = loss_and_grads(short.weights, short.prototypes, x, y, 10.0)[0]
    loss_long = loss_and_grads(long.weights, long.prototypes, x, y, 10.0)[0]
    assert loss_long < loss_short


def test_same_seed_is_deterministic_and_other_seed_differs():
    x, y = _clusters()
    first = _fit(x, y, epochs=3, seed=7)
    second = _fit(x, y, epochs=3, seed=7)
    other = _fit(x, y, epochs=3, seed=8)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.prototypes, second.prototypes)
    assert not np.allclose(first.weights, other.weights)


def test_projection_separates_clusters():
    x, y = _clusters()
    z = project(_fit(x, y), x)
    sim = z @ z.T
    same = y[:, None] == y[None, :]
    diagonal = np.eye(len(y), dtype=bool)
    within = sim[same & ~diagonal].mean()
    between = sim[~same].mean()
    assert within > between + 0.5


def test_fit_rejects_bad_input():
    x, y = _clusters()
    with pytest.raises(ValueError):
        _fit(x, y[:-1])
    with pytest.raises(ValueError):
        _fit(x, y + 1)
    with pytest.raises(ValueError):
        _fit(x, y.astype(np.float64))
    with pytest.raises(ValueError):
        _fit(x, y, scale=0.0)
    with pytest.raises(ValueError):
        _fit(x, y, momentum=1.0)
    with pytest.raises(ValueError):
        _fit(x, y, l2=-0.1)
