import numpy as np
import pytest

from weathercam.attributes.softmax import SoftmaxModel, fit, predict, predict_proba

X = np.array([[2.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
Y = np.array([0, 0, 1, 1])


def test_zero_model_is_uniform():
    model = SoftmaxModel(np.zeros((2, 3)), np.zeros(3))
    p = predict_proba(model, np.array([[5.0, -1.0]]))
    assert p == pytest.approx(np.full((1, 3), 1 / 3))


def test_known_probabilities():
    model = SoftmaxModel(np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros(2))
    p = predict_proba(model, np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert p[0] == pytest.approx([0.7310585786, 0.2689414214])
    assert p[1] == pytest.approx([0.5, 0.5])


def test_rows_sum_to_one():
    model = SoftmaxModel(np.array([[1.0, -2.0, 0.5], [0.3, 0.1, -1.0]]), np.array([0.1, 0.2, 0.3]))
    p = predict_proba(model, np.array([[1.0, 2.0], [-3.0, 4.0], [100.0, -100.0]]))
    assert p.sum(axis=1) == pytest.approx(np.ones(3))


def test_predict_is_argmax():
    model = SoftmaxModel(np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros(2))
    assert list(predict(model, np.array([[2.0, 0.0], [0.0, 2.0]]))) == [0, 1]


def test_fit_separates_toy_data():
    model = fit(X, Y, n_classes=2, l2=0.0, lr=0.5, epochs=200, seed=0)
    assert list(predict(model, X)) == [0, 0, 1, 1]
    assert model.weights.shape == (2, 2)
    assert model.bias.shape == (2,)


def test_fit_same_seed_same_weights():
    a = fit(X, Y, n_classes=2, l2=0.01, lr=0.5, epochs=50, seed=3)
    b = fit(X, Y, n_classes=2, l2=0.01, lr=0.5, epochs=50, seed=3)
    assert np.array_equal(a.weights, b.weights)
    assert np.array_equal(a.bias, b.bias)


def test_fit_label_out_of_range_raises():
    with pytest.raises(ValueError):
        fit(X, np.array([0, 0, 1, 2]), n_classes=2, l2=0.0, lr=0.5, epochs=5, seed=0)
