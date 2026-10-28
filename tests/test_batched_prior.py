"""Parity of vectorized priors with the scalar production transformation."""
import numpy as np
import pytest

from platon.fit_info import FitInfo


def test_mixed_prior_and_ordered_pairs_match_scalar():
    fit = FitInfo(dict(a=1., b=4., cold=500., hot=1000.))
    fit.add_uniform_fit_param("a", -7.3, 18.7)
    fit.add_gaussian_fit_param("b", 3.2)
    fit.add_ordered_uniform_fit_params("cold", "hot", 100., 3000.)
    cubes = np.random.default_rng(94).random((1024, 4))
    cubes[:2] = [[0, 0, 1, 0], [1, 1, 0, 1]]
    expected = np.array([fit._from_unit_interval_array(row) for row in cubes])
    np.testing.assert_array_equal(fit._from_unit_interval_batch(cubes), expected)
    assert fit._from_unit_interval_batch(cubes[:0]).shape == (0, 4)


def test_custom_scalar_prior_is_preserved():
    class Custom:
        def from_unit_interval(self, u):
            assert np.ndim(u) == 0
            return u ** 3

    fit = FitInfo(dict(x=0.))
    fit.add_uniform_fit_param("x", 0., 1.)
    fit.all_params["x"] = Custom()
    cubes = np.array([[0.2], [0.7]])
    np.testing.assert_array_equal(
        fit._from_unit_interval_batch(cubes),
        [fit._from_unit_interval_array(row) for row in cubes])


@pytest.mark.parametrize("cubes", [np.ones(2), np.ones((2, 3)),
                                   [[np.nan]], [[-0.1]], [[1.1]]])
def test_invalid_prior_batches(cubes):
    fit = FitInfo(dict(x=0.5))
    fit.add_uniform_fit_param("x", 0., 1.)
    with pytest.raises(ValueError):
        fit._from_unit_interval_batch(cubes)
