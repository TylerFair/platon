"""CPU parity on small wavelength grids, using real opacity/stellar data."""
import unittest
import json
from pathlib import Path
from unittest import mock

import numpy as np

from platon._batch import BatchEvaluator
from platon.combined_retriever import CombinedRetriever
from platon.constants import AU, M_jup, R_jup, R_sun
from platon.errors import AtmosphereError
from platon.smc_sampler import GaussianModel
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.terminator import TerminatorSector, TwoSectorTerminator
from platon.TP_profile import Profile
from platon.transit_depth_calculator import TransitDepthCalculator


class TestBatchedRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bins = 1e-6 * np.array([[0.5, 0.7], [1.1, 1.4], [2.0, 2.4], [3.2, 4.0]])
        cls.transit = TransitDepthCalculator(include_opacities=['H2O', 'CH4'], downsample=20)
        cls.transit.change_wavelength_bins(cls.bins, max_points_per_bin=3)
        cls.eclipse = EclipseDepthCalculator(include_opacities=['H2O', 'CH4'], downsample=100)
        cls.eclipse.change_wavelength_bins(cls.bins)

    def fit_info(self, **kwargs):
        fi = CombinedRetriever.get_default_fit_info(
            R_sun, M_jup, R_jup, T=1000, T_star=5500, **kwargs)
        fi.add_uniform_fit_param('Rp', 0.95 * R_jup, 1.05 * R_jup)
        fi.add_uniform_fit_param('T', 800, 1400)
        fi.add_uniform_fit_param('error_excess', 0, 1e-4)
        return fi

    def check(self, fi, transit=True, eclipse=False, calc=None, zero_opacities=[],
              extra_params=None):
        retriever = CombinedRetriever()
        cubes = np.random.default_rng(44).uniform(0.001, 0.999, (50, len(fi.fit_param_names)))
        params = fi._from_unit_interval_batch(cubes)
        np.testing.assert_array_equal(params, [fi._from_unit_interval_array(u) for u in cubes])
        if extra_params is not None:
            params = np.vstack([params, extra_params])
        args = (calc or self.transit, self.eclipse, fi,
                np.full(4, 0.01) if transit else None,
                np.full(4, 1e-4) if transit else None,
                np.full(4, 0.001) if eclipse else None,
                np.full(4, 1e-4) if eclipse else None)
        evaluator = BatchEvaluator(16)
        mean, var = retriever._ln_like_batch(params, *args, evaluator=evaluator,
                                            zero_opacities=zero_opacities)
        self.assertTrue(evaluator._cores, "Must execute the batched device path")
        data = np.concatenate([d for d in [args[3], args[5]] if d is not None])
        batch_lnlikes = GaussianModel(None, data).loglike_from(mean, var)
        n_valid = 0
        for i, row in enumerate(params):
            scalar = retriever._ln_like(row, *args, ret_model=True, zero_opacities=zero_opacities)
            lnlike = retriever._ln_like(row, *args, zero_opacities=zero_opacities)
            if isinstance(scalar, tuple):
                n_valid += 1
                np.testing.assert_allclose(mean[i], scalar[0], rtol=1e-6, atol=1e-10)
                np.testing.assert_array_equal(var[i], scalar[1])
                self.assertAlmostEqual(batch_lnlikes[i], lnlike, delta=1e-3)
            else:
                self.assertEqual(lnlike, -np.inf)
                self.assertEqual(batch_lnlikes[i], -np.inf)
                self.assertTrue(np.all(np.isnan(mean[i])))
                self.assertTrue(np.all(np.isnan(var[i])))
        self.assertGreater(n_valid, 0)
        # Exercise padding, cache reuse, and empty batches.
        one = retriever._ln_like_batch(params[:1], *args, evaluator=evaluator,
                                       zero_opacities=zero_opacities)
        np.testing.assert_array_equal(one[0], mean[:1])
        empty = retriever._ln_like_batch(params[:0], *args, evaluator=evaluator)
        self.assertEqual(empty[0].shape, (0, mean.shape[1]))

    def test_isothermal_equilibrium(self):
        fi = self.fit_info(log_P_quench=3, log_CH4_mult=0.3)
        fi.add_uniform_fit_param('logZ', -0.5, 1)
        fi.add_uniform_fit_param('CO_ratio', 0.3, 1.2)
        self.check(fi, zero_opacities=['CH4'])

    def free_guillot(self, **kwargs):
        fi = self.fit_info(logZ=None, CO_ratio=None, fit_vmr=True,
                           transit_profile_type='guillot', beta=0.8,
                           a=0.05 * AU, log_k_th=-3, log_gamma=-1, T_int=100, **kwargs)
        fi.add_uniform_fit_param('beta', 0.7, 0.9)
        fi.add_uniform_fit_param('log_gamma', -1.2, -0.5)
        fi.add_gases_vmr(['H2O', 'CH4', 'H2-He'], 1e-7, 1e-3)
        return fi

    def test_guillot_free_vmr(self):
        self.check(self.free_guillot())

    def test_patchy_clouds(self):
        fi = self.free_guillot(log_cloudtop_P=4, log_scatt_factor=2, scatt_slope=3)
        fi.add_uniform_fit_param('cloud_fraction', 0, 1)
        self.check(fi)
        for fraction in (0, 1):
            self.check(self.free_guillot(cloud_fraction=fraction, log_cloudtop_P=4))

    def test_two_hets_and_visits_offsets(self):
        fi = self.fit_info(T_het=4500, het_cov_frac=0.03, T_het2=6500,
                           het2_cov_frac=0.02, transit_visits={'a': (0, 2), 'b': (2, 4)},
                           transit_offsets={'offset': (2, 4)})
        for name, lo, hi in [('a.het_cov_frac', 0, 0.1), ('b.T_het2', 6000, 7000),
                             ('offset', -1e-4, 1e-4)]:
            fi.add_uniform_fit_param(name, lo, hi)
        self.check(fi)

    def test_stellar_branches(self):
        fi = self.fit_info(T_het=4500, het_cov_frac=0.03)
        fi.add_uniform_fit_param('T_star', 2000, 13000)
        self.check(fi)

    def test_eclipse_only(self):
        self.check(self.fit_info(), transit=False, eclipse=True)

    def test_transit_and_eclipse(self):
        fi = self.fit_info(T_transit=900, eclipse_offsets={'offset': (2, 4)})
        fi.add_uniform_fit_param('offset', -1e-4, 1e-4)
        self.check(fi, eclipse=True)

    def test_clr(self):
        fi = self.fit_info(logZ=None, CO_ratio=None, fit_clr=True)
        fi.add_gases_clr(['H2O', 'CH4', 'H2-He'], low_lim=1e-8)
        self.check(fi)

    def test_invalid_vmr_fill(self):
        fi = self.fit_info(logZ=None, CO_ratio=None, fit_vmr=True)
        fi.add_gases_vmr(['H2O', 'CH4', 'H2-He'], 1e-7, 10)
        self.check(fi)

    def test_mie(self):
        fi = self.fit_info(n=1.5, log_k=-2, log_number_density=4,
                           log_part_size=-6, log_cloudtop_P=5)
        fi.add_uniform_fit_param('log_part_size', -6.5, -5.5)
        self.check(fi, eclipse=True)

    def test_parametric(self):
        fi = self.fit_info(transit_profile_type='parametric', profile_type='parametric',
                           T0=1000, log_P1=2, alpha1=0.5, alpha2=0.5,
                           log_P3=5, T3=1500)
        for name, lo, hi in [('T0', 900, 1100), ('T3', 1400, 1600),
                             ('log_P1', 1, 3), ('log_P3', 4, 6),
                             ('alpha1', 0.4, 0.6), ('alpha2', 0.4, 0.6)]:
            fi.add_uniform_fit_param(name, lo, hi)
        self.check(fi, eclipse=True)

    def test_radiative_solution(self):
        fi = self.fit_info(transit_profile_type='radiative_solution',
                           profile_type='radiative_solution', a=0.05 * AU,
                           beta=0.8, log_k_th=-3, log_gamma=-1, log_gamma2=-0.5,
                           alpha=0.3, T_int=100)
        fi.add_uniform_fit_param('alpha', 0.1, 0.5)
        self.check(fi, eclipse=True)

    def test_two_sector_fixed_fraction(self):
        term = TwoSectorTerminator(TerminatorSector(Profile.isothermal(900), 1e4),
                                   TerminatorSector(Profile.isothermal(1300), 1e6), 0.3)
        fi = self.fit_info(transit_terminator=term)
        fi.add_uniform_fit_param('sector1.T', 800, 1400)
        fi.add_uniform_fit_param('sector2.T', 800, 1400)
        self.check(fi)

    def test_sector_endpoints_and_swaps(self):
        term = TwoSectorTerminator(TerminatorSector(Profile.isothermal(900), 1e4),
                                   TerminatorSector(Profile.isothermal(1300), 1e6))
        fi = self.fit_info(transit_terminator=term)
        fi.add_uniform_fit_param('sector1.T', 800, 1400)
        fi.add_uniform_fit_param('sector2.T', 800, 1400)
        fi.add_gaussian_fit_param('sector1.fraction', 0.2)
        rows = fi._from_unit_interval_batch(np.full((6, len(fi.fit_param_names)), 0.5))
        rows[:, fi.fit_param_names.index('sector1.fraction')] = [0, 1, 0, 1, -0.1, 1.1]
        rows[2:4, fi.fit_param_names.index('sector1.T')] = 1300
        rows[2:4, fi.fit_param_names.index('sector2.T')] = 900
        self.check(fi, eclipse=True, extra_params=rows)

    def test_mixed_heterogeneity_configs(self):
        fi = self.fit_info(T_het=4500, T_het2=6500, het_cov_frac=0.03,
                           het2_cov_frac=0.02)
        fi.add_gaussian_fit_param('het_cov_frac', 0.01)
        fi.add_gaussian_fit_param('het2_cov_frac', 0.01)
        rows = fi._from_unit_interval_batch(np.full((4, len(fi.fit_param_names)), 0.5))
        rows[:, fi.fit_param_names.index('het_cov_frac')] = [0, 0, .03, .03]
        rows[:, fi.fit_param_names.index('het2_cov_frac')] = [0, .02, 0, .02]
        self.check(fi, extra_params=rows)

    def test_guillot_eclipse(self):
        fi = self.free_guillot(profile_type='guillot')
        self.check(fi, transit=False, eclipse=True)

    def test_no_condensation(self):
        calc = TransitDepthCalculator(include_condensation=False,
                                       include_opacities=['H2O', 'CH4'], downsample=20)
        calc.change_wavelength_bins(self.bins, max_points_per_bin=3)
        self.check(self.fit_info(), calc=calc)

    def test_toi6894(self):
        # Importing the example changes cr.TransitDepthCalculator; restore it.
        import platon.combined_retriever as cr
        original = cr.TransitDepthCalculator
        from examples.toi6894_smc.retrieve_toi6894_smc import get_fit_info, GASES
        cr.TransitDepthCalculator = original
        calc = TransitDepthCalculator(include_opacities=GASES, downsample=20,
                                       stellar_grid='newera', logg_star=5.039, feh_star=0.142)
        calc.change_wavelength_bins(self.bins, max_points_per_bin=3)
        fi = get_fit_info()
        truth = json.loads((Path(__file__).parents[1] / 'examples/synthetic_recovery/truth.json').read_text())
        self.assertEqual(truth['names'], fi.fit_param_names)
        self.check(fi, calc=calc, extra_params=truth['values'])

    def test_invalid_rows(self):
        fi = self.fit_info()
        fi.add_gaussian_fit_param('Mp', M_jup)
        args = (self.transit, self.eclipse, fi, np.full(4, .01), np.full(4, 1e-4), None, None)
        params = fi._from_unit_interval_batch(np.full((5, len(fi.fit_param_names)), 0.5))
        params[0, fi.fit_param_names.index('Rp')] = 0.9 * R_jup  # outside prior
        params[1, fi.fit_param_names.index('Mp')] = -1  # nonpositive mass
        params[2, fi.fit_param_names.index('Mp')] = 1e20  # unbound
        fi.all_params['T'].low_lim, fi.all_params['T'].high_lim = 100, 6000
        params[3, fi.fit_param_names.index('T')] = 5000  # outside opacity grid
        mean, _ = CombinedRetriever()._ln_like_batch(params, *args, evaluator=BatchEvaluator(4))
        for i, row in enumerate(params):
            scalar = CombinedRetriever()._ln_like(row, *args, ret_model=True)
            self.assertEqual(np.all(np.isnan(mean[i])), not isinstance(scalar, tuple))
        self.assertTrue(np.all(np.isnan(mean[:4])))
        self.assertTrue(np.all(np.isfinite(mean[4])))

    def test_smc_default_and_scalar_debug(self):
        fi = self.fit_info()
        import platon.combined_retriever as cr

        def calculator(**kwargs):
            return TransitDepthCalculator(include_opacities=['H2O', 'CH4'],
                                           downsample=100, **kwargs)

        options = dict(n_particles=16, inner_steps=1, outer_steps=1,
                       num_final_samples=0, seed=7, cheap_downsample=2,
                       adaptive_steps=False)
        results = []
        with mock.patch.object(cr, 'TransitDepthCalculator', side_effect=calculator), \
             mock.patch.object(cr, 'write_param_estimates_file'):
            for batched in (True, False):
                result = CombinedRetriever().run_smc(
                    self.bins, np.full(4, .01), np.full(4, .01),
                    None, None, None, fi, batched=batched, **options)
                results.append(result)
        np.testing.assert_array_equal(results[0].samples, results[1].samples)
        np.testing.assert_allclose(results[0].logl, results[1].logl, atol=1e-3, rtol=0)
        self.assertAlmostEqual(results[0].final_logz, results[1].final_logz, delta=1e-3)
        self.assertGreater(results[0].n_cheap_evals, 0)

    def test_prior_transform(self):
        fi = self.fit_info()
        fi.add_gaussian_fit_param('Mp', 0.1 * M_jup)
        fi.all_params['T_het'].best_guess = 4000
        fi.all_params['T_het2'].best_guess = 6000
        fi.add_ordered_uniform_fit_params('T_het', 'T_het2', 3000, 7000)
        cubes = np.random.default_rng(5).uniform(0, 1, (50, len(fi.fit_param_names)))
        np.testing.assert_array_equal(fi._from_unit_interval_batch(cubes),
                                      [fi._from_unit_interval_array(u) for u in cubes])
        for invalid in (np.nan, -0.1, 1.1):
            cubes[0, 0] = invalid
            with self.assertRaises(AssertionError):
                fi._from_unit_interval_batch(cubes)
            with self.assertRaises(AssertionError):
                fi._from_unit_interval_array(cubes[0])

    def test_batch_size_validation(self):
        for size in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                BatchEvaluator(size)

    def test_overrides_and_float32(self):
        fi = self.fit_info()
        fi.all_params['T'].low_lim, fi.all_params['T'].high_lim = 1000, 1000.00001
        cubes = np.full((1, len(fi.fit_param_names)), 0.5, dtype=np.float32)
        np.testing.assert_array_equal(fi._from_unit_interval_batch(cubes),
                                      [fi._from_unit_interval_array(cubes[0])])
        fi = self.fit_info()
        params = fi._from_unit_interval_batch(np.full((1, len(fi.fit_param_names)), 0.9))
        args = (self.transit, None, fi, np.full(4, 0.01), np.full(4, 1e-4), None, None)
        # A custom atmosphere validation must still reject the row
        validate = self.transit.atm._validate_params
        def stricter(T, *rest):
            validate(T, *rest)
            if np.max(T) > 1100:
                raise AtmosphereError("custom bound")
        with mock.patch.object(self.transit.atm, '_validate_params', stricter):
            self.assertEqual(CombinedRetriever()._ln_like(params[0], *args, ret_model=True), -np.inf)
            self.assertTrue(np.all(np.isnan(CombinedRetriever()._ln_like_batch(params, *args)[0])))
        # A user-defined _ln_like is evaluated as written
        class Shifted(CombinedRetriever):
            def _ln_like(self, *a, **kw):
                mean, var = super()._ln_like(*a, **kw)
                return mean + 1e-3, var
        mean = Shifted()._ln_like_batch(params, *args)[0]
        np.testing.assert_array_equal(mean[0], CombinedRetriever()._ln_like(
            params[0], *args, ret_model=True)[0] + 1e-3)
