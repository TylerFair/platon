import unittest
from types import SimpleNamespace

import numpy as np

from platon._batched_likelihood import BatchedLikelihood
from platon.combined_retriever import CombinedRetriever
from platon.constants import M_jup, R_jup, R_sun
from platon import _forward_model as fm
from platon.TP_profile import Profile, _default_pressures
from platon.terminator import TerminatorSector, TwoSectorTerminator
from platon.transit_depth_calculator import TransitDepthCalculator


class TestBatchedLikelihood(unittest.TestCase):
    def test_vectorized_profile_pack_matches_scalar(self):
        shared = dict(T_star=6000, Rs=R_sun, a=8e9, Mp=M_jup, Rp=R_jup,
                      log_k_th=-2, T_int=100)
        sectors = [TerminatorSector(Profile.guillot(
            **shared, beta=beta, log_gamma=-1), cloud_fraction=.5)
            for beta in (.7, .9)]
        fit = CombinedRetriever.get_default_fit_info(
            Rs=R_sun, Mp=M_jup, Rp=R_jup,
            transit_terminator=TwoSectorTerminator(*sectors))
        for sector in ("sector1.", "sector2."):
            for name, bounds in (("beta", (.2, 1.5)), ("log_gamma", (-2, .5)),
                                 ("log_cloudtop_P", (-1, 6)),
                                 ("log_scatt_factor", (-4, 10)),
                                 ("scatt_slope", (-2, 20)),
                                 ("cloud_fraction", (0, 1))):
                fit.add_uniform_fit_param(sector + name, *bounds)
        for name, bounds in (("Rp", (.8 * R_jup, 1.5 * R_jup)),
                             ("log_k_th", (-3, -.5)), ("T_int", (30, 630)),
                             ("log_P_quench", (-3, 7))):
            fit.add_uniform_fit_param(name, *bounds)
        rng = np.random.default_rng(194)
        rows = np.array([fit._from_unit_interval_array(u) for u in
                         rng.random((64, len(fit.fit_param_names)))])
        evaluator = object.__new__(BatchedLikelihood)
        evaluator.fit = fit
        evaluator.defaults = {n: p.best_guess for n, p in fit.all_params.items()}
        evaluator.pressures = _default_pressures()
        evaluator.calc = SimpleNamespace(atm=SimpleNamespace(
            min_temperature=100, max_temperature=4000))
        evaluator.template = np.zeros(fm.SC_N_SCALARS + fm.IX_N_INTS +
                                      3 * len(evaluator.pressures) - 1 + 10, np.float32)
        evaluator.free_vmr, evaluator.gases, evaluator.dual = False, None, True
        evaluator.profile_type = "guillot"
        evaluator.cfg = SimpleNamespace(stellar_in_grid=True, has_het=False, sort_layers=True)
        evaluator.stellar_temps = np.array([2300., 12000.])
        evaluator.cloudtop_range = (1e-4, 1e8)
        packed, _, valid, fallback, _, _, _, _ = evaluator.prepare(rows)
        offset = fm.SC_N_SCALARS + fm.IX_N_INTS
        n_clamped = 0
        for i, row in enumerate(rows):
            params = fit._interpret_param_array(row)
            sectors = evaluator.defaults["transit_terminator"].sectors_from_params(params)
            # The scalar path's deep-temperature clamp, then its grid check
            temperatures = np.array([TransitDepthCalculator._clamp_unprobed(
                evaluator.calc, sector.profile.temperatures, sector.profile.pressures,
                10**params["log_P_quench"]) for sector in sectors])
            ok = temperatures.min() >= 100 and temperatures.max() <= 4000
            self.assertEqual(valid[i], ok)
            if ok:
                np.testing.assert_array_equal(packed[i, :, offset:offset + 100],
                                              temperatures.astype(np.float32))
                n_clamped += max(s.profile.temperatures.max() for s in sectors) > 4000
        self.assertGreater(n_clamped, 0)
        self.assertFalse(fallback.any())

    def test_float64_epilogue_and_invalid_rows(self):
        rng = np.random.default_rng(228)
        nbin, nrow = 40, 24
        full = CombinedRetriever.get_default_fit_info(
            Rs=R_sun, Mp=M_jup, Rp=R_jup, T=1200,
            transit_offsets={"offset": (10, nbin)},
            time_shift_offsets={"t0": (0, nbin)},
            time_shift_errors=rng.uniform(5, 15, nbin),
            time_shift_coefficients=np.full(nbin, 10000.))
        # Sampled (Gaussian prior), so the vectorized epilogue applies when
        # the offsets are integrated
        full.add_gaussian_fit_param("error_excess", 5e-5)
        full.add_uniform_fit_param("offset", -1e-3, 1e-3)
        full.add_gaussian_fit_param("t0", 20)
        full.all_params["time_shifts"].best_guess = rng.normal(0, 10, nbin)

        for marginalize in (False, True):
            fit = CombinedRetriever._marginalize(full) if marginalize else full
            rows = np.array([fit._from_unit_interval_array(u) for u in
                             rng.uniform(.01, .99, (nrow, len(fit.fit_param_names)))])
            values = {n: p.best_guess for n, p in fit.all_params.items()}
            values.update(zip(fit.fit_param_names, rows.T))
            for name in getattr(fit, "marginal_params", ()):
                values[name] = 0.
            values["_batch_swap"] = rng.random(nrow) > .5
            fractions = rng.uniform(.1, .9, nrow)
            clouds = rng.uniform(.1, .9, (nrow, 2))
            indices = np.arange(nrow)[1:]
            raw = np.zeros((len(indices), 2, 2 * nbin + 2), np.float32)
            raw[:, :, :-2] = rng.uniform(.01, .04, raw[:, :, :-2].shape)
            raw[::7, 0, -2] = 1  # Device rejects an unbound atmosphere.

            evaluator = object.__new__(BatchedLikelihood)
            evaluator.retriever = CombinedRetriever()
            evaluator.fit = fit
            evaluator.depths = rng.uniform(.01, .02, nbin)
            evaluator.errors = rng.uniform(80e-6, 150e-6, nbin)
            evaluator.vectorized_finish = True
            evaluator.dual = True
            expected = evaluator._finish_scalar(
                rows, raw, indices, values, clouds, fractions)
            depths = evaluator.last_depths.copy()
            actual = evaluator.finish(rows, raw, indices, values, clouds, fractions)
            np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
            np.testing.assert_array_equal(evaluator.last_depths, depths)
            finite = np.isfinite(expected)
            np.testing.assert_allclose(actual[finite], expected[finite], rtol=0, atol=1e-8)
            self.assertTrue(np.isneginf(actual[0]))

    def test_matches_scalar_likelihood(self):
        """Batched rows equal _ln_like exactly, with the same rejections."""
        low, high, depths, errors = np.loadtxt(
            "tests/testing_data/hd209458b_transit_depths", unpack=True)
        bins = np.array([low, high]).T
        n = len(depths)
        calc = TransitDepthCalculator()
        calc.change_wavelength_bins(bins)
        retriever = CombinedRetriever()
        rng = np.random.default_rng(941)
        Rs, a = 1.19 * R_sun, 7e9
        shared = dict(T_star=6000, Rs=Rs, a=a, Mp=0.73 * M_jup, Rp=1.4 * R_jup,
                      log_k_th=-2, T_int=100)
        for patchy in (True, False):
            sectors = [TerminatorSector(Profile.guillot(**shared, beta=beta, log_gamma=-1),
                                        cloudtop_pressure=1e5, cloud_fraction=.5 if patchy else 1)
                       for beta in (.7, .9)]
            full = CombinedRetriever.get_default_fit_info(
                Rs=Rs, Mp=0.73 * M_jup, Rp=1.4 * R_jup, T_star=6000, logZ=1, CO_ratio=.5,
                transit_terminator=TwoSectorTerminator(*sectors), a=a, b=.3, period=3.5 * 86400,
                time_shifts=rng.normal(0, 10, n), time_shift_errors=np.full(n, 10.),
                time_shift_offsets={"t0": (0, n)}, transit_offsets={"offset": (n // 2, n)})
            priors = {"beta": (.6, 1.), "log_gamma": (-1.5, 0), "log_cloudtop_P": (2, 6),
                      "log_scatt_factor": (0, 2), "scatt_slope": (2, 6)}
            if patchy:
                priors["cloud_fraction"] = (0, 1)
            for sector in ("sector1.", "sector2."):
                for name, bounds in priors.items():
                    full.add_uniform_fit_param(sector + name, *bounds)
            for name, bounds in (("Rp", (1.3 * R_jup, 1.5 * R_jup)), ("log_k_th", (-2.5, -1)),
                                 ("T_int", (50, 300)), ("logZ", (-1, 2)), ("CO_ratio", (.2, 1)),
                                 ("log_P_quench", (-1, 7)), ("offset", (-1e-3, 1e-3)),
                                 ("t0", (-50, 50))):
                full.add_uniform_fit_param(name, *bounds)
            # An integrated error_excess takes the scalar epilogue; offsets with a
            # sampled one (Gaussian prior) the vectorized one
            if patchy:
                full.add_uniform_fit_param("error_excess", 0, 1e-4)
            else:
                full.add_gaussian_fit_param("error_excess", 5e-5)
            fit = CombinedRetriever._marginalize(full)
            batch = retriever._batched_ln_like(calc, None, fit, depths, errors, None, None,
                                               batch_size=16)
            self.assertIsNotNone(batch)
            self.assertEqual(batch.dual, patchy)
            self.assertEqual(batch.vectorized_finish, not patchy)
            rows = np.array([fit._from_unit_interval_array(u) for u in
                             rng.uniform(.02, .98, (24, len(fit.fit_param_names)))])
            expected = np.array([retriever._ln_like(row, calc, None, fit, depths, errors,
                                                    None, None) for row in rows])
            actual = batch(rows)
            self.assertGreater(np.isfinite(expected).sum(), 8)
            np.testing.assert_array_equal(actual, expected)

        # Mie scattering stays on the scalar likelihood
        mie = CombinedRetriever.get_default_fit_info(Rs=Rs, Mp=0.73 * M_jup, Rp=1.4 * R_jup,
                                                     T=1200, n=1.7, log_k=-1,
                                                     log_number_density=8, log_scatt_factor=0)
        mie.add_uniform_fit_param("T", 800, 1800)
        self.assertIsNone(retriever._batched_ln_like(calc, None, mie, depths, errors,
                                                     None, None))

    def test_matches_scalar_isothermal_and_parametric(self):
        """Two-sector isothermal and Madhusudhan-Seager sectors: batched rows
        equal _ln_like exactly, with the same rejections."""
        low, high, depths, errors = np.loadtxt(
            "tests/testing_data/hd209458b_transit_depths", unpack=True)
        bins = np.array([low, high]).T
        n = len(depths)
        calc = TransitDepthCalculator()
        calc.change_wavelength_bins(bins)
        retriever = CombinedRetriever()
        rng = np.random.default_rng(1941)
        Rs, a = 1.19 * R_sun, 7e9
        profiles = {
            "isothermal": ([Profile.isothermal(T) for T in (1000, 1300)],
                           {"T": (700, 1800)}),
            "parametric": ([Profile.parametric(T0, 1e2, .5, .5, 1e6, 1700) for T0 in (900, 1100)],
                           {"T0": (800, 1300), "log_P1": (0, 3), "alpha1": (.3, .8),
                            "alpha2": (.3, .8), "log_P3": (5, 7), "T3": (1200, 2200)})}
        for kind, (sector_profiles, profile_priors) in profiles.items():
            for patchy in (True, False):
                sectors = [TerminatorSector(profile, cloudtop_pressure=1e5,
                                            cloud_fraction=.5 if patchy else 1)
                           for profile in sector_profiles]
                full = CombinedRetriever.get_default_fit_info(
                    Rs=Rs, Mp=0.73 * M_jup, Rp=1.4 * R_jup, logZ=1, CO_ratio=.5,
                    transit_terminator=TwoSectorTerminator(*sectors), a=a, b=.3,
                    period=3.5 * 86400, time_shifts=rng.normal(0, 10, n),
                    time_shift_errors=np.full(n, 10.), time_shift_offsets={"t0": (0, n)},
                    transit_offsets={"offset": (n // 2, n)})
                priors = dict(profile_priors, log_cloudtop_P=(2, 6), log_scatt_factor=(0, 2),
                              scatt_slope=(2, 6))
                if patchy:
                    priors["cloud_fraction"] = (0, 1)
                for sector in ("sector1.", "sector2."):
                    for name, bounds in priors.items():
                        full.add_uniform_fit_param(sector + name, *bounds)
                for name, bounds in (("Rp", (1.3 * R_jup, 1.5 * R_jup)), ("logZ", (-1, 2)),
                                     ("CO_ratio", (.2, 1)), ("offset", (-1e-3, 1e-3)),
                                     ("t0", (-50, 50)), ("error_excess", (0, 1e-4))):
                    full.add_uniform_fit_param(name, *bounds)
                fit = CombinedRetriever._marginalize(full)
                batch = retriever._batched_ln_like(calc, None, fit, depths, errors, None, None,
                                                   batch_size=16)
                self.assertIsNotNone(batch, kind)
                self.assertEqual(batch.dual, patchy)
                rows = np.array([fit._from_unit_interval_array(u) for u in
                                 rng.uniform(.02, .98, (24, len(fit.fit_param_names)))])
                expected = np.array([retriever._ln_like(row, calc, None, fit, depths, errors,
                                                        None, None) for row in rows])
                actual = batch(rows)
                self.assertGreater(np.isfinite(expected).sum(), 8, kind)
                np.testing.assert_array_equal(actual, expected, err_msg=kind)
                # and the batch, not its scalar fallback, evaluated them
                self.assertGreater(np.sum(batch.last_reasons == ""), 8, kind)

    def test_matches_scalar_one_dimensional(self):
        """1-D (single-terminator) isothermal, Guillot and parametric profiles,
        with and without a patchy cloud_fraction, equilibrium chemistry or free
        VMRs, an offset and an integrated error_excess: batched rows equal
        _ln_like exactly, with the same rejections."""
        low, high, depths, errors = np.loadtxt(
            "tests/testing_data/hd209458b_transit_depths", unpack=True)
        bins = np.array([low, high]).T
        n = len(depths)
        calc = TransitDepthCalculator()
        calc.change_wavelength_bins(bins)
        retriever = CombinedRetriever()
        rng = np.random.default_rng(1961)
        Rs = 1.19 * R_sun
        profiles = {
            "isothermal": (dict(T=1200), {"T": (800, 1800)}),
            "guillot": (dict(transit_profile_type="guillot", T_star=6000, a=7e9, beta=.8,
                             log_k_th=-2, log_gamma=-1, T_int=100),
                        {"beta": (.6, 1.), "log_gamma": (-1.5, 0), "log_k_th": (-2.5, -1),
                         "T_int": (50, 300)}),
            "parametric": (dict(transit_profile_type="parametric", T0=1000, log_P1=2,
                                alpha1=.5, alpha2=.5, log_P3=6, T3=1700),
                           {"T0": (800, 1300), "log_P1": (0, 3), "alpha1": (.3, .8),
                            "alpha2": (.3, .8), "log_P3": (5, 7), "T3": (1200, 2200)})}
        for kind, (profile_kwargs, profile_priors) in profiles.items():
            for patchy in (True, False):
                for free in (False, True):
                    chemistry = dict(logZ=None, CO_ratio=None, fit_vmr=True) if free else \
                        dict(logZ=1, CO_ratio=.5)
                    full = CombinedRetriever.get_default_fit_info(
                        Rs=Rs, Mp=0.73 * M_jup, Rp=1.4 * R_jup, log_cloudtop_P=5,
                        cloud_fraction=.5 if patchy else 1,
                        transit_offsets={"offset": (n // 2, n)}, **chemistry, **profile_kwargs)
                    priors = dict(profile_priors, log_cloudtop_P=(2, 6), log_scatt_factor=(0, 2),
                                  scatt_slope=(2, 6), Rp=(1.3 * R_jup, 1.5 * R_jup),
                                  offset=(-1e-3, 1e-3), error_excess=(0, 1e-4))
                    if patchy:
                        priors["cloud_fraction"] = (0, 1)
                    if free:
                        full.add_gases_vmr(["H2O", "CO2", "CO", "H2-He"], 1e-8, 1e-2)
                    else:
                        priors.update(logZ=(-1, 2), CO_ratio=(.2, 1))
                    for name, bounds in priors.items():
                        full.add_uniform_fit_param(name, *bounds)
                    fit = CombinedRetriever._marginalize(full)
                    label = f"{kind}, patchy={patchy}, free={free}"
                    batch = retriever._batched_ln_like(calc, None, fit, depths, errors, None,
                                                       None, batch_size=16)
                    self.assertIsNotNone(batch, label)
                    self.assertTrue(batch.one_d, label)
                    self.assertEqual(batch.dual, patchy, label)
                    rows = np.array([fit._from_unit_interval_array(u) for u in
                                     rng.uniform(.02, .98, (24, len(fit.fit_param_names)))])
                    expected = np.array([retriever._ln_like(row, calc, None, fit, depths,
                                                            errors, None, None) for row in rows])
                    actual = batch(rows)
                    self.assertGreater(np.isfinite(expected).sum(), 8, label)
                    np.testing.assert_array_equal(actual, expected, err_msg=label)
                    self.assertGreater(np.sum(batch.last_reasons == ""), 8, label)
                    self.assertEqual(batch.last_depths.shape, (24, 1, n), label)

    def test_matches_scalar_vmrs_on_equilibrium(self):
        """Free constant VMRs on top of equilibrium chemistry (shared or per
        sector), and per-sector free VMRs with a filler: batched rows equal
        _ln_like exactly, with the same rejections."""
        low, high, depths, errors = np.loadtxt(
            "tests/testing_data/hd209458b_transit_depths", unpack=True)
        bins = np.array([low, high]).T
        n = len(depths)
        calc = TransitDepthCalculator()
        calc.change_wavelength_bins(bins)
        retriever = CombinedRetriever()
        rng = np.random.default_rng(1939)
        Rs, a, Mp, Rp = 1.19 * R_sun, 7e9, 0.73 * M_jup, 1.4 * R_jup
        shared = dict(T_star=6000, Rs=Rs, a=a, Mp=Mp, Rp=Rp, log_k_th=-2, T_int=100)
        guillot = TwoSectorTerminator(*[TerminatorSector(
            Profile.guillot(**shared, beta=beta, log_gamma=-1), cloudtop_pressure=1e5)
            for beta in (.7, .9)])
        isothermal = TwoSectorTerminator(*[TerminatorSector(
            Profile.isothermal(T), cloudtop_pressure=1e5) for T in (1000, 1300)])
        cases = {
            # WASP-39b-like: Guillot limbs with time shifts, shared logZ and C/O,
            # one shared free SO2 on top, integrated offsets and error_excess
            "guillot, shared SO2 on equilibrium": dict(
                fit=dict(T_star=6000, logZ=1, CO_ratio=.5, fit_vmr=True,
                         transit_terminator=guillot, a=a, b=.3, period=3.5 * 86400,
                         time_shifts=rng.normal(0, 10, n), time_shift_errors=np.full(n, 10.),
                         time_shift_offsets={"t0": (0, n)},
                         transit_offsets={"off_a": (n // 3, 2 * n // 3), "off_b": (2 * n // 3, n)}),
                gases=(["SO2"], False),
                sector_priors={"beta": (.6, 1.), "log_gamma": (-1.5, 0)},
                priors={"log_k_th": (-2.5, -1), "T_int": (50, 300), "logZ": (-1, 2),
                        "CO_ratio": (.2, 1), "off_a": (-1e-3, 1e-3), "off_b": (-1e-3, 1e-3),
                        "t0": (-50, 50)}),
            "isothermal, per-sector gases on equilibrium": dict(
                fit=dict(logZ=1, CO_ratio=.5, fit_vmr=True, transit_terminator=isothermal),
                gases=(["SO2", "H2O"], True), sector_priors={"T": (700, 1800)},
                priors={"logZ": (-1, 2), "CO_ratio": (.2, 1)}),
            "isothermal, per-sector free VMRs": dict(
                fit=dict(logZ=None, CO_ratio=None, fit_vmr=True, transit_terminator=isothermal),
                gases=(["H2O", "CO2", "CO", "H2-He"], True), sector_priors={"T": (700, 1800)},
                priors={}),
            "1-D isothermal, SO2 on equilibrium": dict(
                fit=dict(T=1200, logZ=1, CO_ratio=.5, fit_vmr=True, log_cloudtop_P=5),
                gases=(["SO2"], False), sector_priors={},
                priors={"T": (800, 1800), "logZ": (-1, 2), "CO_ratio": (.2, 1),
                        "log_cloudtop_P": (2, 6), "log_scatt_factor": (0, 2),
                        "scatt_slope": (2, 6)})}
        for label, case in cases.items():
            full = CombinedRetriever.get_default_fit_info(Rs=Rs, Mp=Mp, Rp=Rp, **case["fit"])
            gases, per_sector = case["gases"]
            full.add_gases_vmr(gases, 1e-10, 1e-2, per_sector=per_sector)
            if "transit_terminator" in case["fit"]:
                for sector in ("sector1.", "sector2."):
                    for name, bounds in dict(case["sector_priors"], log_cloudtop_P=(2, 6),
                                             log_scatt_factor=(0, 2), scatt_slope=(2, 6)).items():
                        full.add_uniform_fit_param(sector + name, *bounds)
            for name, bounds in dict(case["priors"], Rp=(1.3 * R_jup, 1.5 * R_jup),
                                     error_excess=(0, 1e-4)).items():
                full.add_uniform_fit_param(name, *bounds)
            fit = CombinedRetriever._marginalize(full)
            batch = retriever._batched_ln_like(calc, None, fit, depths, errors, None, None,
                                               batch_size=16)
            self.assertIsNotNone(batch, label)
            self.assertEqual(batch.per_sector_gases, per_sector, label)
            rows = np.array([fit._from_unit_interval_array(u) for u in
                             rng.uniform(.02, .98, (24, len(fit.fit_param_names)))])
            expected = np.array([retriever._ln_like(row, calc, None, fit, depths, errors,
                                                    None, None) for row in rows])
            actual = batch(rows)
            self.assertGreater(np.isfinite(expected).sum(), 8, label)
            np.testing.assert_array_equal(actual, expected, err_msg=label)
            self.assertGreater(np.sum(batch.last_reasons == ""), 8, label)

    def test_all_rejected_and_empty_batch(self):
        evaluator = object.__new__(BatchedLikelihood)
        evaluator.depths = np.zeros(3)
        evaluator.vectorized_finish = True
        result = evaluator.finish(np.empty((2, 1)), np.empty((0, 2, 8)),
                                  np.array([], dtype=int), {}, None, None)
        np.testing.assert_array_equal(result, [-np.inf, -np.inf])
        self.assertEqual(evaluator.last_depths.shape, (2, 2, 3))
        self.assertEqual(evaluator(np.empty((0, 1))).shape, (0,))
        self.assertEqual(evaluator.last_depths.shape, (0, 2, 3))


if __name__ == "__main__":
    unittest.main()
