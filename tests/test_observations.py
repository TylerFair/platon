from pathlib import Path
import tempfile
import unittest

import numpy as np

from platon.observations import load_spectrum_csvs
from platon._offsets import apply_offsets
from platon.fit_info import FitInfo


class TestObservationCSV(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def csv(self, name, text):
        path = Path(self.directory.name) / name
        path.write_text(text)
        return path

    def test_explicit_bins_units_groups_and_overlapping_wavelengths(self):
        first = self.csv("a.csv", "wavelength_low,wavelength_high,depth,error\n1,1.2,10000,20\n1.2,1.4,10100,30\n")
        second = self.csv("b.csv", "wavelength_low,wavelength_high,depth,error\n1,1.3,11000,40\n")
        data = load_spectrum_csvs({"a": first, "b": second, "a2": first},
                                  depth_unit="ppm", offset_groups={"a2": "a"})
        np.testing.assert_allclose(data.wavelength_bins[:2], [[1e-6, 1.2e-6], [1.2e-6, 1.4e-6]])
        np.testing.assert_allclose(data.depths, [0.01, 0.0101, 0.011, 0.01, 0.0101])
        self.assertEqual(data.dataset_slices, {"a": (0, 2), "b": (2, 3), "a2": (3, 5)})
        self.assertEqual(data.offset_windows(), {"offset_b": [(2, 3)]})
        self.assertEqual(data.offset_windows(reference="b"), {"offset_a": [(0, 2), (3, 5)]})
        fit_info = FitInfo({"transit_offset_windows": data.offset_windows(reference="b")})
        fit_info.add_uniform_fit_param("offset_a", -1000, 1000)
        params = fit_info._interpret_param_array([100])
        np.testing.assert_allclose(apply_offsets(np.zeros(5), params, "transit"), [1e-4, 1e-4, 0, 1e-4, 1e-4])

    def test_alternative_columns_and_half_widths(self):
        path = self.csv("legacy.csv", "wavelength,wavelength_err,depth00,depth_err00\n1000,10,0.01,0.001\n")
        data = load_spectrum_csvs({"visit": path}, wavelength_unit="nm",
                                  columns={"depth": "depth00", "error": "depth_err00"})
        np.testing.assert_allclose(data.wavelength_bins, [[990e-9, 1010e-9]])
        np.testing.assert_allclose(data.errors, [0.001])
        self.assertEqual(data.offset_windows(), {})

    def test_custom_columns_and_full_widths(self):
        path = self.csv("custom.csv", "wave,width,d,sigma\n2,0.4,1.2,0.01\n")
        data = load_spectrum_csvs({"visit": path}, depth_unit="percent",
                                  columns={"wavelength": "wave", "bin_width": "width",
                                           "depth": "d", "error": "sigma"})
        np.testing.assert_allclose(data.wavelength_bins, [[1.8e-6, 2.2e-6]])
        np.testing.assert_allclose(data.depths, [0.012])
        np.testing.assert_allclose(data.errors, [0.0001])

    def test_center_bin_inference_is_per_file(self):
        a = self.csv("a.csv", "wavelength,depth,error\n1,0.01,0.001\n1.4,0.01,0.001\n2,0.01,0.001\n")
        b = self.csv("b.csv", "wavelength,depth,error\n1,0.01,0.001\n1.2,0.01,0.001\n")
        data = load_spectrum_csvs({"a": a, "b": b})
        np.testing.assert_allclose(data.wavelength_bins * 1e6, [[0.8, 1.2], [1.2, 1.7], [1.7, 2.3], [0.9, 1.1], [1.1, 1.3]])

    def test_automatic_offsets_and_shared_reference(self):
        path = self.csv("visit.csv", "wavelength,bin_width,depth,error\n1,0.1,0.01,0.001\n")
        data = load_spectrum_csvs(
            {"niriss_o1": path, "nirspec": path, "niriss_o2": path},
            offset_groups={"niriss_o1": "niriss", "niriss_o2": "niriss"})
        info = FitInfo({})
        self.assertIs(data.add_offsets(info, reference="nirspec", prior_half_width=200), info)
        self.assertEqual(info.fit_param_names, ["offset_niriss"])
        params = info._interpret_param_array([100])
        np.testing.assert_allclose(apply_offsets(np.zeros(3), params, "transit"), [100e-6, 0, 100e-6])
        np.testing.assert_allclose(info._from_unit_interval_array([0.75]), [100])
        self.assertEqual(info._ln_prior([300]), -np.inf)

    def test_automatic_eclipse_offsets_and_single_instrument(self):
        path = self.csv("visit.csv", "wavelength,bin_width,depth,error\n1,0.1,0.01,0.001\n")
        data = load_spectrum_csvs({"reference": path, "miri": path})
        info = data.add_offsets(FitInfo({}), spectrum="eclipse")
        np.testing.assert_allclose([info._from_unit_interval_array([x])[0]
                                   for x in (0, 0.5, 1)],
                                   [-500, 0, 500])
        params = info._interpret_param_array([-200])
        np.testing.assert_allclose(apply_offsets(np.zeros(2), params, "eclipse"), [0, -200e-6])
        np.testing.assert_array_equal(apply_offsets(np.zeros(2), params, "transit"), [0, 0])
        single = load_spectrum_csvs({"miri": path})
        self.assertEqual(single.add_offsets(FitInfo({})).fit_param_names, [])

    def test_automatic_offsets_reject_bad_priors_and_collisions(self):
        path = self.csv("visit.csv", "wavelength,bin_width,depth,error\n1,0.1,0.01,0.001\n")
        data = load_spectrum_csvs({"reference": path, "a": path, "b": path})
        for value in [0, -1e-4, np.nan, True, "ppm"]:
            info = FitInfo({})
            with self.subTest(value=value), self.assertRaises(ValueError):
                data.add_offsets(info, prior_half_width=value)
            self.assertEqual(info.all_params, {})
        info = FitInfo({"offset_b": 100})
        with self.assertRaisesRegex(ValueError, "already exist"):
            data.add_offsets(info)
        self.assertEqual(set(info.all_params), {"offset_b"})
        with self.assertRaisesRegex(ValueError, "spectrum"):
            data.add_offsets(FitInfo({}), spectrum="other")

    def test_native_files_with_different_headers_and_units(self):
        standard = self.csv("standard.csv", "wavelength,bin_width,depth,error\n1000,100,10000,20\n")
        archive = self.csv("archive.csv", "CENTRALWAVELNG,BANDWIDTH,PL_TRANDEP,PL_TRANDEPERR1\n1,0.2,1.2,0.003\n")
        data = load_spectrum_csvs(
            {"nirspec": standard, "hst": archive},
            wavelength_unit="nm", depth_unit="ppm",
            dataset_options={"hst": {"wavelength_unit": "um", "depth_unit": "percent",
                "columns": {"wavelength": "CENTRALWAVELNG", "bin_width": "BANDWIDTH",
                            "depth": "PL_TRANDEP", "error": "PL_TRANDEPERR1"}}})
        np.testing.assert_allclose(data.wavelength_bins, [[0.95e-6, 1.05e-6], [0.9e-6, 1.1e-6]])
        np.testing.assert_allclose(data.depths, [0.01, 0.012])
        np.testing.assert_allclose(data.errors, [20e-6, 30e-6])
        self.assertEqual(data.offset_windows(), {"offset_hst": [(1, 2)]})

    def test_dataset_column_overrides_merge_with_common_mapping(self):
        a = self.csv("a.csv", "wavelength,bin_width,d,sigma\n1,0.1,0.01,0.001\n")
        b = self.csv("b.csv", "wavelength,bin_width,d,err\n1,0.1,0.02,0.002\n")
        data = load_spectrum_csvs(
            {"a": a, "b": b}, columns={"depth": "d", "error": "sigma"},
            dataset_options={"b": {"columns": {"error": "err"}}})
        np.testing.assert_allclose(data.errors, [0.001, 0.002])

    def test_invalid_per_dataset_options(self):
        path = self.csv("visit.csv", "wavelength,bin_width,depth,error\n1,0.1,0.01,0.001\n")
        for options in [["visit"], {"other": {}}, {"visit": None},
                        {"visit": {"unknown": 1}}, {"visit": {"depth_unit": "guess"}},
                        {"visit": {"wavelength_unit": "angstrom"}},
                        {"visit": {"columns": None}}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                load_spectrum_csvs({"visit": path}, dataset_options=options)

    def test_explicit_bins_preserve_unsorted_rows(self):
        path = self.csv("visit.csv", "wavelength_low,wavelength_high,depth,error\n2,3,0.02,0.002\n1,2,0.01,0.001\n")
        data = load_spectrum_csvs({"visit": path})
        np.testing.assert_allclose(data.wavelength_bins, [[2e-6, 3e-6], [1e-6, 2e-6]])
        np.testing.assert_allclose(data.depths, [0.02, 0.01])

    def test_automatic_csv_offsets_reach_joint_likelihood(self):
        from platon.combined_retriever import CombinedRetriever

        class Calculator:
            def __init__(self, baseline, size):
                self.depths = np.full(size, baseline)

            def compute_depths(self, *args, **kwargs):
                return np.arange(len(self.depths)), self.depths, {}

        transit_ref = self.csv("transit_ref.csv", "wavelength,bin_width,depth,error\n1,0.1,10000,10\n1.2,0.1,10000,10\n")
        transit_visit = self.csv("transit_visit.csv", "wavelength,bin_width,depth,error\n1,0.1,10100,10\n")
        eclipse_ref = self.csv("eclipse_ref.csv", "wavelength,bin_width,depth,error\n5,0.1,1000,20\n")
        eclipse_visit = self.csv("eclipse_visit.csv", "wavelength,bin_width,depth,error\n5,0.1,800,20\n")
        transit_data = load_spectrum_csvs({"niriss": transit_ref, "nirspec": transit_visit}, depth_unit="ppm")
        eclipse_data = load_spectrum_csvs({"reference": eclipse_ref, "miri": eclipse_visit}, depth_unit="ppm")
        fit_info = CombinedRetriever.get_default_fit_info(7e8, 1.9e27, 7e7, T=1000, error_multiple=2)
        transit_data.add_offsets(fit_info)
        eclipse_data.add_offsets(fit_info, spectrum="eclipse")
        self.assertEqual(fit_info.fit_param_names, ["offset_nirspec", "offset_miri"])
        transit, eclipse = Calculator(0.01, 3), Calculator(0.001, 2)
        retriever = CombinedRetriever()
        retriever.params_to_lnlike = {}
        args = (transit, eclipse, fit_info, transit_data.depths, transit_data.errors,
                eclipse_data.depths, eclipse_data.errors)
        pointwise = retriever._ln_like([100, -200], *args, lnlike_per_point=True)
        scaled_errors = 2 * np.concatenate([transit_data.errors, eclipse_data.errors])
        np.testing.assert_allclose(pointwise, -0.5 * np.log(2 * np.pi * scaled_errors**2))
        zero_ln_like = retriever._ln_like([0, 0], *args)
        self.assertAlmostEqual(pointwise.sum() - zero_ln_like, 25)
        best_transit, _, best_eclipse, _ = retriever._ln_like(
            [100, -200], *args, ret_best_fit=True)
        np.testing.assert_allclose(best_transit, transit_data.depths)
        np.testing.assert_allclose(best_eclipse, eclipse_data.depths)
        np.testing.assert_array_equal(transit.depths, [0.01] * 3)
        np.testing.assert_array_equal(eclipse.depths, [0.001] * 2)

    def test_numbered_headers_require_explicit_mapping(self):
        path = self.csv("numbered.csv", "wavelength,bin_width,depth00,depth_err00\n1,0.1,0.01,0.001\n")
        with self.assertRaisesRegex(ValueError, "columns mapping"):
            load_spectrum_csvs({"visit": path})

    def test_unit_sanity_checks_name_likely_units(self):
        for depth, unit, likely in [(10000, "fraction", "ppm"),
                                     (2, "fraction", "percent"),
                                     (.01, "ppm", "fraction")]:
            path = self.csv("depth.csv", f"wavelength,bin_width,depth,error\n1,0.1,{depth},0.001\n")
            with self.subTest(unit=unit, depth=depth), self.assertRaisesRegex(ValueError, f"depth_unit='{likely}'"):
                load_spectrum_csvs({"visit": path}, depth_unit=unit)
        for wave, unit, likely in [(2000, "um", "nm"), (2, "m", "um"),
                                    (2e-6, "um", "m"), (2e-6, "nm", "m")]:
            path = self.csv("wave.csv", f"wavelength,bin_width,depth,error\n{wave},{wave / 10},0.01,0.001\n")
            with self.subTest(unit=unit), self.assertRaisesRegex(ValueError, f"wavelength_unit='{likely}'"):
                load_spectrum_csvs({"visit": path}, wavelength_unit=unit)
        path = self.csv("override.csv", "wavelength,bin_width,depth,error\n1,0.1,10000,20\n")
        with self.assertRaisesRegex(ValueError, "depth_unit='ppm'"):
            load_spectrum_csvs({"visit": path}, depth_unit="ppm",
                               dataset_options={"visit": {"depth_unit": "fraction"}})

    def test_dataset_names_are_independent_of_parameter_syntax(self):
        path = self.csv("visit.csv", "wavelength,bin_width,depth,error\n1,0.1,0.01,0.001\n")
        data = load_spectrum_csvs({"start": path, "visit_end": path})
        info = data.add_offsets(FitInfo({}))
        assert info.fit_param_names == ['offset_visit_end']

    def test_reject_invalid_observation_rows(self):
        for rows in ["1,1.2,0.01,0", "1,1.2,nan,0.01", "1,1,0.01,0.01",
                     "-1,1,0.01,0.01", "1,inf,0.01,0.01", "1,1.2,x,0.01"]:
            path = self.csv("invalid.csv", "wavelength_low,wavelength_high,depth,error\n" + rows + "\n")
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_spectrum_csvs({"visit": path})

    def test_reject_ambiguous_formats_and_configuration(self):
        single = self.csv("single.csv", "wavelength,depth,error\n1,0.01,0.001\n")
        with self.assertRaisesRegex(ValueError, "at least two"):
            load_spectrum_csvs({"visit": single})
        valid = self.csv("valid.csv", "wavelength,bin_width,depth,error\n1,0.1,0.01,0.001\n")
        for kwargs in [{"depth_unit": "guess"}, {"wavelength_unit": "angstrom"},
                       {"offset_groups": {"unknown": "a"}},
                       {"columns": {"depth": "missing"}}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                load_spectrum_csvs({"visit": valid}, **kwargs)
        data = load_spectrum_csvs({"visit": valid})
        with self.assertRaisesRegex(ValueError, "reference"):
            data.offset_windows(reference="unknown")


if __name__ == "__main__":
    unittest.main()
