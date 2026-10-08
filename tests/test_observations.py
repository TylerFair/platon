"""load_spectra: formats, units, widths, offsets, visits, and mistakes."""
from unittest import mock
import warnings

import numpy as np
import pytest

from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_jup, M_jup
from platon.observations import load_spectra


@pytest.fixture
def write(tmp_path):
    def write(name, text):
        path = tmp_path / name
        path.write_text(text)
        return path
    return write


EDGES = "wavelength_low,wavelength_high,depth,error\n1,1.2,0.01,1e-4\n1.2,1.4,0.0101,1e-4\n1.4,1.6,0.0102,2e-4\n"


def test_exotedrf_csv_half_widths_ppm_and_orders(write):
    # exoTEDRF: '#' metadata, wave_err is a half width, dppm in ppm, order 2 first
    path = write("w39_soss.csv", "# Column wave_err: Wavelength bin halfwidth (micron)\n#\n"
                 "wave,wave_err,dppm,dppm_err,order\n"
                 "0.85,0.01,21000,90,2\n0.87,0.01,21050,95,2\n0.89,0.01,21060,95,2\n"
                 "0.90,0.015,21100,80,1\n0.93,0.015,21020,70,1\n0.96,0.015,21080,75,1\n")
    data = load_spectra([path])
    assert list(data.ranges) == ["w39_soss"]
    np.testing.assert_allclose(data.bins[0], [0.84e-6, 0.86e-6])
    np.testing.assert_allclose(data.bins[3], [0.885e-6, 0.915e-6])
    np.testing.assert_allclose(data.depths[:2], [0.021, 0.02105])
    np.testing.assert_allclose(data.errors[:2], [90e-6, 95e-6])
    assert data.offsets == {}


def test_eureka_ecsv_half_width_and_asymmetric_errors(write):
    path = write("S6_table.txt", "# %ECSV 1.0\n# ---\n# datatype:\n"
                 "# - {name: wavelength, datatype: float64}\n# - {name: bin_width, datatype: float64}\n"
                 "# - {name: rp^2_value, datatype: float64}\n# - {name: rp^2_errorneg, datatype: float64}\n"
                 "# - {name: rp^2_errorpos, datatype: float64}\n# schema: astropy-2.0\n"
                 "wavelength bin_width rp^2_value rp^2_errorneg rp^2_errorpos\n"
                 "3.0 0.05 0.0210 0.0001 0.00012\n3.1 0.05 0.0211 0.0001 0.00011\n"
                 "3.2 0.05 0.0212 0.0001 0.00010\n3.3 0.05 0.0212 0.0001 0.00010\n")
    data = load_spectra({"G395H": path})
    np.testing.assert_allclose(data.bins[0], [2.95e-6, 3.05e-6])
    np.testing.assert_allclose(data.errors[:2], [1.1e-4, 1.05e-4])


def test_eureka_radius_ratio_is_squared(write):
    path = write("rp.txt", "wavelength bin_width rp_value rp_errorneg rp_errorpos\n"
                 "3.0 0.05 0.1 0.001 0.001\n3.1 0.05 0.11 0.001 0.001\n3.2 0.05 0.12 0.001 0.001\n")
    data = load_spectra({"x": path})
    np.testing.assert_allclose(data.depths, [0.01, 0.0121, 0.0144])
    np.testing.assert_allclose(data.errors, 2 * np.array([0.1, 0.11, 0.12]) * 0.001)


def test_archive_style_full_width_percent_and_crosscheck(write):
    # Units row like the NASA Exoplanet Archive's IPAC tables
    path = write("archive.tbl",
                 "|CENTRALWAVELNG|BANDWIDTH|PL_TRANDEP|PL_TRANDEPERR1|PL_TRANDEPERR2|PL_RATROR|PL_RATRORERR1|PL_RATRORERR2|\n"
                 "|double        |double   |double    |double        |double        |double   |double       |double       |\n"
                 "|microns       |microns  |%         |              |              |         |             |             |\n"
                 "  3.0             0.1       2.25       0.01          -0.012          0.15      0.0003        -0.0003\n"
                 "  3.1             0.1       2.25       0.01          -0.010          0.15      0.0003        -0.0003\n"
                 "  3.2             0.1       2.25       0.01          -0.010          0.15      0.0003        -0.0003\n")
    data = load_spectra({"NIRCam": path})
    np.testing.assert_allclose(data.bins[0], [2.95e-6, 3.05e-6])
    np.testing.assert_allclose(data.depths, 0.0225)
    np.testing.assert_allclose(data.errors[0], 1.1e-4)
    bad = write("bad.tbl", path.read_text().replace("2.25 ", "4.00 "))
    with pytest.raises(ValueError, match="disagree with the squared radius ratios"):
        load_spectra({"NIRCam": bad})


def test_units_from_headers_and_conflicts(write):
    path = write("u.csv", "Wavelength (nm),Depth [ppm],Error [ppm]\n1000,10000,50\n1100,10100,50\n1200,10050,50\n")
    data = load_spectra({"a": path})
    np.testing.assert_allclose(data.bins[0], [0.95e-6, 1.05e-6])
    np.testing.assert_allclose(data.depths[0], 0.01)
    with pytest.raises(ValueError, match="table says its values are in 'ppm'"):
        load_spectra({"a": path}, depth_unit="percent")
    mixed = write("m.csv", "wavelength,depth (ppm),error (%)\n1,10000,0.01\n1.1,10000,0.01\n")
    with pytest.raises(ValueError, match="depths are in ppm but"):
        load_spectra({"a": mixed})


@pytest.mark.parametrize("depth, unit, likely", [
    (10000, None, "ppm"), (2.1, None, "percent"), (0.01, "ppm", "fraction"),
    (21000, "percent", "ppm")])
def test_implausible_depth_units_suggest_the_likely_one(write, depth, unit, likely):
    path = write("d.csv", "wavelength,depth,error\n1,{0},1\n1.1,{0},1\n".format(depth))
    with pytest.raises(ValueError, match="did you mean depth_unit='{}'".format(likely)):
        load_spectra({"a": path}, depth_unit=unit)


@pytest.mark.parametrize("scale, unit, likely", [
    (1000, None, "nm"), (1e-6, "um", "m"), (2e4, "um", "nm' or wavelength_unit='angstrom")])
def test_implausible_wavelength_units_suggest_the_likely_one(write, scale, unit, likely):
    path = write("w.csv", "wavelength,depth,error\n{},0.01,1e-4\n{},0.01,1e-4\n".format(
        1.0 * scale, 1.1 * scale))
    with pytest.raises(ValueError, match="did you mean wavelength_unit='{}'".format(likely)):
        load_spectra({"a": path}, wavelength_unit=unit)


def test_ambiguous_widths_are_inferred_or_refused(write):
    contiguous_full = "wavelength,bin_width,depth,error\n1,0.1,0.01,1e-4\n1.1,0.1,0.01,1e-4\n1.2,0.1,0.01,1e-4\n"
    np.testing.assert_allclose(load_spectra({"a": write("f.csv", contiguous_full)}).bins[0],
                               [0.95e-6, 1.05e-6])
    unclear = "wavelength,bin_width,depth,error\n1,0.03,0.01,1e-4\n1.1,0.03,0.01,1e-4\n1.2,0.03,0.01,1e-4\n"
    path = write("u.csv", unclear)
    with pytest.raises(ValueError, match="width='half' or\\s+width='full'"):
        load_spectra({"a": path})
    np.testing.assert_allclose(load_spectra({"a": path}, width="full").bins[0],
                               [0.985e-6, 1.015e-6])
    with pytest.warns(UserWarning, match="look like full widths"):
        load_spectra({"a": write("f2.csv", contiguous_full)}, width="half")


def test_centres_only_use_midpoints_but_not_across_gaps(write):
    path = write("c.csv", "wavelength,depth,error\n1,0.01,1e-4\n1.4,0.01,1e-4\n2,0.01,1e-4\n")
    np.testing.assert_allclose(load_spectra({"a": path}).bins * 1e6,
                               [[0.8, 1.2], [1.2, 1.7], [1.7, 2.3]])
    gap = write("g.csv", "wavelength,depth,error\n3,0.01,1e-4\n3.1,0.01,1e-4\n3.2,0.01,1e-4\n"
                "3.9,0.01,1e-4\n4.0,0.01,1e-4\n")
    with pytest.raises(ValueError, match="gap after 3.2"):
        load_spectra({"a": gap})


def test_headerless_tables_need_named_columns(write):
    path = write("h.txt", "1 1.2 0.01 1e-4\n1.2 1.4 0.0101 1e-4\n")
    with pytest.raises(ValueError, match="has no header"):
        load_spectra({"a": path})
    data = load_spectra({"a": path}, columns=dict(wavelength_low=0, wavelength_high=1,
                                                  depth=2, error=3))
    np.testing.assert_allclose(data.depths, [0.01, 0.0101])


@pytest.mark.parametrize("text, match", [
    ("wavelength,depth\n1,0.01\n1.1,0.01\n", "no error column"),
    ("wavelength,error\n1,1e-4\n1.1,1e-4\n", "no depth column"),
    ("wavelength,depth,error_low\n1,0.01,1e-4\n1.1,0.01,1e-4\n", "only one of the lower/upper"),
    ("wavelength,depth,dppm,error\n1,0.01,100,1e-4\n1.1,0.01,100,1e-4\n", "could all be the depth"),
    ("wavelength,depth,error\n1,0.01,0\n1.1,0.01,1e-4\n", "errors <= 0"),
    ("wavelength,depth,error\n1,nan,1e-4\n1.1,0.01,1e-4\n", "non-finite"),
    ("wavelength_low,wavelength_high,depth,error\n1.2,1,0.01,1e-4\n", "upper bin edges"),
    ("wavelength_low,depth,error\n1,0.01,1e-4\n", "only one bin-edge"),
])
def test_bad_tables_name_the_problem(write, text, match):
    with pytest.raises(ValueError, match=match):
        load_spectra({"a": write("bad.csv", text)})


def test_offsets_reference_and_shared_offsets(write):
    a, b, c = (write(n + ".csv", EDGES.replace("0.01,", "0.0{},".format(i)))
               for i, n in enumerate(("soss", "nrs1", "nrs2"), start=1))
    data = load_spectra({"NIRISS": a, "NRS1": b, "NRS2": c})
    assert data.ranges == {"NIRISS": (0, 3), "NRS1": (3, 6), "NRS2": (6, 9)}
    assert data.offsets == {"offset_NRS1": (3, 6), "offset_NRS2": (6, 9)}
    assert load_spectra({"NIRISS": a, "NRS1": b, "NRS2": c},
                        reference="NRS1").offsets == {"offset_NIRISS": (0, 3), "offset_NRS2": (6, 9)}
    shared = load_spectra({"NIRISS": a, "NRS1": b, "NRS2": c},
                          offsets={"offset_G395H": ["NRS1", "NRS2"]})
    assert shared.offsets == {"offset_G395H": (3, 9)}
    assert load_spectra({"NIRISS": a, "NRS1": b}, offsets={}).offsets == {}
    assert "offset_NRS1" in str(data) and "(reference)" in str(data)
    with pytest.raises(ValueError, match="listed next to each other"):
        load_spectra({"NRS1": b, "NIRISS": a, "NRS2": c},
                     offsets={"offset_G395H": ["NRS1", "NRS2"]})
    with pytest.raises(ValueError, match="degenerate with the planet radius"):
        load_spectra({"NIRISS": a, "NRS1": b}, offsets={"o1": "NIRISS", "o2": "NRS1"})
    with pytest.raises(ValueError, match="not one of the datasets"):
        load_spectra({"NIRISS": a, "NRS1": b}, reference="NRS3")
    with pytest.raises(ValueError, match="not both"):
        load_spectra({"NIRISS": a, "NRS1": b}, reference="NIRISS", offsets={})


def test_visits_group_adjacent_datasets(write):
    a, b, c = (write(n + ".csv", EDGES.replace("0.01,", "0.0{},".format(i)))
               for i, n in enumerate(("soss", "nrs1", "nrs2"), start=1))
    data = load_spectra({"NIRISS": a, "NRS1": b, "NRS2": c},
                        visits={"v1": "NIRISS", "v2": ["NRS1", "NRS2"]})
    assert data.visits == {"v1": (0, 3), "v2": (3, 9)}
    with pytest.raises(ValueError, match="not in any visit"):
        load_spectra({"NIRISS": a, "NRS1": b, "NRS2": c}, visits={"v1": "NIRISS"})
    with pytest.raises(ValueError, match="in both visits"):
        load_spectra({"NIRISS": a, "NRS1": b}, visits={"v1": ["NIRISS", "NRS1"], "v2": "NRS1"})
    with pytest.raises(ValueError, match="may only contain"):
        load_spectra({"NIRISS": a, "NRS1": b}, visits={"v.1": ["NIRISS", "NRS1"]})


def test_copy_paste_mistakes_are_caught(write):
    a = write("a.csv", EDGES)
    with pytest.raises(ValueError, match="same file"):
        load_spectra({"x": a, "y": a})
    b = write("b.csv", EDGES)
    with pytest.raises(ValueError, match="identical data"):
        load_spectra({"x": a, "y": b})
    with pytest.raises(ValueError, match="may only contain"):
        load_spectra({"NIRSpec G395H": a})
    with pytest.raises(FileNotFoundError):
        load_spectra({"x": a.with_name("missing.csv")})
    with pytest.raises(ValueError, match="must include 'file'"):
        load_spectra({"x": {"path": a}})


def test_per_dataset_options_override_defaults(write):
    ppm = write("ppm.csv", "wavelength,bin_width,depth,error\n1,0.1,10000,50\n1.1,0.1,10000,50\n1.2,0.1,10000,50\n")
    frac = write("frac.csv", EDGES)
    data = load_spectra({"jwst": ppm, "hst": {"file": frac, "depth_unit": "fraction"}},
                        depth_unit="ppm")
    np.testing.assert_allclose(data.depths[:3], 0.01)
    np.testing.assert_allclose(data.depths[3:], [0.01, 0.0101, 0.0102])


def fit_info_for(data, **kwargs):
    return CombinedRetriever.get_default_fit_info(
        R_sun, M_jup, R_jup, T=1000, transit_offsets=data.offsets, **kwargs)


def check(fit_info, depths, errors=None):
    n = len(depths)
    bins = np.column_stack([np.linspace(1, 2, n), np.linspace(1.1, 2.1, n)]) * 1e-6
    CombinedRetriever._check_data(fit_info, bins, depths,
                                  np.full(n, 1e-4) if errors is None else errors,
                                  None, None, None)


def test_retriever_checks_offsets_against_the_data(write):
    a, b = write("a.csv", EDGES), write("b.csv", EDGES.replace("0.01,", "0.02,"))
    data = load_spectra({"NIRISS": a, "NRS1": b})
    fit_info = fit_info_for(data)
    fit_info.add_uniform_fit_param("offset_NRS1", -500, 500)
    with pytest.raises(ValueError, match="units of depth, not ppm"):
        check(fit_info, data.depths)
    fit_info = fit_info_for(data)
    fit_info.add_uniform_fit_param("offset_NRS1", -5e-4, 5e-4)
    check(fit_info, data.depths)
    with pytest.raises(ValueError, match="only 4 transit depths"):
        check(fit_info, data.depths[:4])
    with pytest.raises(ValueError, match="different lengths"):
        check(fit_info, data.depths, np.full(3, 1e-4))
    excess = fit_info_for(data)
    excess.add_uniform_fit_param("error_excess", 0, 100)
    with pytest.raises(ValueError, match="error_excess"):
        check(excess, data.depths)


def test_retriever_warns_when_every_point_has_a_free_offset():
    fit_info = CombinedRetriever.get_default_fit_info(
        R_sun, M_jup, R_jup, T=1000, transit_offsets={"o1": (0, 3), "o2": (3, 6)})
    for name in ("o1", "o2", "Rp"):
        low, high = (-1e-4, 1e-4) if name != "Rp" else (0.9 * R_jup, 1.1 * R_jup)
        fit_info.add_uniform_fit_param(name, low, high)
    with pytest.warns(UserWarning, match="degenerate with Rp"):
        check(fit_info, np.full(6, 0.01))


def test_retriever_checks_visit_parameters():
    def visit_info():
        return CombinedRetriever.get_default_fit_info(
            R_sun, M_jup, R_jup, T=1000, T_star=4000.,
            transit_visits={"v1": (0, 3), "v2": (3, 6)})
    fit_info = visit_info()
    fit_info.add_uniform_fit_param("v1.f_het", 0, .3)
    with pytest.raises(ValueError, match="T_het is not set"):
        check(fit_info, np.full(6, 0.01))
    fit_info.add_uniform_fit_param("T_het", 2500., 3900.)
    check(fit_info, np.full(6, 0.01))
    every = visit_info()
    every.all_params["T_het"].best_guess = 3000.
    for name in ("f_het", "v1.f_het", "v2.f_het"):
        every.add_uniform_fit_param(name, 0, .3)
    with pytest.raises(ValueError, match="f_het has no effect"):
        check(every, np.full(6, 0.01))
    with pytest.raises(ValueError, match="only 5 transit depths"):
        check(visit_info(), np.full(5, 0.01))


def test_run_drivers_check_data_before_sampling(write):
    data = load_spectra({"NIRISS": write("a.csv", EDGES), "NRS1": write("b.csv", EDGES.replace("0.01,", "0.02,"))})
    fit_info = fit_info_for(data)
    fit_info.add_uniform_fit_param("offset_NRS1", -200, 200)
    retriever = CombinedRetriever()
    with mock.patch.object(retriever, "_make_calculators",
                           side_effect=AssertionError("checked too late")):
        for run in (retriever.run_emcee, retriever.run_dynesty):
            with pytest.raises(ValueError, match="not ppm"):
                run(data.bins, data.depths, data.errors, None, None, None, fit_info)
