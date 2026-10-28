"""Runner lifecycle checks with real Nautilus and a cheap synthetic forward.

These are orchestration tests. Physical scalar/batch parity is also required
by run_wasp94.py itself on the allocated compute node.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import pytest


@pytest.fixture
def run_target(tmp_path):
    nautilus = pytest.importorskip("nautilus")
    script = tmp_path / "run_wasp94.py"
    shutil.copyfile(Path(__file__).with_name("run_wasp94.py"), script)
    # The independent snapshot tests exercise cryptographic verification.
    # Here isolate the real runner/sampler lifecycle from opacity resources.
    (tmp_path / "freeze_sources.py").write_text(
        "import json\n"
        "def verify_snapshot(path):\n"
        "    return json.loads((path / 'manifest.json').read_text())\n")
    snapshot = tmp_path / "snapshot"
    package = snapshot / "platon" / "platon"
    package.mkdir(parents=True)
    (snapshot / "manifest.json").write_text('{"synthetic_target": true}\n')
    (package / "__init__.py").write_text("")
    (package / "combined_retriever.py").write_text('''
import numpy as np
class Parameter:
    def __init__(self, low, high):
        self.low_lim, self.high_lim = low, high
class Fit:
    def __init__(self, n):
        self.fit_param_names = ['sector1.scatt_slope', 'sector2.scatt_slope'] + ['x'+str(i) for i in range(n-2)]
        self.all_params = {name: Parameter(-2, 15) if 'scatt_slope' in name else Parameter(0, 1) for name in self.fit_param_names}
    def _get(self, name):
        p = self.all_params[name]
        return (p.low_lim+p.high_lim)/2
    def _stellar_grid_options(self):
        return {}
    def _from_unit_interval_array(self, u):
        low = np.array([self.all_params[n].low_lim for n in self.fit_param_names])
        high = np.array([self.all_params[n].high_lim for n in self.fit_param_names])
        return low+(high-low)*u
    def _from_unit_interval_batch(self, u):
        return self._from_unit_interval_array(u)
class CombinedRetriever:
    @staticmethod
    def _marginalize(full):
        return Fit(18)
    def _validate_params(self, *args):
        pass
    def _ln_like(self, parameters, calculator, eclipse, fit, *args):
        values = [(v-fit._get(n))/(fit.all_params[n].high_lim-fit.all_params[n].low_lim) for n,v in zip(fit.fit_param_names,parameters)]
        return -float(np.dot(values, values))
''')
    (package / "_batched_likelihood.py").write_text('''
import numpy as np
class BatchedLikelihood:
    def __init__(self, retriever, calculator, eclipse, fit, *args, **kwargs):
        self.r,self.c,self.f = retriever,calculator,fit
    def __call__(self, rows):
        return np.array([self.r._ln_like(row,self.c,None,self.f) for row in rows])
''')
    (snapshot / "platon" / "jax.py").write_text('''
class Device:
    device_kind='synthetic CPU'
    def __str__(self): return 'CPU:0'
def default_backend(): return 'cpu'
def devices(): return [Device()]
''')
    (snapshot / "driver.py").write_text('''
import numpy as np
from platon.combined_retriever import CombinedRetriever, Fit
def load_data():
    data=np.array([(0.,1.)],dtype=[('depth',float),('depth_err',float)])
    return data,np.array([[1.,2.]]),{'synthetic':(0,1)}
def build_fit(data,ranges): return Fit(22)
class Calculator:
    def __init__(self,**kwargs): pass
    def change_wavelength_bins(self,bins): pass
def use_opacities(downsample):
    assert downsample==1
    return Calculator
''')
    shutil.copytree(Path(nautilus.__file__).parent, snapshot / "nautilus" / "nautilus",
                    ignore=shutil.ignore_patterns("__pycache__"))
    output = tmp_path / "result"
    base = [sys.executable, str(script), "--snapshot", str(snapshot), "--output", str(output),
            "--backend", "stock", "--no-streaming-statistics", "--n-live", "100",
            "--n-update", "100", "--n-like-new-bound", "1000", "--n-eff", "100",
            "--n-networks", "1", "--explore-batch", "8", "--sample-batch", "8",
            "--seconds", "20", "--max-calls", "16", "--checkpoint-calls", "8"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(*extra):
        return subprocess.run(base + list(extra), capture_output=True, text=True, env=env, timeout=60)

    return run, output, snapshot


def test_batch_resume_uses_checkpoint_ledger_not_stale_summary(run_target):
    run, output, _ = run_target
    first = run()
    assert first.returncode == 0, first.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["calls"] == 16
    assert summary["timings"]["likelihood_calls"] == 16
    assert summary["native_calls_including_warmup"] == 50
    setup = json.loads((output / "setup.json").read_text())
    assert setup["scalar_batch_parity"]["bitwise_equal"]
    assert not summary["success"]
    # A summary can be stale or absent after a killed process. Durable state
    # and accounting must come from the same atomic HDF5 checkpoint.
    (output / "summary.json").write_text('{"timings": {"likelihood_calls": 999999}}')
    second = run("--max-calls", "24")
    assert second.returncode == 0, second.stderr
    resumed = json.loads((output / "summary.json").read_text())
    assert resumed["calls_before_segment"] == 16
    assert resumed["calls_this_segment"] == 8
    assert resumed["timings"]["likelihood_calls"] == 24
    assert resumed["native_calls_including_warmup"] == 92
    with h5py.File(output / "checkpoint.hdf5", "r") as checkpoint:
        ledger = json.loads(checkpoint.attrs["benchmark_ledger"])
        assert ledger["calls"] == checkpoint["sampler"].attrs["n_like"] == 24
        assert ledger["clean_segment_end"]


def test_resume_rejects_changed_live_population_and_snapshot(run_target):
    run, output, snapshot = run_target
    first = run()
    assert first.returncode == 0, first.stderr
    changed = run("--n-live", "200")
    assert changed.returncode != 0
    assert "Resume refused" in changed.stderr
    (snapshot / "manifest.json").write_text('{"synthetic_target": false}\n')
    changed = run()
    assert changed.returncode != 0
    assert "Resume refused" in changed.stderr
    with h5py.File(output / "checkpoint.hdf5", "r") as checkpoint:
        assert checkpoint["sampler"].attrs["n_like"] == 16


def test_resume_flags_unknown_work_after_ungraceful_stop(run_target):
    run, output, _ = run_target
    first = run("--likelihood", "scalar")
    assert first.returncode == 0, first.stderr
    with h5py.File(output / "checkpoint.hdf5", "a") as checkpoint:
        ledger = json.loads(checkpoint.attrs["benchmark_ledger"])
        ledger["clean_segment_end"] = False
        checkpoint.attrs["benchmark_ledger"] = json.dumps(ledger)
    second = run("--likelihood", "scalar", "--max-calls", "24")
    assert second.returncode == 0, second.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["interrupted_segments"] == 1
    assert summary["lost_work_after_checkpoint_unknown"]
    assert summary["native_calls_including_warmup"] == 28
