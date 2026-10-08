import difflib
import warnings

import numpy as np
from ._params import _UniformParam, _GaussianParam, _Param

# Older parameter names, still accepted when choosing what to fit.  Two-sector
# retrievals now sample unordered sectors and label them cold/hot afterwards.
_RENAMED = {"T_spot": "T_het", "spot_cov_frac": "f_het",
            "cold_fraction": "sector1.fraction"}
_RENAMED_PREFIXES = {"cold.": "sector1.", "hot.": "sector2."}


class FitInfo:
    def __init__(self, guesses_dict):
        self.fit_param_names = []
        self.ordered_pairs = []
        self.all_params = dict()

        for key in guesses_dict:
            self.all_params[key] = _Param(guesses_dict[key])

    def _known_name(self, name):
        """Returns the name under which `name` is stored, translating older
        names; raises a KeyError suggesting close matches if it is unknown."""
        new_name = _RENAMED.get(name)
        for old_prefix, new_prefix in _RENAMED_PREFIXES.items():
            if name.startswith(old_prefix):
                new_name = new_prefix + name[len(old_prefix):]
        if name not in self.all_params and new_name in self.all_params:
            warnings.warn("{} has been renamed {}".format(name, new_name),
                          DeprecationWarning, stacklevel=3)
            return new_name
        if name not in self.all_params:
            close = difflib.get_close_matches(name, list(self.all_params), 3)
            raise KeyError("Unknown parameter {}{}".format(
                name, "; did you mean {}?".format(" or ".join(close))
                if close else ""))
        return name

    def add_uniform_fit_param(self, name, low_lim, high_lim,
                              low_guess=None, high_guess=None):
        '''Fit for the parameter `name` using a uniform prior between `low_lim`
        and `high_lim`.  If using emcee, the walkers' initial values for this
        parameter are randomly selected to be between `low_guess` and
        `high_guess`.  If not specified, `low_guess` is set to `low_lim`, and
        similarly with `high_guess`.'''

        name = self._known_name(name)
        if name in self.fit_param_names:
            raise ValueError("Already fitting for {0}".format(name))

        if low_guess is None:
            low_guess = low_lim
        if high_guess is None:
            high_guess = high_lim
        best_guess = self.all_params[name].best_guess

        param = _UniformParam(best_guess, low_lim, high_lim, low_guess, high_guess)
        self.fit_param_names.append(name)
        self.all_params[name] = param

    def add_ordered_uniform_fit_params(
            self, cold_name, hot_name, low_lim, high_lim,
            low_guess=None, high_guess=None):
        """Fit an exchangeable pair while keeping the cold value first.

        Both values have the same uniform prior. Nested samplers draw both
        values and sort them, so no prior volume is discarded.
        """
        cold_name = self._known_name(cold_name)
        hot_name = self._known_name(hot_name)
        if cold_name == hot_name:
            raise ValueError("Ordered parameters must have different names")
        if cold_name in self.fit_param_names or hot_name in self.fit_param_names:
            raise ValueError("Already fitting for an ordered parameter")
        if self.all_params[cold_name].best_guess > \
           self.all_params[hot_name].best_guess:
            raise ValueError("cold best guess must not exceed hot best guess")
        self.add_uniform_fit_param(
            cold_name, low_lim, high_lim, low_guess, high_guess)
        self.add_uniform_fit_param(
            hot_name, low_lim, high_lim, low_guess, high_guess)
        if not hasattr(self, "ordered_pairs"):
            self.ordered_pairs = []
        self.ordered_pairs.append((cold_name, hot_name))

    def add_gaussian_fit_param(self, name, std, low_guess=None, high_guess=None):
        '''Fit for the parameter `name` using a Gaussian prior with standard
        deviation `std`.  If using emcee, the walkers' initial values for this
        parameter are randomly selected to be between `low_guess` and
        `high_guess`.  If `low_guess` is None, it is set to mean-2*std; if
        `high_guess` is None, it is set to mean+2*std.'''

        name = self._known_name(name)
        if name in self.fit_param_names:
            raise ValueError("Already fitting for {0}".format(name))

        mean = self.all_params[name].best_guess
        if mean is None:
            raise ValueError(
                "{0} has no value to center its Gaussian prior on; set "
                "fit_info.all_params['{0}'].best_guess first".format(name))
        if low_guess is None:
            low_guess = mean - 2 * std
        if high_guess is None:
            high_guess = mean + 2 * std

        param = _GaussianParam(mean, std, low_guess, high_guess)
        self.fit_param_names.append(name)
        self.all_params[name] = param

    def add_gases_clr(self, gases, low_lim=1e-12):
        self.gases = gases
        self.clr_low_lim = low_lim
        ln_limit = np.log(np.min(low_lim))
        n = len(gases)
        clr_min =  (n-1)/n * (ln_limit + np.log(n-1))
        clr_max = -(n-1)/n * ln_limit
        for g in gases[:-1]:
            self.all_params[f'clr_{g}'] = _Param(0)
            self.add_uniform_fit_param(f'clr_{g}', clr_min, clr_max)

    def add_gases_vmr(self, gases, low_lim, high_lim):
        self.gases = gases
        for g in gases[:-1]:
            self.all_params[f'log_{g}'] = _Param(np.log10(low_lim))
            self.add_uniform_fit_param(f'log_{g}', np.log10(low_lim), np.log10(high_lim))
        
    def _interpret_param_array(self, array):
        if len(array) != len(self.fit_param_names):
            raise ValueError("Fit array invalid")

        result = dict()
        for i, key in enumerate(self.fit_param_names):
            result[key] = array[i]

        for key in self.all_params:
            if key not in result:
                result[key] = self.all_params[key].best_guess

        return result

    def _within_limits(self, array):
        if len(array) != len(self.fit_param_names):
            raise ValueError("Fit array invalid")

        for i, key in enumerate(self.fit_param_names):
            if not self.all_params[key].within_limits(array[i]):
                return False

        values = dict(zip(self.fit_param_names, array))
        for cold_name, hot_name in getattr(self, "ordered_pairs", []):
            if values[cold_name] > values[hot_name]:
                return False
        return True

    def _generate_rand_param_arrays(self, num_arrays):
        result = []

        for i in range(num_arrays):
            row = []
            for name in self.fit_param_names:
                if i == 0:
                    # Have one walker with fiducial value
                    row.append(self.all_params[name].best_guess)
                else:
                    row.append(self.all_params[name].get_random_value())
            for cold_name, hot_name in getattr(self, "ordered_pairs", []):
                cold_i = self.fit_param_names.index(cold_name)
                hot_i = self.fit_param_names.index(hot_name)
                row[cold_i], row[hot_i] = sorted(
                    (row[cold_i], row[hot_i]))
            result.append(row)

        return np.array(result)

    def _get(self, name):
        return self.all_params[name].best_guess

    def _get_num_fit_params(self):
        return len(self.fit_param_names)

    def _from_unit_interval(self, index, u):
        name = self.fit_param_names[index]
        return self.all_params[name].from_unit_interval(u)

    def _from_unit_interval_array(self, cube):
        result = np.array([
            self._from_unit_interval(i, u) for i, u in enumerate(cube)])
        for cold_name, hot_name in getattr(self, "ordered_pairs", []):
            cold_i = self.fit_param_names.index(cold_name)
            hot_i = self.fit_param_names.index(hot_name)
            result[cold_i], result[hot_i] = sorted(
                (result[cold_i], result[hot_i]))
        return result

    def _ln_prior(self, array):
        if not self._within_limits(array):
            return -np.inf
        result = 0
        for i, name in enumerate(self.fit_param_names):
            result += self.all_params[name].ln_prior(array[i])

        return result

    def __repr__(self):
        return "Params to fit: {}; all params: {}".format(self.fit_param_names,
                                                          self.all_params)
