import numpy as np
from . import _mie_multi_x

class MieCache:
    def __init__(self):
        self.all_xs = np.array([])
        self.all_Qexts = np.array([])
        self.all_ms = np.array([], dtype=complex)

        
    def get_from_cache(self, m, xs, max_frac_error=0.05):
        xs = np.asarray(xs, dtype=float)
        result = np.ones(len(xs)) * np.nan
        matches = self.all_ms == m
        if not np.any(matches):
            return result
        cached_xs = self.all_xs[matches]
        upper = np.searchsorted(cached_xs, xs)
        right = np.minimum(upper, len(cached_xs) - 1)
        left = np.maximum(upper - 1, 0)
        exact = cached_xs[right] == xs
        # Both interpolation vertices must be nearby and belong to this
        # refractive index; a nearby point of another material is irrelevant.
        interior = (upper > 0) & (upper < len(cached_xs))
        in_cache = exact | (interior &
                            (xs - cached_xs[left] <= max_frac_error * xs) &
                            (cached_xs[right] - xs <= max_frac_error * xs))
        result[in_cache] = np.interp(
            xs[in_cache], cached_xs,
            self.all_Qexts[self.all_ms == m])

        return result
        
    def get_and_update(self, m, xs):
        # Get from cache if available, from Mie calculations if not. 
        # Put results of Mie calculations into cache.
        xs = np.asarray(xs, dtype=float)
        Qexts = self.get_from_cache(m, xs)
        cache_misses = np.isnan(Qexts)
        if np.sum(cache_misses) > 0:
            Qexts[cache_misses] = _mie_multi_x.get_Qext(m, xs[cache_misses])
            self.add(m, xs[cache_misses], Qexts[cache_misses])
        return Qexts

    
    def add(self, m, xs, Qexts, size_limit=1000000):
        if len(xs) == 0:
            return
        
        self.all_xs = np.append(self.all_xs, xs)
        self.all_Qexts = np.append(self.all_Qexts, Qexts)
        self.all_ms = np.append(self.all_ms, np.array([m] * len(xs)))
        if len(self.all_xs) > size_limit:
            to_remove = np.random.choice(
                range(len(self.all_xs)), len(self.all_xs) - size_limit,
                replace=False)
            
            self.all_xs = np.delete(self.all_xs, to_remove)
            self.all_Qexts = np.delete(self.all_Qexts, to_remove)
            self.all_ms = np.delete(self.all_ms, to_remove)

        p = np.argsort(self.all_xs)
        self.all_xs = self.all_xs[p]
        self.all_Qexts = self.all_Qexts[p]
        self.all_ms = self.all_ms[p]
        
