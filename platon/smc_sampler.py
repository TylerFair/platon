"""Adaptive tempered SMC on a unit-cube prior, with delayed acceptance.

Likelihoods take an array of unit-cube rows and return log likelihoods.
GaussianModel additionally exposes predictions for a spectrum correction and
Gauss-Newton radius map. Sequential cross-fitting freezes all mutation fits
on the current complement. Mutation is invariant at fixed beta given that
complement; adaptive tempering/resampling still preclude a finite-N unbiased
evidence claim. Fitting on self (folds=1) is biased and kept for comparison.
"""
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, ndtr


class GaussianModel:
    """Batched ``predict(u) -> (mean, variance)`` and observed data."""

    def __init__(self, predict, data):
        self.predict = predict
        self.data = np.asarray(data, float)

    def loglike_from(self, mean, var):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            ll = -0.5 * np.sum((mean - self.data)**2 / var +
                               np.log(2 * np.pi * var), axis=1)
        return np.where(np.isfinite(ll) & np.all(var > 0, axis=1), ll, -np.inf)

    def __call__(self, u):
        return self.loglike_from(*self.predict(u))


def _unit(z):
    return np.clip(ndtr(z), 1e-12, 1 - 1e-12)


def _weights(logw):
    return np.exp(logw - logsumexp(logw))


def _next_beta(ll, beta, target):
    def enough(b):
        return 1 / np.sum(_weights((b - beta) * ll)**2) >= target
    if enough(1.0):
        return 1.0
    lo, hi = beta, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if enough(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _systematic(rng, weights):
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, (rng.random() + np.arange(len(weights))) / len(weights))


def _resampling_folds(z_before, parents, k, rng):
    # Deduplicate *current values*, not old root lineage labels: different
    # moved descendants are distinct current training observations. This
    # rejects immediate self-copy leakage without accumulating entire roots
    # into a single fold after many resampling generations.
    _, inverse = np.unique(z_before, axis=0, return_inverse=True)
    family = inverse[parents]
    ids, counts = np.unique(family, return_counts=True)
    loads = np.zeros(k, int)
    assignment = {}
    # Random tie order independent of coordinates; largest families first
    # balance the population while never splitting one exact clone family.
    order = rng.permutation(len(ids))
    order = order[np.argsort(-counts[order], kind='stable')]
    for j in order:
        choices = np.flatnonzero(loads == loads.min())
        f = int(rng.choice(choices))
        assignment[int(ids[j])] = f
        loads[f] += counts[j]
    labels = np.array([assignment[int(x)] for x in family])
    return [np.flatnonzero(labels == f) for f in range(k) if np.any(labels == f)]


class _Correction:
    """Ridge quadratic fit, frozen throughout active-fold mutation."""

    def _features(self, x):
        i, j = np.triu_indices(x.shape[1])
        return np.column_stack([np.ones(len(x)), x, x[:, i] * x[:, j]])

    def __init__(self, z, diff, ridge):
        self.mu = z.mean(0)
        self.sd = z.std(0) + 1e-9
        x = (z - self.mu) / self.sd
        self.lo, self.hi = x.min(0), x.max(0)
        f = self._features(x)
        penalty = np.full(f.shape[1], ridge)
        penalty[0] = 1e-8
        self.coef = np.linalg.solve(f.T @ f + np.diag(penalty), f.T @ diff)

    def __call__(self, z):
        return self._features(np.clip((z - self.mu) / self.sd,
                                      self.lo, self.hi)) @ self.coef


class _Reference:
    """Student-t reference fitted independently of the move implementation."""

    def __init__(self, x, dof):
        self.mu = x.mean(0)
        dx = x - self.mu
        cov = dx.T @ dx / len(x) + 1e-9 * np.eye(x.shape[1])
        self.chol = np.linalg.cholesky(cov)
        self.dof = dof

    def density(self, x):
        y = solve_triangular(self.chol, (x - self.mu).T, lower=True).T
        maha = np.sum(y**2, axis=1)
        return -0.5 * (self.dof + x.shape[1]) * np.log1p(maha / self.dof), maha

    def propose(self, x, maha, rho, rng):
        scale = 0.5 * (self.dof + maha) / rng.gamma(
            0.5 * (self.dof + x.shape[1]), size=len(x))
        return (self.mu + np.sqrt(1 - rho**2) * (x - self.mu) +
                rho * np.sqrt(scale)[:, None] *
                (rng.standard_normal(x.shape) @ self.chol.T))


class _StiffMap:
    """One-coordinate conditional-mode map with an exact Jacobian.

    The predictor, one Newton step and conditional scale depend only on the
    free coordinates. This makes the map deterministic and invertible even
    when the local quadratic approximation is poor.
    """

    def __init__(self, z, index, beta, surrogate, predict=None, data=None):
        self.index, self.beta, self.surrogate = index, beta, surrogate
        self.predict, self.data = predict, data
        self.free = [i for i in range(z.shape[1]) if i != index]
        x = np.column_stack([np.ones(len(z)), z[:, self.free]])
        self.coef = np.linalg.lstsq(x, z[:, index], rcond=None)[0]

    def _block(self, r):
        s = np.column_stack([np.ones(len(r)), r]) @ self.coef
        z = np.empty((len(r), r.shape[1] + 1))
        z[:, self.free], z[:, self.index] = r, s
        h = 1e-3
        if self.predict is not None:
            zp = z.copy()
            zp[:, self.index] += h
            m, v = self.predict(np.vstack([z, zp]))
            m0, m1 = np.split(m, 2)
            v0, v1 = np.split(v, 2)
            jm, jv = (m1 - m0) / h, (v1 - v0) / h
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                residual = self.data - m0
                g = np.sum(jm * residual / v0 +
                           0.5 * jv * (residual**2 / v0**2 - 1 / v0), axis=1)
                curvature = np.sum(jm**2 / v0 + 0.5 * jv**2 / v0**2, axis=1)
        else:
            zp, zm = z.copy(), z.copy()
            zp[:, self.index] += h
            zm[:, self.index] -= h
            l0, lp, lm = np.split(self.surrogate(np.vstack([z, zp, zm])), 3)
            g = (lp - lm) / (2 * h)
            curvature = np.maximum((2 * l0 - lp - lm) / h**2, 0)
        precision = 1 + self.beta * curvature
        gradient = -s + self.beta * g
        ok = np.isfinite(precision) & np.isfinite(gradient) & (precision > 0)
        precision = np.where(ok, precision, 1.0)
        s += np.where(ok, gradient, 0.0) / precision
        return s, np.sqrt(precision)

    def forward(self, z):
        r = z[:, self.free]
        s, scale = self._block(r)
        return np.column_stack([r, (z[:, self.index] - s) * scale]), -np.log(scale)

    def inverse(self, x):
        r = x[:, :-1]
        s, scale = self._block(r)
        z = np.empty_like(x)
        z[:, self.free], z[:, self.index] = r, s + x[:, -1] / scale
        return z, -np.log(scale)


def _move(z, ll, beta, surrogate, mapping, reference, steps, rho, rng):
    x, jac = (z.copy(), np.zeros(len(z))) if mapping is None else mapping.forward(z)
    lt, maha = reference.density(x)
    lp = -0.5 * np.sum(z**2, axis=1) + beta * ll + jac
    accepted = 0
    for _ in range(steps):
        y = reference.propose(x, maha, rho, rng)
        zy, jy = (y, np.zeros(len(y))) if mapping is None else mapping.inverse(y)
        ly = surrogate(zy)
        lty, my = reference.density(y)
        lpy = -0.5 * np.sum(zy**2, axis=1) + beta * ly + jy
        with np.errstate(invalid="ignore"):
            accept = np.log(rng.random(len(z))) < (lpy - lty) - (lp - lt)
        accept &= np.isfinite(ly)
        x[accept], z[accept], ll[accept] = y[accept], zy[accept], ly[accept]
        lp[accept], lt[accept], maha[accept] = lpy[accept], lty[accept], my[accept]
        accepted += accept.sum()
    return z, ll, accepted / (len(z) * steps)


def run_smc(loglike, ndim, n_particles=512, cheap_loglike=None,
            inner_steps=60, outer_steps=12, seed=None, ess_fraction=0.5,
            t_dof=5.0, ridge=1.0, archive_max=8000, stiff_index=None,
            folds=2, n_samples=None, max_prior_draws=1000000, log=None,
            adaptive_steps=False, inner_corr=0.5, corr_target=0.1, same_support=False):
    """Return equal-weight unit-cube samples, likelihoods and log evidence.

    Non-finite full likelihoods define the invalid prior region. Initial
    rejection sampling records *all* draws (including invalid ones), and
    log evidence includes the estimated valid prior fraction. Cheap models
    must be finite wherever the full likelihood is finite; invalid cheap
    values fall back to full evaluations, preserving target support.

    ``stiff_index`` enables a scalar probit-coordinate map when rho < 0.4.
    ``folds=2`` fits references, correction and map on the current complement,
    updating folds sequentially. A disposable 16-step complement pilot freezes
    rho and the map choice before active moves. ``folds=1`` fits on self and
    is biased; it retains the legacy algorithm for comparison.
    At fixed beta, mutation is invariant conditional on the complement.
    Adaptive tempering/resampling still preclude finite-N unbiasedness claims.
    Cross-fit corrections use cached current predictions, not archive_max.
    The evidence error is an SMC incremental-weight estimate; it excludes
    adaptation bias and correlation between temperatures.
    Defaults are fixed 60 inner / 12 outer moves. With adaptive_steps=True,
    cross-fitting forecasts inner length from pilot rho, with fixed outer count.
    Only folds=1 retains active acceptance adaptation and outer stopping.
    History corr is a heuristic acceptance-based forecast, not a measured
    autocorrelation; n_parents and n_unique count distinct parents and final rows.
    same_support=True declares identical full/cheap validity regions, allowing
    cheap rejection screening; generic cheap models retain the full fallback.
    """
    integers = (ndim, n_particles, inner_steps, outer_steps, archive_max,
                max_prior_draws, n_samples if n_samples is not None else n_particles)
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in integers):
        raise ValueError("dimensions, population and step counts must be positive integers")
    if n_particles < 2 or not 0 < ess_fraction < 1 or t_dof <= 0 or ridge <= 0:
        raise ValueError("require n_particles >= 2, 0 < ess_fraction < 1, dof and ridge > 0")
    if not isinstance(folds, (int, np.integer)) or not 1 <= folds <= n_particles:
        raise ValueError("folds must be an integer in [1, n_particles]")
    if folds >= 2 and n_particles - int(np.ceil(n_particles / folds)) <= ndim + 2:
        raise ValueError("cross-fit complement must contain more than ndim + 2 rows")
    if not 0 < inner_corr < 1 or not 0 < corr_target < 1:
        raise ValueError("inner_corr and corr_target must be between 0 and 1")
    if stiff_index is not None and not 0 <= stiff_index < ndim:
        raise ValueError("stiff_index outside parameter dimensions")
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    counts = [0, 0]
    archive_z, archive_diff = [], []
    spectral = isinstance(loglike, GaussianModel) and isinstance(cheap_loglike, GaussianModel)

    def evaluate(model, z, which):
        counts[which] += len(z)
        if isinstance(model, GaussianModel):
            mean, var = model.predict(_unit(z))
            return model.loglike_from(mean, var), (mean, var)
        ll = np.asarray(model(_unit(z)), float)
        if ll.shape != (len(z),):
            raise ValueError("likelihood must return one value per unit-cube row")
        return np.where(np.isfinite(ll), ll, -np.inf), None

    def full(z, cached=False):
        keep = np.ones(len(z), bool)
        low, low_prediction = None, None
        if same_support and cheap_loglike is not None:
            low, low_prediction = evaluate(cheap_loglike, z, 1)
            keep = np.isfinite(low)
        ll = np.full(len(z), -np.inf)
        if not keep.any():
            return (ll, None) if cached else ll
        selected = z[keep]
        values, prediction = evaluate(loglike, selected, 0)
        ll[keep] = values
        if cheap_loglike is not None and folds == 1:
            if low is None:
                low, low_prediction = evaluate(cheap_loglike, selected, 1)
            else:
                low = low[keep]
                if low_prediction is not None:
                    low_prediction = tuple(p[keep] for p in low_prediction)
            good = np.isfinite(values) & np.isfinite(low)
            with np.errstate(invalid="ignore"):
                diff = prediction[0] - low_prediction[0] if spectral else values - low
            if good.any():
                archive_z.extend(selected[good])
                archive_diff.extend(diff[good])
                del archive_z[:-archive_max]
                del archive_diff[:-archive_max]
        payload = None
        if cached:
            payload = np.full((len(z), prediction[0].shape[1]), np.nan) if spectral else ll.copy()
            if spectral:
                payload[keep] = prediction[0]
        return (ll, payload) if cached else ll

    zs, ls, ps, drawn, valid = [], [], [], 0, 0
    while valid < n_particles:
        size = min(max(256, n_particles), max_prior_draws - drawn)
        if size <= 0:
            raise RuntimeError("could not initialize valid particles within max_prior_draws")
        z = rng.standard_normal((size, ndim))
        ll, payload = full(z, cached=True)
        ok = np.isfinite(ll)
        drawn += size
        valid += ok.sum()
        zs.extend(z[ok])
        ls.extend(ll[ok])
        if payload is not None:
            ps.extend(payload[ok])
    z, ll = np.array(zs[:n_particles]), np.array(ls[:n_particles])
    payload = np.array(ps[:n_particles])
    log_valid = np.log(valid / drawn)
    logz, variance, beta = log_valid, (1 - valid / drawn) / valid, 0.0
    rho = min(0.99, 2.38 / np.sqrt(ndim))
    use_stiff = False
    history, final_z, final_ll = [], [], []
    requested = n_particles if n_samples is None else n_samples
    while beta < 1 or sum(map(len, final_ll)) < requested:
        next_beta = _next_beta(ll, beta, ess_fraction * n_particles)
        logw = (next_beta - beta) * ll
        w = _weights(logw)
        logz += logsumexp(logw) - np.log(n_particles)
        variance += max(n_particles * np.sum(w**2) - 1, 0) / n_particles
        idx = _systematic(rng, w)
        groups = (_resampling_folds(z, idx, folds, rng) if folds >= 2
                  else [np.arange(n_particles)])
        z, ll, payload, beta = z[idx].copy(), ll[idx].copy(), payload[idx].copy(), next_beta
        population, population_ll, population_payload = z, ll, payload
        inner_accepts, outer_accepts, lengths, rhos = [], [], [], []
        corr_forecast = 0.0
        for active in groups:
            corr_prod = 1.0
            complement = np.ones(n_particles, bool)
            complement[active] = False
            train, train_ll = population[complement], population_ll[complement]
            train_payload = population_payload[complement]
            if folds >= 2 and len(train) <= ndim + 2:
                # An unsplittable clone family can exhaust the complement.
                # Identity mutation is invariant and avoids fitting on self.
                corr_forecast += len(active) / n_particles
                continue
            z, ll = population[active].copy(), population_ll[active].copy()
            payload = population_payload[active].copy()
            corr = None
            if folds == 1 and cheap_loglike is not None and len(archive_z) > 50:
                corr = _Correction(np.array(archive_z), np.array(archive_diff), ridge)

            def predict(z):
                low, prediction = evaluate(cheap_loglike, z, 1)
                mean, var = prediction
                if corr is not None:
                    mean = mean + corr(z)
                return mean, var

            def surrogate(z):
                if cheap_loglike is None:
                    return full(z)
                if spectral:
                    mean, var = predict(z)
                    low = cheap_loglike.loglike_from(mean, var)
                    base_valid = np.all(np.isfinite(var) & (var > 0), axis=1)
                else:
                    low, _ = evaluate(cheap_loglike, z, 1)
                    base_valid = np.isfinite(low)
                    if corr is not None:
                        low = low + corr(z)
                bad = ~np.isfinite(low)
                if same_support:
                    bad &= base_valid
                if bad.any():
                    low[bad] = full(z[bad])
                return low

            mapping, reference = None, None
            if folds >= 2:
                if cheap_loglike is not None:
                    low, prediction = evaluate(cheap_loglike, train, 1)
                    good = np.isfinite(low)
                    with np.errstate(invalid="ignore"):
                        diff = train_payload - prediction[0] if spectral else train_ll - low
                    if good.sum() > 50:
                        corr = _Correction(train[good], diff[good], ridge)

                def fit(stiff):
                    mapping = (_StiffMap(train, stiff_index, beta, surrogate,
                                        predict if spectral else None,
                                        loglike.data if spectral else None) if stiff else None)
                    x = train if mapping is None else mapping.forward(train)[0]
                    return mapping, _Reference(x, t_dof)

                def pilot(rho, mapping, reference):
                    p, pl = train.copy(), surrogate(train)
                    for _ in range(16):
                        p, pl, rate = _move(p, pl, beta, surrogate, mapping,
                                            reference, 1, rho, rng)
                        rho = min(0.99, rho * np.exp(0.5 * (rate - 0.3)))
                    return rho

                mapping, reference = fit(False)
                rho = pilot(min(0.99, 2.38 / np.sqrt(ndim)), mapping, reference)
                if stiff_index is not None and rho < 0.4:
                    mapping, reference = fit(True)
                    rho = pilot(0.5, mapping, reference)
            sl = ll.copy() if cheap_loglike is None else surrogate(z)
            for _ in range(outer_steps):
                if folds == 1:
                    mapping = None
                if folds == 1 and use_stiff:
                    mapping = _StiffMap(z, stiff_index, beta, surrogate,
                                        predict if spectral else None,
                                        loglike.data if spectral else None)
                if folds == 1:
                    x = z if mapping is None else mapping.forward(z)[0]
                    reference = _Reference(x, t_dof)
                n_inner = inner_steps
                if adaptive_steps:
                    per = 1 - 0.3 * (1 - np.sqrt(1 - rho**2))
                    n_inner = int(np.clip(np.ceil(np.log(inner_corr) / np.log(per)),
                                         min(5, inner_steps), inner_steps))
                y, sy, rate = _move(z.copy(), sl.copy(), beta, surrogate, mapping,
                                    reference, n_inner, rho, rng)
                if cheap_loglike is None:
                    z, ll, sl = y, sy, sy.copy()
                    payload = ll.copy()
                    outer_rate = 1.0
                else:
                    moved = np.any(y != z, axis=1)
                    ly = ll.copy()
                    if moved.any():
                        if folds == 1:
                            ly[moved] = full(y[moved])
                        else:
                            ly[moved], proposed_payload = full(y[moved], cached=True)
                    with np.errstate(invalid="ignore"):
                        accept = moved & (np.log(rng.random(len(z))) <
                                          beta * ((ly - sy) - (ll - sl)))
                    accept &= np.isfinite(ly)
                    z[accept], ll[accept], sl[accept] = y[accept], ly[accept], sy[accept]
                    if folds >= 2 and accept.any():
                        payload[accept] = proposed_payload[accept[moved]]
                    outer_rate = accept.mean()
                inner_accepts.append(rate)
                outer_accepts.append(outer_rate)
                lengths.append(n_inner)
                rhos.append(rho)
                c_inner = (1 - rate * (1 - np.sqrt(1 - rho**2)))**n_inner
                corr_prod *= 1 - outer_rate * (1 - c_inner)
                if folds == 1:
                    rho = min(0.99, rho * np.exp(0.5 * (rate - 0.3)))
                if folds == 1 and adaptive_steps and len(lengths) >= min(2, outer_steps) and corr_prod <= corr_target:
                    break
            if folds == 1 and stiff_index is not None and not use_stiff and np.mean(rhos) < 0.4:
                use_stiff, rho = True, 0.5
            corr_forecast += len(active) / n_particles * corr_prod
            population[active], population_ll[active], population_payload[active] = z, ll, payload
        z, ll, payload = population, population_ll, population_payload
        if beta == 1:
            final_z.append(z.copy())
            final_ll.append(ll.copy())
        info = dict(beta=beta, logz=float(logz), rho=float(rho),
                    n_parents=len(np.unique(idx)), n_unique=len(np.unique(z, axis=0)),
                    inner_accept=float(np.mean(inner_accepts)) if inner_accepts else 0.0,
                    outer_accept=float(np.mean(outer_accepts)) if outer_accepts else 0.0,
                    n_inner=float(np.mean(lengths)) if lengths else 0.0, n_outer=len(lengths) / len(groups),
                    corr=float(corr_forecast),
                    n_evals=counts[0], n_cheap_evals=counts[1],
                    median_loglike=float(np.median(ll)),
                    seconds=time.monotonic() - start)
        history.append(info)
        if log is not None:
            log(info)
    pick = rng.permutation(sum(map(len, final_ll)))[:requested]
    return dict(samples=_unit(np.vstack(final_z)[pick]),
                samples_loglike=np.concatenate(final_ll)[pick],
                weights=np.full(requested, 1 / requested), logz=float(logz),
                logz_err=float(np.sqrt(variance)), log_valid_fraction=float(log_valid),
                n_evals=counts[0], n_cheap_evals=counts[1], history=history,
                seconds=time.monotonic() - start)
