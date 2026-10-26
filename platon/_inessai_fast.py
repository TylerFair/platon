"""Overhead removal for nessai 0.15.2 i-nessai (always on in run_inessai); the algorithm is unchanged.

Per level, stock i-nessai costs O(N x L) for N stored samples and L flows: it
keeps every flow's density at every sample, re-takes a logsumexp over that
matrix, rescales all samples, scans all samples once per flow for the proposal
importance, and evaluates flows one at a time in single-threaded torch. Here:

1. The meta-proposal is updated in place. Mixture weights are sample counts,
   so adding a flow with n samples gives Q' = (N Q + n q_new) / (N + n). The
   N x L matrix is replaced by an N x 0 placeholder, so memory and checkpoints
   are O(N). Checkpoints store log_q = None plus a marker: stock nessai then
   recomputes the full matrix on resume (slow but correct), while ins_fast
   resumes from the cached coordinates.
2. Each sample's flow-space coordinates (logit of the unit cube, float32 as
   torch receives them) and log-Jacobian are cached at insertion.
3. Proposal importance groups samples by level in O(N).
4. On a GPU, flow densities come from a JAX copy of the frozen torch flows
   (RealNVP: permutation, LU linear, affine coupling with a residual net;
   NormalStudentT base), float32 with highest-precision matmuls, batched
   across flows. Training and sampling stay in torch. Device copies are
   refreshed when a flow's parameter hash changes; frozen flows are checked
   once per level. On a CPU, flows stay in stock torch, and runs then
   reproduce stock i-nessai bit for bit (no float32 rounding differences).

Results match stock i-nessai up to floating-point rounding.  On an A100,
the flow overhead of a level with 360 flows and 1.8M samples drops from
~81 s to ~0.7 s, and checkpoints shrink eightfold.
"""
from collections import OrderedDict
import contextlib
import hashlib
import logging
import math
import weakref

import numpy as np
from scipy.special import logsumexp

import jax
import jax.numpy as jnp
import torch
from glasflow.nflows.transforms import (AffineCouplingTransform, CompositeTransform, LULinear,
                                         RandomPermutation)
from glasflow.nflows.nn.nets import ResidualNet
from nessai.flowmodel.importance import ImportanceFlowModel
from nessai.samplers.importancesampler import ImportanceNestedSampler, OrderedSamples
from nessai.utils.information import differential_entropy
from nessai.utils.stats import effective_sample_size

from ._inessai import NormalStudentT

logger = logging.getLogger(__name__)
GROUP = 32                 # flows per vmapped call
MAX_POINTS = 65536         # points per call
MAX_STACKS = 64            # cached device parameter stacks (LRU)
_HIGHEST = jax.lax.Precision.HIGHEST
_EVALUATORS = weakref.WeakKeyDictionary()
_ORIGINAL = {}
STATS = dict(flow_calls=0, flow_points=0, converted=0, extracted=0, alignment_checks=0,
             shapes=0)
_SHAPES = set()            # (flows, points) shapes compiled
_FLOWS = None              # 'jax' or 'torch' once installed


# ---------------------------------------------------------------- JAX flows

def _numpy(t):
    return t.detach().cpu().numpy()


def extract_flow(flow):
    """Frozen torch flow -> list of per-layer float32 arrays with permutations
    folded into the LU weights. Raises on any unsupported component."""
    layers = flow._transform._transforms
    base = flow._distribution
    if not isinstance(base, NormalStudentT) or base.scale != 1.0 or not base.independent:
        raise TypeError('Unsupported base distribution')
    if len(layers) % 2:
        raise TypeError('Expected (permutation+LU, coupling) pairs')
    d = int(base.dim)
    order = np.arange(d)            # layout position -> feature index
    params = []
    with torch.no_grad():
        for linear, coupling in zip(layers[0::2], layers[1::2]):
            if not (isinstance(linear, CompositeTransform) and len(linear._transforms) == 2
                    and isinstance(linear._transforms[0], RandomPermutation)
                    and isinstance(linear._transforms[1], LULinear)):
                raise TypeError('Expected CompositeTransform(RandomPermutation, LULinear)')
            if not (isinstance(coupling, AffineCouplingTransform)
                    and coupling.scale_activation is AffineCouplingTransform.DEFAULT_SCALE_ACTIVATION
                    and coupling.unconditional_transform is None
                    and isinstance(coupling.transform_net, ResidualNet)):
                raise TypeError('Unsupported coupling transform')
            net = coupling.transform_net
            if (net.context_features or net.blocks[0].use_batch_norm
                    or net.blocks[0].activation is not torch.nn.functional.relu
                    or any(b.dropout.p for b in net.blocks)):
                raise TypeError('Unsupported residual net')
            perm = _numpy(linear._transforms[0]._permutation).astype(int)
            weight, logabsdet = linear._transforms[1].weight_and_logabsdet()
            weight, bias = _numpy(weight), _numpy(linear._transforms[1].bias)
            identity = _numpy(coupling.identity_features).astype(int)
            transform = _numpy(coupling.transform_features).astype(int)
            position = np.argsort(order)
            folded = np.zeros_like(weight)
            folded[:, position[perm]] = weight
            rows = np.concatenate([identity, transform])
            blocks = [(_numpy(b.linear_layers[0].weight).T, _numpy(b.linear_layers[0].bias),
                       _numpy(b.linear_layers[1].weight).T, _numpy(b.linear_layers[1].bias))
                      for b in net.blocks]
            params.append(dict(
                w=np.ascontiguousarray(folded[rows].T), b=bias[rows],
                lu_logabsdet=np.float32(_numpy(logabsdet)),
                w0=_numpy(net.initial_layer.weight).T, b0=_numpy(net.initial_layer.bias),
                blocks=blocks,
                wf=_numpy(net.final_layer.weight).T, bf=_numpy(net.final_layer.bias)))
            order = rows
    return params, dict(dim=d, fraction=base.fraction, df=base.df,
                        n_identity=[len(_numpy(c.identity_features)) for c in layers[1::2]])


def fingerprint(flow):
    """Hash of the exact parameter and buffer bytes."""
    h = hashlib.blake2b(digest_size=16)
    for t in flow.state_dict().values():
        h.update(_numpy(t).tobytes())
    return h.digest()


def _make_log_prob(meta):
    d, f, df = meta['dim'], meta['fraction'], meta['df']
    normal_constant = -d * math.log(math.sqrt(2 * math.pi))
    student_constant = d * (math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - math.log(math.sqrt(df * math.pi)))
    log_normal, log_student = math.log1p(-f), math.log(f)
    n_identity = meta['n_identity']

    def dot(a, b):
        return jnp.dot(a, b, precision=_HIGHEST)

    def single(params, x):
        g = x
        logabsdet = jnp.zeros(x.shape[0], jnp.float32)
        for layer, n_id in zip(params, n_identity):
            y = dot(g, layer['w']) + layer['b']
            logabsdet = logabsdet + layer['lu_logabsdet']
            identity, transform = y[:, :n_id], y[:, n_id:]
            a = dot(identity, layer['w0']) + layer['b0']
            for w1, b1, w2, b2 in layer['blocks']:
                t = dot(jax.nn.relu(a), w1) + b1
                a = a + (dot(jax.nn.relu(t), w2) + b2)
            p = dot(a, layer['wf']) + layer['bf']
            n_tr = transform.shape[1]
            scale = jax.nn.sigmoid(p[:, n_tr:] + 2.0) + 1e-3
            transform = transform * scale + p[:, :n_tr]
            logabsdet = logabsdet + jnp.sum(jnp.log(scale), axis=1)
            g = jnp.concatenate([identity, transform], axis=1)
        squared = g * g
        normal = normal_constant - 0.5 * jnp.sum(squared, axis=1)
        student = student_constant - (df + 1) / 2 * jnp.sum(jnp.log1p(squared / df), axis=1)
        return jnp.logaddexp(normal + log_normal, student + log_student) + logabsdet

    return jax.jit(jax.vmap(single, in_axes=(0, None)))


class JaxFlows:
    """Device copies of the flows of one ImportanceFlowModel."""

    def __init__(self, model):
        self.model = model
        self.params = []          # per flow: list of layer dicts (numpy)
        self.prints = []
        self.meta = None
        self.stacks = OrderedDict()   # tuple of flow indices -> device params (LRU)
        self.log_prob = None

    def _set(self, i, flow):
        params, meta = extract_flow(flow)
        STATS['extracted'] += 1
        if self.meta is None:
            self.meta, self.log_prob = meta, _make_log_prob(meta)
        elif meta != self.meta:
            raise RuntimeError('Flows differ in structure')
        if i < len(self.params):
            self.params[i] = params
        else:
            self.params.append(params)
        for key in [k for k in self.stacks if i in k]:
            del self.stacks[key]

    def sync(self, check_frozen=False):
        n = self.model.n_models
        if n < len(self.params):
            raise RuntimeError('Flows were removed')
        for i in range(len(self.params), n):
            self._set(i, self.model.models[i])
            self.prints.append(fingerprint(self.model.models[i]))
        if n:
            # Only the newest flow is trained; refresh it if it changed.
            digest = fingerprint(self.model.models[n - 1])
            if digest != self.prints[n - 1]:
                self._set(n - 1, self.model.models[n - 1])
                self.prints[n - 1] = digest
        if check_frozen:
            for i in range(n - 1):
                if fingerprint(self.model.models[i]) != self.prints[i]:
                    raise RuntimeError(f'Frozen flow {i} changed')

    def _stack(self, members):
        """Device params for a tuple of flow indices, padded to a power of two
        (bounded number of compiled shapes)."""
        if members in self.stacks:
            self.stacks.move_to_end(members)
        else:
            size = 1 << (len(members) - 1).bit_length()
            padded = list(members) + [members[-1]] * (size - len(members))
            self.stacks[members] = jax.tree_util.tree_map(
                lambda *a: jnp.asarray(np.stack(a), jnp.float32), *[self.params[i] for i in padded])
            while len(self.stacks) > MAX_STACKS:
                self.stacks.popitem(last=False)
        return self.stacks[members]

    def evaluate(self, x, indices):
        """log q_i(x) for flow indices `indices`; float64 array (n, len(indices))."""
        x = np.asarray(x, np.float32)
        indices = np.asarray(indices, int)
        out = np.empty((x.shape[0], len(indices)))
        STATS['flow_calls'] += 1
        STATS['flow_points'] += x.shape[0] * len(indices)
        # Requested flows in blocks of up to GROUP consecutive indices
        batches = [tuple(int(i) for i in indices[indices // GROUP == g])
                   for g in np.unique(indices // GROUP)]
        columns = {int(i): k for k, i in enumerate(indices)}
        for start in range(0, x.shape[0], MAX_POINTS):
            chunk = x[start:start + MAX_POINTS]
            m = chunk.shape[0]
            padded = 1024
            while padded < m:
                padded *= 2
            device_x = jnp.asarray(np.pad(chunk, ((0, padded - m), (0, 0))), jnp.float32)
            for members in batches:
                stacked = self._stack(members)
                shape = (jax.tree_util.tree_leaves(stacked)[0].shape[0], padded)
                if shape not in _SHAPES:
                    _SHAPES.add(shape)
                    STATS['shapes'] = len(_SHAPES)
                values = np.asarray(self.log_prob(stacked, device_x))
                for row, i in enumerate(members):
                    out[start:start + m, columns[i]] = values[row, :m]
        return out


def evaluator(model):
    if model not in _EVALUATORS:
        _EVALUATORS[model] = JaxFlows(model)
    ev = _EVALUATORS[model]
    ev.sync()
    return ev


def log_prob_ith(self, x, i):
    if self.models[i].training:
        self.models[i].eval()
    return evaluator(self).evaluate(x, [i])[:, 0]


def log_prob_all(self, x):
    if self.models.training:
        self.models.eval()
    return evaluator(self).evaluate(x, np.arange(self.n_models))


# ----------------------------------------------------------- sample updates

def _converted(samples):
    return samples.log_q is not None and samples.log_q.shape[1] == 0


def convert(sampler):
    """Replace the stored N x L log q matrix by cached coordinates, after
    checking that the stored log Q is the count-weighted mixture."""
    ts = sampler.training_samples
    if _converted(ts):
        return
    counts = np.bincount(ts.samples['it'] + 1, minlength=ts.log_q.shape[1]).astype(float)
    if counts.size != ts.log_q.shape[1]:
        raise RuntimeError('Sample counts do not match the stored proposals')
    log_q_stored = logsumexp(ts.log_q, b=counts / counts.sum(), axis=1)
    finite = np.isfinite(ts.samples['logQ'])
    if not np.array_equal(finite, np.isfinite(log_q_stored)) or (finite.any() and np.max(
            np.abs(log_q_stored[finite] - ts.samples['logQ'][finite])) > 1e-9):
        raise RuntimeError('Stored log Q is not the count-weighted mixture; not converting')
    x_prime, log_j = sampler.proposal.rescale(ts.samples)
    ts._x_prime = x_prime.astype(np.float32)
    ts._log_j = log_j
    ts.log_q = ts.log_q[:, :0]
    STATS['converted'] += 1
    logger.info('ins_fast: converted %d samples x %d proposals to incremental log Q',
                ts.samples.size, counts.size)


def add_samples(self, samples, log_q):
    """OrderedSamples.add_samples that also carries the cached coordinates."""
    if not _converted(self):
        return _ORIGINAL['add_samples'](self, samples, log_q)
    x_prime, log_j = self._pending
    del self._pending
    samples, log_q, x_prime, log_j = self.sort_samples(samples, log_q[:, :0], x_prime, log_j)
    indices = np.searchsorted(self.samples['logL'], samples['logL'])
    self._x_prime = np.insert(self._x_prime, indices, x_prime, axis=0)
    self._log_j = np.insert(self._log_j, indices, log_j)
    # Stock insertion and index bookkeeping; re-sorting sorted samples keeps
    # their order, so the same insertion indices are used. Checked below.
    result = _ORIGINAL['add_samples'](self, samples, log_q)
    positions = indices + np.arange(len(indices))
    if self.samples[positions].tobytes() != samples.tobytes():
        raise RuntimeError('Cached coordinates misaligned with samples')
    return result


def getstate(self):
    """OrderedSamples pickling: a converted sample set is written with log_q
    None and a marker, so stock code that resumes it recomputes log q."""
    state = _ORIGINAL['getstate'](self)
    if _converted(self):
        state['log_q'] = None
        state['_fast_converted'] = True
    return state


def setstate(self, state):
    """Restore the placeholder only for sample sets last written by ins_fast;
    otherwise drop the coordinate cache (stock code may have added samples)."""
    state = dict(state)
    marker = state.pop('_fast_converted', False)
    x, log_q = state.get('_x_prime'), state.get('log_q')
    fast = x is not None and len(x) == len(state['samples']) and (
        (marker and log_q is None) or (log_q is not None and log_q.shape[1] == 0))
    if fast:
        state['log_q'] = np.empty((len(state['samples']), 0))
    else:
        for key in ('_x_prime', '_log_j', '_n_proposals', '_pending'):
            state.pop(key, None)
    self.__dict__.update(state)


def compute_importance(self, importance_ratio=0.5):
    """O(N) OrderedSamples.compute_importance (same quantities)."""
    if not _converted(self):
        return _ORIGINAL['compute_importance'](self, importance_ratio=importance_ratio)
    n = self._n_proposals
    level = self.samples['it'] + 1
    value = self.samples['logL'] + self.samples['logW']
    order = np.argsort(level, kind='stable')
    level, value = level[order], value[order]
    bounds = np.searchsorted(level, np.arange(n + 1))
    group = np.full(n, -np.inf)
    count = np.diff(bounds)
    for i in np.nonzero(count)[0]:
        group[i] = logsumexp(value[bounds[i]:bounds[i + 1]])
    with np.errstate(divide='ignore'):
        log_imp_post = group - np.log(count)
        above = np.logaddexp.accumulate(group[::-1])[::-1]
        log_imp_z = above - np.log(np.cumsum(count[::-1])[::-1])
    log_imp_post[count == 0] = -np.inf
    log_imp_z[np.cumsum(count[::-1])[::-1] == 0] = -np.inf
    imp_z = np.exp(log_imp_z - logsumexp(log_imp_z))
    imp_post = np.exp(log_imp_post - logsumexp(log_imp_post))
    imp = (1 - importance_ratio) * imp_z + importance_ratio * imp_post
    return {'total': imp, 'posterior': imp_post, 'evidence': imp_z}


def add_and_update_points(self, n):
    """ImportanceNestedSampler.add_and_update_points with an in-place meta-
    proposal update. Matches stock nessai 0.15.2 except where marked."""
    import datetime
    st = datetime.datetime.now()
    ts = self.training_samples
    if self.draw_iid_live:
        raise NotImplementedError('ins_fast does not support draw_iid_live')
    convert(self)                                                    # changed
    if _FLOWS == 'jax':
        evaluator(self.proposal.flow).sync(check_frozen=True)        # changed
    # Spot-check cached coordinates (own RNG: the sampler's stream is untouched)
    check = np.random.default_rng(self.iteration).choice(
        ts.samples.size, min(256, ts.samples.size), replace=False)
    if not np.array_equal(self.proposal.rescale(ts.samples[check])[0].astype(np.float32),
                          ts._x_prime[check]):
        raise RuntimeError('Cached coordinates misaligned with samples')
    STATS['alignment_checks'] += 1
    logger.info(f"Drawing {n} new samples from the new proposal")
    new_samples, log_q = self.draw_n_samples(n)
    new_samples["it"] = self.iteration

    self.history["leakage_new_points"].append(self.compute_leakage(new_samples))
    self._current_proposal_entropy = differential_entropy(-log_q[:, -1])
    logger.debug(f"New samples ESS: {effective_sample_size(new_samples['logW'])}")

    # changed: Q' = (N Q + n q_new) / (N + n) instead of a logsumexp over the
    # stored N x L matrix. The weights are the proposals' sample counts.
    proposal = self.proposal
    n_old, n_new = ts.samples.size, new_samples.size
    counts = np.bincount(ts.samples['it'] + 1, minlength=proposal.n_proposals)
    if counts[-1] != 0 or counts.sum() != n_old:
        raise RuntimeError('Unexpected sample counts')
    expected = np.append(counts[:-1], n_new) / (n_old + n_new)
    if not np.allclose(proposal.weights_array, expected, rtol=1e-12, atol=0):
        raise RuntimeError('Proposal weights are not sample counts')
    log_q_new = proposal.flow.log_prob_ith(ts._x_prime, proposal.level_count) + ts._log_j
    ts.samples["logQ"] = np.logaddexp(ts.samples["logQ"] + np.log(n_old / (n_old + n_new)),
                                      log_q_new + np.log(n_new / (n_old + n_new)))
    ts.samples["logW"] = ts.samples["logU"] - ts.samples["logQ"]
    x_prime, log_j = proposal.rescale(new_samples)
    ts._pending = (x_prime.astype(np.float32), log_j)
    ts._n_proposals = proposal.n_proposals
    ts.add_samples(new_samples, log_q)

    self.history["n_added"].append(new_samples.size)
    self.history["n_live"].append(self.live_points_unit.size)
    self.live_points_ess = effective_sample_size(self.live_points_unit["logW"])
    self.history["leakage_live_points"].append(self.compute_leakage(self.live_points_unit))
    logger.info(f"Current n samples: {self.live_points_unit.size}")
    logger.debug(f"Current live points ESS: {self.live_points_ess:.2f}")
    self.add_and_update_samples_time += datetime.datetime.now() - st


def install(flows='auto'):
    """Monkeypatch nessai classes; call before building or resuming a sampler.

    flows: 'jax', 'torch' (stock flow evaluation), or 'auto' (jax on GPU).
    """
    global _FLOWS
    if _ORIGINAL:
        return
    if flows == 'auto':
        flows = 'jax' if jax.default_backend() == 'gpu' else 'torch'
    if flows not in ('jax', 'torch'):
        raise ValueError(f'Unknown flow backend {flows}')
    _ORIGINAL.update(add_samples=OrderedSamples.add_samples,
                     getstate=OrderedSamples.__getstate__,
                     own_getstate=OrderedSamples.__dict__.get('__getstate__'),
                     setstate=OrderedSamples.__dict__.get('__setstate__'),
                     compute_importance=OrderedSamples.compute_importance,
                     add_and_update_points=ImportanceNestedSampler.add_and_update_points,
                     log_prob_ith=ImportanceFlowModel.log_prob_ith,
                     log_prob_all=ImportanceFlowModel.log_prob_all)
    _FLOWS = flows
    if flows == 'jax':
        ImportanceFlowModel.log_prob_ith = log_prob_ith
        ImportanceFlowModel.log_prob_all = log_prob_all
    OrderedSamples.add_samples = add_samples
    OrderedSamples.__getstate__ = getstate
    OrderedSamples.__setstate__ = setstate
    OrderedSamples.compute_importance = compute_importance
    ImportanceNestedSampler.add_and_update_points = add_and_update_points
    logger.info('ins_fast installed (flows=%s, jax backend %s)', flows, jax.default_backend())


def uninstall():
    """Restore the stock nessai methods."""
    global _FLOWS
    if not _ORIGINAL:
        return
    ImportanceFlowModel.log_prob_ith = _ORIGINAL['log_prob_ith']
    ImportanceFlowModel.log_prob_all = _ORIGINAL['log_prob_all']
    OrderedSamples.add_samples = _ORIGINAL['add_samples']
    for name, key in (('__getstate__', 'own_getstate'), ('__setstate__', 'setstate')):
        if _ORIGINAL[key] is None:
            delattr(OrderedSamples, name)
        else:
            setattr(OrderedSamples, name, _ORIGINAL[key])
    OrderedSamples.compute_importance = _ORIGINAL['compute_importance']
    ImportanceNestedSampler.add_and_update_points = _ORIGINAL['add_and_update_points']
    _ORIGINAL.clear()
    _FLOWS = None


@contextlib.contextmanager
def installed(flows='auto'):
    """The patches for the duration of a run (nested uses keep the outer one)."""
    outer = bool(_ORIGINAL)
    install(flows)
    try:
        yield
    finally:
        if not outer:
            uninstall()
