"""Train existing retained FP16 base tensors; no adapter or quantizer changes.

FP32 masters are independent leaves, outside the model parameter inventory.
Every native block/checkpoint replay casts them to FP16 inside functional_call.
The original 507 base tensors remain frozen. Export replaces only the existing
393 retained FP16 tensors; all 114 FP4 code/scale/global buffers stay unchanged.
This helper defines arithmetic and integrity checks, not a training recipe.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

RETAINED_TENSORS = 393
RETAINED_PARAMETERS = 3_580_928
RETAINED_BYTES = 7_161_856
FAMILY_COUNTS = dict(block_and_final_norm=57, gated_norm=56, conv_weight=56,
                     conv_bias=56, A_log=56, dt_bias=56, D=56)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def tensor_hash(value):
    digest = hashlib.sha256()
    value = value.detach()
    if value.ndim == 0:
        digest.update(value.cpu().contiguous().numpy().tobytes())
    else:
        for first in range(0, value.shape[0], 128):
            digest.update(value[first:first + 128].cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_family(name):
    if name.endswith('.mixer.norm.weight'):
        return 'gated_norm'
    if name.endswith('.norm.weight') or name == 'backbone.norm_f.weight':
        return 'block_and_final_norm'
    if name.endswith('.mixer.conv1d.weight'):
        return 'conv_weight'
    if name.endswith('.mixer.conv1d.bias'):
        return 'conv_bias'
    for family in ('A_log', 'dt_bias', 'D'):
        if name.endswith('.mixer.' + family):
            return family
    raise ValueError('Unexpected retained parameter: ' + name)


def retained_inventory(ledger, *, production=True):
    values = ledger['tensors']
    selected = {}
    for name, entry in sorted(values.items()):
        if entry['kind'] == 'fp16':
            parameter_family(name)
            need(math.prod(entry['shape']) == entry['numel'], 'Retained ledger shape/count differs')
            selected[name] = dict(shape=list(entry['shape']), numel=entry['numel'],
                                  decoded_sha256=entry['decoded_sha256'])
        else:
            need(entry['kind'] == 'fp4_g16_e4m3', 'Unexpected quantized representation')
    if production:
        need(ledger['format_name'] == 'fp4_g16_e4m3' and ledger['complete'] is True
             and len(values) == 507 and len(selected) == RETAINED_TENSORS
             and sum(v['numel'] for v in selected.values()) == RETAINED_PARAMETERS
             and Counter(parameter_family(n) for n in selected) == FAMILY_COUNTS,
             'Expected all 393 retained production tensors, including seven parameter families')
    need(selected and 'backbone.norm_f.weight' in selected, 'Final norm must be selected')
    return selected


def _validate_export(model, values, inventory):
    parameters = dict(model.named_parameters())
    need(set(values) == set(inventory), 'FP16 export must contain every selected retained tensor')
    for name, value in values.items():
        need(isinstance(value, torch.Tensor) and value.dtype == torch.float16
             and list(value.shape) == inventory[name]['shape']
             and value.numel() == inventory[name]['numel'] and bool(torch.isfinite(value).all())
             and name in parameters and value.shape == parameters[name].shape,
             'Invalid retained FP16 export: ' + name)


@contextmanager
def install_fp16(model, values, inventory=None):
    """Temporary native readback only; restore original parameter objects always."""
    if inventory is None:
        inventory = {}
        for name, parameter in model.named_parameters():
            try:
                parameter_family(name)
            except ValueError:
                continue
            inventory[name] = dict(shape=list(parameter.shape), numel=parameter.numel())
        if len(dict(model.named_parameters())) == 507:
            need(len(inventory) == RETAINED_TENSORS
                 and sum(v['numel'] for v in inventory.values()) == RETAINED_PARAMETERS,
                 'Native retained tensor geometry differs')
    _validate_export(model, values, inventory)
    previous = {}
    try:
        for name, value in values.items():
            owner, leaf = name.rsplit('.', 1)
            module = model.get_submodule(owner)
            previous[name] = getattr(module, leaf)
            setattr(module, leaf, nn.Parameter(value.detach().to(
                device=previous[name].device, dtype=torch.float16).clone(), requires_grad=False))
        yield
    finally:
        for name, parameter in previous.items():
            owner, leaf = name.rsplit('.', 1)
            setattr(model.get_submodule(owner), leaf, parameter)


class SmallMasters:
    """Differentiable original SSD prefill using all existing small parameters."""
    def __init__(self, model, conversion_ledger, *, production=True):
        self.model = model
        self.production = bool(production)
        self.inventory = retained_inventory(conversion_ledger, production=production)
        self._base = dict(model.named_parameters())
        need(set(self._base) == set(conversion_ledger['tensors']), 'Model/ledger tensor inventory differs')
        need(not model.training and all(p.dtype == torch.float16 and not p.requires_grad
                                       and p.grad is None for p in self._base.values()),
             'An eval-mode, frozen FP16 decoded base with no gradients is required')
        if production:
            need(len(self._base) == 507 and sum(p.numel() for p in self._base.values()) == 8236999680
                 and len(model.backbone.layers) == 56, 'Wrong pure Mamba2-8B geometry')
        need(not model.backbone.fused_add_norm and not model.backbone.residual_in_fp32,
             'Original unfused FP16 residual orchestration is required')
        self._identities = {n: (id(p), p.data_ptr(), p._version) for n, p in self._base.items()}
        self._hashes = {n: entry['decoded_sha256'] for n, entry in conversion_ledger['tensors'].items()}
        self.masters = {}
        for name, entry in self.inventory.items():
            original = self._base[name]
            need(list(original.shape) == entry['shape'] and original.numel() == entry['numel'],
                 'Selected base geometry differs: ' + name)
            self.masters[name] = nn.Parameter(original.detach().float().clone(), requires_grad=True)
        self.block_frozen, self.block_selected = [], []
        mapped = set()
        for index, layer in enumerate(model.backbone.layers):
            need(not layer.fused_add_norm and not layer.residual_in_fp32 and layer.mlp is None
                 and not layer.mixer.use_mem_eff_path, 'Unsupported native block path')
            for module in (layer.mixer, layer.mixer.norm):
                need(not module._forward_pre_hooks and not module._forward_hooks
                     and getattr(module.forward, '__func__', None) is type(module).forward,
                     'Original native forwards with no adapter/state-controller hooks are required')
            frozen = dict(layer.named_parameters()) | dict(layer.named_buffers())
            prefix = f'backbone.layers.{index}.'
            selected = {name[len(prefix):]: name for name in self.masters if name.startswith(prefix)}
            need(set(selected).issubset(frozen), 'Selected block tensor missing')
            self.block_frozen.append(frozen); self.block_selected.append(selected)
            mapped.update(selected.values())
        mapped.add('backbone.norm_f.weight')
        need(mapped == set(self.masters), 'Selected tensors must all participate in native functional mapping')
        self.final_frozen = dict(model.backbone.norm_f.named_parameters()) | dict(model.backbone.norm_f.named_buffers())

    def parameters(self):
        return list(self.masters.values())

    def state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self.masters.items()}

    def load_state_dict(self, values):
        need(set(values) == set(self.masters), 'Master checkpoint keys differ')
        for name, value in values.items():
            need(value.dtype == torch.float32 and value.shape == self.masters[name].shape
                 and bool(torch.isfinite(value).all()), 'Invalid FP32 master: ' + name)
        with torch.no_grad():
            for name, value in values.items():
                self.masters[name].copy_(value)

    def assert_frozen(self, check_values=False):
        now = dict(self.model.named_parameters())
        need(set(now) == set(self._base), 'Original base parameter inventory changed')
        for name, parameter in now.items():
            need((id(parameter), parameter.data_ptr(), parameter._version) == self._identities[name]
                 and not parameter.requires_grad and parameter.grad is None,
                 'Original base identity/version/gradient changed: ' + name)
            if check_values:
                need(tensor_hash(parameter) == self._hashes[name], 'Original base bytes changed: ' + name)
        return dict(complete=True, tensors=len(now), parameters=sum(p.numel() for p in now.values()),
                    identity_version_gradients_unchanged=True, actual_content_checked=bool(check_values))

    def forward(self, input_ids, use_checkpoint=True):
        hidden = self.model.backbone.embedding(input_ids)
        residual = None
        for index, layer in enumerate(self.model.backbone.layers):
            def run_block(h, r, index=index, layer=layer):
                values = dict(self.block_frozen[index])
                # Cast inside EVERY original invocation and backward replay.
                for local, name in self.block_selected[index].items():
                    values[local] = self.masters[name].to(torch.float16)
                return functional_call(layer, values, (h, r), {'inference_params': None},
                                       tie_weights=False, strict=True)
            if use_checkpoint and torch.is_grad_enabled():
                hidden, residual = checkpoint(run_block, hidden, residual, use_reentrant=False,
                                               preserve_rng_state=True)
            else:
                hidden, residual = run_block(hidden, residual)
        residual = hidden + residual if residual is not None else hidden
        final = dict(self.final_frozen)
        final['weight'] = self.masters['backbone.norm_f.weight'].to(torch.float16)
        return functional_call(self.model.backbone.norm_f, final,
            (residual.to(dtype=self.model.backbone.norm_f.weight.dtype),), tie_weights=False, strict=True)

    @torch.no_grad()
    def export_fp16(self):
        self.stability_receipt()
        values = {name: value.detach().cpu().half().clone() for name, value in self.masters.items()}
        _validate_export(self.model, values, self.inventory)
        if self.production:
            need(sum(v.numel() * v.element_size() for v in values.values()) == RETAINED_BYTES,
                 'Retained export payload size changed')
        self.assert_frozen()
        return values

    @torch.no_grad()
    def stability_receipt(self):
        decay, bias_delta = [], []
        for name, master in self.masters.items():
            rounded = master.half()
            need(bool(torch.isfinite(master).all()) and bool(torch.isfinite(rounded).all()),
                 'Nonfinite master or FP16 cast: ' + name)
            if name.endswith('.A_log'):
                value = -torch.exp(rounded.float())
                need(bool(torch.isfinite(value).all()) and bool((value < 0).all()),
                     'Native decay must be finite and strictly negative: ' + name)
                decay.append(value)
            elif name.endswith('.dt_bias'):
                value = F.softplus(rounded.float())
                need(bool(torch.isfinite(value).all()) and bool((value > 0).all()),
                     'Native dt-bias softplus diagnostic must be finite and positive: ' + name)
                bias_delta.append(value)
        result = dict(complete=True, all_masters_and_fp16_casts_finite=True,
                      native_A_finite=True, dt_bias_softplus_finite=True,
                      new_clamps_or_reparameterization=False,
                      scope='parameter diagnostics; token-dependent states/logits checked by trainer')
        for label, values in [('native_A', decay), ('softplus_dt_bias_only', bias_delta)]:
            if values:
                joined = torch.cat(values)
                result[label] = dict(minimum=float(joined.min()), maximum=float(joined.max()),
                                     zeros=int(torch.count_nonzero(joined == 0)))
        return result

    @torch.no_grad()
    def family_ranges(self, gradients=False):
        groups = {}
        for name, master in self.masters.items():
            value = master.grad if gradients else master
            need(value is not None, 'Missing retained gradient: ' + name)
            groups.setdefault(parameter_family(name), []).append(value.detach().flatten())
        result = {}
        for family, values in groups.items():
            joined = torch.cat(values); finite = bool(torch.isfinite(joined).all())
            result[family] = dict(tensors=len(values), parameters=joined.numel(), finite=finite,
                nonzero=int(torch.count_nonzero(joined)),
                minimum=float(joined.min()) if finite else None,
                maximum=float(joined.max()) if finite else None)
        return result

    @torch.no_grad()
    def export_parity(self, ids, retained=None):
        values = self.export_fp16() if retained is None else retained
        expected = self.forward(ids, use_checkpoint=False)
        expected_logits = self.model.lm_head(expected[:, -8:])
        try:
            with install_fp16(self.model, values, self.inventory):
                actual = self.model.backbone(ids)
                actual_logits = self.model.lm_head(actual[:, -8:])
                need(torch.equal(expected, actual) and torch.equal(expected_logits, actual_logits),
                     'Functional/exported native hidden or logits differ')
        finally:
            self.assert_frozen()
        return dict(complete=True, passed=True, bitwise_equal=True, hidden_bitwise_equal=True,
            last_eight_logits_bitwise_equal=True, input_tokens=ids.numel(),
            hidden_sha256=tensor_hash(expected), logits_sha256=tensor_hash(expected_logits),
            original_parameter_objects_restored=True)


def family_ranges(bank, gradients=False):
    return bank.family_ranges(gradients=gradients)


def export_parity(bank, values, ids):
    return bank.export_parity(ids, retained=values)
