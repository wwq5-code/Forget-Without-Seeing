"""Shared 20261006 research protocol for all four CLIP/SigLIP entry points.

Outputs are anchored to the ENTRY SCRIPT, never to cwd:
  <entry>/outputs/<entry_stem>_20261006.txt  (JSON, metrics + aggregate + status)
  <entry>/outputs/<entry_stem>_20261006/     (offline audit scores)

These files are OFFLINE RESEARCH AUDITS, not DP releases. They contain private
evaluation statistics. Only a SINGLE protected message is a DP mechanism;
releasing several messages requires composition. The guarantee conditions on
the already-released provider, fixed public pools/bases and fixed hyperparameters.
Receiver reconstruction has no dataset/forget-record arguments; the offline
attacker separately receives known candidate examples, never an oracle gradient
from the true member set. This is a code-level boundary, not an OS sandbox.
"""

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import sys
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from On_CIFAR100 import unlearning_clip_by_subliminal as core


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def digest_indices(indices):
    return hashlib.sha256(np.asarray(sorted(indices), dtype=np.int64).tobytes()).hexdigest()


def configure_research(args):
    if args.smoke:
        args.source_epochs = args.personal_epochs = 1
        args.max_source_steps = 1
        args.student_steps = 2
        args.seeds = [43]
        args.train_batch_size = args.eval_batch_size = 16
        args.private_batch_size = 8
        args.num_workers = 0
        args.channel_rank = 4
        args.carrier_count = 8
        args.message_rank = 4
        args.unlearning_sizes = [16, 32]
        args.dp_unlearning_size = 32
        args.epsilons = ["inf", "8", "2"]
        args.personal_train_size = 32
        args.public_pool_size = 32
        args.mia_nonmembers = 64
        args.noise_repeats = 2
        args.capacity_counts = [4, 8]
        args.capacity_ranks = [2, 4]
        args.audit_eval_limit = 64
    if not args.smoke and (len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds)):
        raise ValueError("Full research runs require at least three distinct model seeds")
    if min(args.unlearning_sizes + [args.dp_unlearning_size]) < 8:
        raise ValueError("MIA fit/calibration/test requires at least eight members")
    if args.mia_nonmembers < 16 or args.personal_train_size < 4 or args.public_pool_size < 4:
        raise ValueError("Insufficient disjoint audit/local/public examples")
    if args.noise_repeats < 1 or args.message_rank < 1:
        raise ValueError("Positive message rank and repeat count required")
    if args.dp_clip_norm <= 0 or not 0 < args.dp_delta < 1:
        raise ValueError("Invalid DP clipping norm/delta")
    if args.channel_rank < 1 or min([args.carrier_count] + args.capacity_counts) < args.channel_rank:
        raise ValueError("Every carrier count must be >= adapter channel rank")
    if any(r < 1 for r in args.capacity_ranks) or any(not 0 <= s <= 1 for s in args.receiver_strengths):
        raise ValueError("Invalid capacity rank or receiver interpolation strength")
    if args.personal_preservation_budget < 0 or args.student_steps < 0:
        raise ValueError("Invalid preservation budget/student steps")
    if min(args.source_epochs, args.personal_epochs, args.train_batch_size, args.eval_batch_size) < 1:
        raise ValueError("Positive epochs/batches required")
    if args.max_source_steps is not None and args.max_source_steps < 1:
        raise ValueError("max_source_steps must be positive")
    core.parse_epsilons(args.epsilons)
    if not args.run_tag or any(c not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-" for c in args.run_tag):
        raise ValueError("run-tag must be a simple filename component")


def make_splits(dataset, args, seed, forget_target, personal_target):
    """Fixed public reservation precedes the deletion request. All sets disjoint."""
    labels = np.asarray(dataset.targets)
    public_rng = np.random.default_rng(args.carrier_seed)
    eligible_public = np.flatnonzero((labels != forget_target) & (labels != personal_target))
    public = public_rng.permutation(eligible_public)[:args.public_pool_size].tolist()
    rng = np.random.default_rng(seed + 1900)
    available = np.ones(len(dataset), dtype=bool)
    available[public] = False
    local_pool = np.flatnonzero(available & (labels != personal_target))
    local = rng.permutation(local_pool)[:args.personal_train_size].tolist()
    available[local] = False
    candidates = rng.permutation(np.flatnonzero(available & (labels != forget_target)))
    n = max(args.unlearning_sizes + [args.dp_unlearning_size])
    if len(candidates) < n + args.mia_nonmembers:
        raise ValueError("Insufficient records for independent deletion/audit pools")
    private = candidates[:n].tolist()
    nonmembers = candidates[n:n + args.mia_nonmembers].tolist()
    groups = {"public_carriers": public, "local_retain": local,
              "private_forget": private, "attack_nonmembers": nonmembers}
    if len(public) != args.public_pool_size or len(local) != args.personal_train_size:
        raise ValueError("Insufficient public/local records")
    values = list(groups.values())
    if len(set().union(*(set(v) for v in values))) != sum(map(len, values)):
        raise AssertionError("Data-access partition overlap")
    return groups


@torch.no_grad()
def encode_records(backbone, dataset, indices, mean, std, args, device, kind=None):
    parts, labels = [], []
    for images, target in core.make_loader(Subset(dataset, indices), args.eval_batch_size,
                                           False, 0, args):
        images = images.to(device)
        if kind:
            images = core.apply_trigger(images, mean, std, args.trigger_size, args.trigger_margin, kind)
        parts.append(backbone(images).detach())
        labels.append(target.to(device))
    return torch.cat(parts), torch.cat(labels)


@torch.no_grad()
def cache_evaluation(backbone, dataset, mean, std, args, device):
    # Stratified subsampling is for smoke/diagnostic runs only, never paper metrics.
    if args.audit_eval_limit:
        indices = np.random.default_rng(12345).permutation(len(dataset))[:args.audit_eval_limit].tolist()
    else:
        indices = list(range(len(dataset)))
    clean, labels = encode_records(backbone, dataset, indices, mean, std, args, device)
    forget, _ = encode_records(backbone, dataset, indices, mean, std, args, device, "forget")
    personal, _ = encode_records(backbone, dataset, indices, mean, std, args, device, "personal")
    return {"clean": clean, "forget": forget, "personal": personal, "labels": labels}


@torch.no_grad()
def behavior(cache, projection, text, scale, forget_target, personal_target, reference=None):
    labels = cache["labels"]
    result = {"n_clean": len(labels)}
    for kind in ("clean", "forget", "personal"):
        logits = core.classification_logits(cache[kind], projection, text, scale)
        prediction = logits.argmax(-1)
        mask = torch.ones_like(labels, dtype=torch.bool) if kind == "clean" else labels.ne(
            forget_target if kind == "forget" else personal_target)
        if not int(mask.sum()):
            raise ValueError("Evaluation subset lacks eligible examples")
        if kind == "clean":
            result["clean_accuracy"] = float(prediction.eq(labels).float().mean())
        else:
            target = forget_target if kind == "forget" else personal_target
            result[kind + "_asr"] = float(prediction[mask].eq(target).float().mean())
            result[kind + "_trigger_true_accuracy"] = float(prediction[mask].eq(labels[mask]).float().mean())
        if reference is not None and kind in ("clean", "personal"):
            before = core.classification_logits(cache[kind], reference, text, scale)
            result[kind + "_heldout_kl"] = float(core.kl_to_reference(logits, before).clamp_min(0))
            result[kind + "_prediction_agreement"] = float(prediction.eq(before.argmax(-1)).float().mean())
    return result


@torch.no_grad()
def encoder_audit(model, backbone, projection, dataset, text, scale, args, device):
    images, labels = next(iter(core.make_loader(dataset, min(32, args.eval_batch_size), False, 0, args)))
    images = images.to(device)
    native = F.normalize(model.encode_image(images).float(), dim=-1)
    reconstructed = F.normalize(backbone(images) @ projection, dim=-1)
    error = float((native - reconstructed).abs().max())
    if not math.isfinite(error) or error > 2e-4:
        raise RuntimeError(f"Native encoder / adapter parity failed: max error {error}")
    native_prediction = (scale * native @ text.t()).argmax(-1)
    reconstructed_prediction = (scale * reconstructed @ text.t()).argmax(-1)
    return {"max_normalized_embedding_error": error,
            "native_adapter_prediction_agreement": float(native_prediction.eq(reconstructed_prediction).float().mean()),
            "preprocess": dict(open_clip.get_model_preprocess_cfg(model)),
            "class_count": len(text), "native_output_dim": native.shape[1],
            "sample_count": len(labels), "checkpoint": getattr(model, "_checkpoint_audit", "open_clip pretrained loader"),
            "scope": "Parity audit only; full pretrained clean accuracy is reported separately."}


@dataclass(frozen=True)
class ReceiverState:
    """Only public carriers/model state and disjoint local retain features."""
    base_projection: torch.Tensor
    input_basis: torch.Tensor
    coefficient: torch.Tensor
    carrier_features: torch.Tensor
    local_features: tuple
    text_features: torch.Tensor
    logit_scale: torch.Tensor

    def reconstruct(self, message, args, seed):
        return core.reconstruct_personalized_student(
            self.base_projection, self.input_basis, self.coefficient,
            self.carrier_features, self.carrier_features @ self.input_basis,
            message, self.local_features, self.text_features, self.logit_scale, args, seed)


@torch.no_grad()
def contributions(features, labels, base, basis, provider, carriers, text, scale, args):
    """Curator or attacker-with-known-candidate calculation; never a receiver input."""
    projection = core.compose_projection(base, basis, provider)
    carrier_basis = carriers @ basis
    preconditioner = core.make_public_preconditioner(carrier_basis, args.preconditioner_damping,
                                                    args.preconditioner_max_gain)
    carrier_projected = carriers @ projection
    parts = []
    for start in range(0, len(labels), args.private_batch_size):
        f = features[start:start + args.private_batch_size]
        target = labels[start:start + args.private_batch_size]
        projected = f @ projection
        norm = projected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        normalized = projected / norm
        probabilities = (scale * normalized @ text.t()).softmax(-1)
        gradient = scale * (probabilities - F.one_hot(target, len(text))) @ text
        gradient = (gradient - (gradient * normalized).sum(-1, keepdim=True) * normalized) / norm
        coefficient_gradient = torch.einsum("br,bd->brd", f @ basis, gradient)
        updates = -args.influence_scale * torch.einsum("rs,bsd->brd", preconditioner, coefficient_gradient)
        raw = torch.einsum("mr,brd->bmd", carrier_basis, updates)
        parts.append(core.normalized_embedding_jvp(carrier_projected, raw))
    return torch.cat(parts)


def private_message(record_contributions, sample_count, stage, output_basis, epsilon, args):
    """One Gaussian release with fixed denominator and add/remove-record sensitivity.

    Projection, clipping and noise are separate interventions. The public output
    basis is fixed independently of deletion membership. Secret fresh randomness
    is NOT derived from the published experiment seed or saved in audit output.
    """
    values = record_contributions[:sample_count]
    projected = stage != "full_raw"
    if projected:
        values = values @ output_basis
    norms = values.flatten(1).norm(dim=1)
    clipped = stage in ("clipped", "noisy")
    factors = (args.dp_clip_norm / norms.clamp_min(1e-12)).clamp(max=1) if clipped else torch.ones_like(norms)
    mean = (values * factors[:, None, None]).sum(0) / sample_count
    multiplier, achieved = core.calibrate_noise_multiplier(epsilon, args.dp_delta)
    if stage == "noisy":
        generator = torch.Generator(device=mean.device)
        generator.manual_seed(secrets.randbits(63))
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        noise *= multiplier * args.dp_clip_norm / sample_count
        released = mean + noise
    else:
        if math.isfinite(epsilon):
            raise ValueError("Finite epsilon requires the noisy stage")
        noise = torch.zeros_like(mean)
        released = mean
    decoded = released @ output_basis.t() if projected else released
    full_reference = record_contributions[:sample_count].mean(0)
    projected_reference = (full_reference @ output_basis @ output_basis.t()) if projected else full_reference
    lifted_mean = mean @ output_basis.t() if projected else mean
    reference_norm = full_reference.norm().clamp_min(1e-12)
    return decoded, released, {
        "formal_dp": stage == "noisy", "epsilon": core.epsilon_tag(epsilon),
        "achieved_epsilon": achieved if math.isfinite(achieved) else None,
        "delta": args.dp_delta if stage == "noisy" else None,
        "noise_multiplier": multiplier, "clip_norm": args.dp_clip_norm if clipped else None,
        "adjacency": "record_add_remove", "public_denominator": sample_count,
        "projection_applied": projected, "clipping_applied": clipped,
        "clipped_fraction_audit_only": float((factors < 1).float().mean()),
        "release_dimension": released.numel(), "noise_l2_audit_only": float(noise.norm()),
        "message_l2": float(released.norm()), "mean_signal_l2_audit_only": float(mean.norm()),
        "projection_distortion": float((projected_reference-full_reference).norm()/reference_norm),
        "clipping_distortion": float((lifted_mean-projected_reference).norm()/reference_norm),
        "noise_distortion": float((decoded-lifted_mean).norm()/reference_norm),
        "randomness": "fresh_unpublished_secret_seed" if stage == "noisy" else "none",
        "scope": "One release conditional on fixed public provider, pools, bases and hyperparameters; audits are not released.",
    }


def split_attack_indices(labels, seed):
    rng = np.random.default_rng(seed)
    partitions = [[], [], []]
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(labels == label))
        cut = len(indices) // 4
        if cut < 2:
            raise ValueError("Too few members/nonmembers for disjoint MIA fit/calibration/test")
        for result, block in zip(partitions, (indices[:cut], indices[cut:2 * cut], indices[2 * cut:])):
            result.extend(block.tolist())
    return tuple(np.asarray(part) for part in partitions)


def wilson(successes, count):
    if count == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / count
    center = (p + z*z/(2*count)) / (1 + z*z/count)
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return [max(0, center-radius), min(1, center+radius)]


def roc_auc_score(labels, scores):
    """Mann-Whitney AUC with exact average tie ranks, no sklearn dependency."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(scores)]
    ranks = np.empty(len(scores), dtype=np.float64)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[left:right]] = (left + right + 1) / 2
    positives = int((labels == 1).sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUC needs both classes")
    return (ranks[labels == 1].sum() - positives*(positives+1)/2) / (positives*negatives)


def fit_attack(features, labels, split):
    """Fit scores on fit only; fix direction/threshold on calibration, never test."""
    fit, calibration, test = split
    features = np.asarray(features, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite attack features")
    center = features[fit].mean(0)
    spread = features[fit].std(0)
    spread[spread < 1e-10] = 1
    design = np.c_[(features - center) / spread, np.ones(len(features))]
    y_fit = labels[fit]
    weight = np.where(y_fit == 1, len(fit)/(2*(y_fit == 1).sum()), len(fit)/(2*(y_fit == 0).sum()))
    regularizer = np.eye(design.shape[1]); regularizer[-1, -1] = 0
    coefficient = np.linalg.solve(design[fit].T @ (weight[:, None]*design[fit]) + regularizer,
                                  design[fit].T @ (weight*(2*y_fit-1)))
    scores = design @ coefficient
    if roc_auc_score(labels[calibration], scores[calibration]) < .5:
        scores = -scores
        direction = -1
    else:
        direction = 1
    calibration_negatives = scores[calibration][labels[calibration] == 0]
    threshold = float(np.quantile(calibration_negatives, .99, method="higher"))
    y = labels[test]
    predicted = scores[test] > threshold
    positives, negatives = int((y == 1).sum()), int((y == 0).sum())
    tp, fp = int(predicted[y == 1].sum()), int(predicted[y == 0].sum())
    return {
        "auc": float(roc_auc_score(y, scores[test])),
        "tpr_at_calibrated_1pct_fpr": tp/positives,
        "achieved_test_fpr": fp/negatives,
        "tpr_wilson95": wilson(tp, positives), "fpr_wilson95": wilson(fp, negatives),
        "n_test_members": positives, "n_test_nonmembers": negatives,
        "n_calibration_nonmembers": len(calibration_negatives),
        "low_fpr_resolution_sufficient": min(negatives, len(calibration_negatives)) >= 1000,
        "threshold": threshold, "calibration_selected_direction": direction,
        "estimator": "balanced_standardized_ridge_discriminator_lambda1_fit_only",
        "warning": "Conditional on one deletion request; noise repeats are not independent populations. AUC is not flipped/selected on test.",
    }, scores


@torch.no_grad()
def candidate_features(features, labels, projection, text, scale, target):
    logits = core.classification_logits(features, projection, text, scale)
    logp = logits.log_softmax(-1)
    return torch.stack((-logp.gather(1, labels[:, None]).squeeze(1), -logp[:, target],
                        logits.softmax(-1).max(-1).values), dim=1)


def audit_mia(candidate_visual, candidate_labels, candidate_contributions, private_count,
              member_count, provider_projection, before_projection, after_projection,
              released, output_basis, privacy, text, scale, target, seed, score_path):
    """Known-candidate attacks on provider/receiver, channel, and post-update receiver."""
    keep = torch.cat((torch.arange(member_count, device=candidate_visual.device),
                      torch.arange(private_count, len(candidate_visual), device=candidate_visual.device)))
    visual, targets = candidate_visual[keep], candidate_labels[keep]
    provider = candidate_features(visual, targets, provider_projection, text, scale, target)
    before = candidate_features(visual, targets, before_projection, text, scale, target)
    after = candidate_features(visual, targets, after_projection, text, scale, target)
    baseline = torch.cat((provider, before), dim=1)
    fingerprints = candidate_contributions[keep]
    if privacy["projection_applied"]:
        fingerprints = fingerprints @ output_basis
    norms = fingerprints.flatten(1).norm(dim=1)
    if privacy["clipping_applied"]:
        fingerprints = fingerprints * (privacy["clip_norm"] / norms.clamp_min(1e-12)).clamp(max=1)[:, None, None]
    flat = fingerprints.flatten(1)
    dot = flat @ released.flatten()
    cosine = dot / (flat.norm(dim=1) * released.norm()).clamp_min(1e-12)
    channel = torch.stack((dot, cosine, norms), dim=1)
    variants = {"M0_only": baseline, "M0_plus_message": torch.cat((baseline, channel), dim=1),
                "M0_plus_student": torch.cat((baseline, after, after-before), dim=1)}
    y = np.r_[np.ones(member_count, dtype=np.int64), np.zeros(len(visual)-member_count, dtype=np.int64)]
    split = split_attack_indices(y, seed + 85000)
    results, scores = {}, {"membership_audit_only": y, "fit_indices": split[0],
                            "calibration_indices": split[1], "test_indices": split[2]}
    for name, values in variants.items():
        results[name], scores[name] = fit_attack(values.detach().cpu().numpy(), y, split)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(score_path, **scores)
    return {"target": "membership_in_deletion_request_not_pretraining_membership",
            "candidate_distribution": "same training pool, excluded from public/local pools; all candidates were eligible for request selection",
            "candidate_access": "known images and original labels; no true-member aggregate gradients",
            "fit_calibration_test_disjoint": True, "attacks": results,
            "attack_selection": "All three prespecified attacks reported; no best-test-AUC selection",
            "score_file_audit_only": str(score_path)}


METRIC_DEFINITIONS = {
    "forget_asr": {"unit": "fraction", "direction": "lower", "definition": "Target prediction rate on held-out forget-trigger images, excluding target-class images."},
    "forget_asr_reduction": {"unit": "fraction_difference", "direction": "higher", "definition": "Matched receiver-before ASR minus student ASR; multiply by 100 for percentage points."},
    "clean_accuracy": {"unit": "fraction", "direction": "higher", "definition": "Top-1 accuracy on the untouched held-out split."},
    "personal_asr": {"unit": "fraction", "direction": "compare_with_before", "definition": "Personal-trigger target success; larger alone does not prove behavior preservation."},
    "personal_asr_change": {"unit": "fraction_difference", "direction": "near_zero_for_fidelity", "definition": "Student minus matched receiver-before personal ASR."},
    "local_preservation_kl": {"unit": "nats", "direction": "lower", "definition": "Sum of clean/personal KL(reference||student) on disjoint LOCAL retain anchors; hard constraint applies here only."},
    "heldout_preservation_kl": {"unit": "nats", "direction": "lower", "definition": "Sum of clean/personal KL on held-out evaluation images; not constrained or used for selection."},
    "relative_carrier_error": {"unit": "relative_L2", "direction": "lower", "definition": "Student JVP minus received decoded message, divided by message norm; not teacher-normalized forgetting."},
    "projection_distortion": {"unit": "relative_L2", "direction": "lower", "definition": "Projection-only message minus full raw message, normalized by full raw message norm; offline audit only."},
    "clipping_distortion": {"unit": "relative_L2", "direction": "lower", "definition": "Clipped message minus projected raw message, normalized by full raw message norm; offline audit only."},
    "noise_distortion": {"unit": "relative_L2", "direction": "lower", "definition": "Noisy message minus clipped message, normalized by full raw message norm; offline audit only."},
    "student_to_unprotected_signal_relative_l2": {"unit": "relative_L2", "direction": "lower", "definition": "Receiver JVP minus curator full raw functional target on the carrier bank, normalized by full raw target norm; not a teacher forgetting ratio."},
    "auc": {"unit": "fraction", "direction": "closer_to_0.5_for_lower_measured_leakage", "definition": "Held-out deletion-request membership ROC AUC; direction fixed using calibration, not test."},
    "tpr_at_calibrated_1pct_fpr": {"unit": "fraction", "direction": "lower_leakage", "definition": "Test TPR at a threshold calibrated for <=1% empirical calibration FPR; always read achieved_test_fpr and confidence intervals."},
    "achieved_test_fpr": {"unit": "fraction", "direction": "report", "definition": "Actual test FPR at the fixed calibration threshold; not guaranteed to equal 1%."},
    "curator_contribution_seconds": {"unit": "seconds", "direction": "lower", "definition": "Per-arm functional contribution construction from cached frozen forget features."},
    "message_release_seconds": {"unit": "seconds", "direction": "lower", "definition": "Per-arm public projection, optional per-record clipping, averaging, noise and lifting."},
    "student_unlearning_seconds": {"unit": "seconds", "direction": "lower", "definition": "Receiver reconstruction including local hard-budget feasibility checks."},
    "running_time_seconds": {"unit": "seconds", "direction": "lower", "definition": "Sum of curator_contribution_seconds + message_release_seconds + student_unlearning_seconds, CUDA-synchronized. Excludes shared frozen-feature extraction, pre-release training and offline audits; NOT a cold-start end-to-end time."},
    "evaluation_seconds": {"unit": "seconds", "direction": "report", "definition": "Per-arm held-out behavior evaluation from cached features."},
    "mia_seconds": {"unit": "seconds", "direction": "report", "definition": "Offline attack feature calculation, fit/calibration/test, and score saving; excluded from unlearning runtime."},
    "total_wall_time_seconds": {"unit": "seconds", "direction": "report", "definition": "Whole script elapsed time including model loading, all training, feature extraction, all arms and audits."},
    "seed_mean_sample_std": {"unit": "same_as_metric", "direction": "report", "definition": "Average noise repeats within each model seed first, then mean and sample SD (ddof=1) across independent model seeds. Single-seed SD is null."},
}
for _attack_name in ("M0_only", "M0_plus_message", "M0_plus_student"):
    for _metric_name in ("auc", "tpr_at_calibrated_1pct_fpr", "achieved_test_fpr"):
        METRIC_DEFINITIONS[f"{_attack_name}.{_metric_name}"] = {
            **METRIC_DEFINITIONS[_metric_name], "attacker_view": _attack_name,
        }


def tick(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def aggregate_metrics(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["setting_id"], []).append(row)
    output = []
    for setting, observations in groups.items():
        metrics = sorted({k for r in observations for k in r["metrics"]})
        summary = {}
        for metric in metrics:
            per_seed = {}
            for row in observations:
                value = row["metrics"].get(metric)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    per_seed.setdefault(row["seed"], []).append(value)
            if not per_seed:
                continue
            means = [statistics.mean(v) for v in per_seed.values()]
            summary[metric] = {"mean": statistics.mean(means),
                               "sample_std": statistics.stdev(means) if len(means) > 1 else None,
                               "n_model_seeds": len(means), "n_observations": sum(map(len, per_seed.values())),
                               "noise_repeat_std_by_seed": {str(k): statistics.stdev(v) if len(v) > 1 else None
                                                             for k, v in per_seed.items()}}
        output.append({"setting_id": setting, "configuration": observations[0]["configuration"], "metrics": summary})
    return output


def research_plan(args):
    baseline = ("spectral", args.carrier_count, args.message_rank)
    configs = {baseline}
    configs.update((kind, args.carrier_count, args.message_rank) for kind in args.carrier_types)
    configs.update(("spectral", count, args.message_rank) for count in args.capacity_counts)
    configs.update(("spectral", args.carrier_count, rank) for rank in args.capacity_ranks)
    plan = []
    for config in sorted(configs):
        points = []
        if config == baseline:
            points.extend(("size", n, "clipped", math.inf, 0, 1.0, args.preservation_mode)
                          for n in args.unlearning_sizes)
            points.extend(("protection", args.dp_unlearning_size, stage, math.inf, 0, 1.0, args.preservation_mode)
                          for stage in ("full_raw", "projected_raw", "clipped"))
            points.extend(("protection", args.dp_unlearning_size, "noisy", eps, draw, 1.0, args.preservation_mode)
                          for eps in core.parse_epsilons(args.epsilons) if math.isfinite(eps)
                          for draw in range(args.noise_repeats))
            points.extend(("receiver", args.dp_unlearning_size, "clipped", math.inf, 0, strength, args.preservation_mode)
                          for strength in args.receiver_strengths if strength != 1)
            # Report the cost of enforcing the budget; never silently call a penalty a constraint.
            alternate = "soft" if args.preservation_mode == "hard" else "hard"
            points.append(("preservation", args.dp_unlearning_size, "clipped", math.inf, 0, 1.0, alternate))
        else:
            points.append(("capacity", args.dp_unlearning_size, "clipped", math.inf, 0, 1.0, args.preservation_mode))
            points.extend(("capacity", args.dp_unlearning_size, "noisy", 8.0, draw, 1.0, args.preservation_mode)
                          for draw in range(args.noise_repeats))
        plan.append((config, points))
    return plan


@torch.no_grad()
def carrier_bank(kind, count, backbone, public_features, image_size, mean, std, args, device):
    if kind == "public_images":
        if count > len(public_features):
            raise ValueError("Public image carrier count exceeds reserved public pool")
        return public_features[:count]
    sampler = core.SpectralCarrierSampler(count, image_size, 1.0 if kind == "spectral" else 0.0,
                                         mean, std, device, args.carrier_seed)
    images = sampler.sample()
    # Avoid allocating a huge activation batch for frozen SigLIP.
    return torch.cat([backbone(batch) for batch in images.split(min(64, args.eval_batch_size))])


def run_research(args, entry_file, dataset_name, family):
    configure_research(args)
    device = core.select_device(args.gpu, args.device)
    args.device = str(device)
    torch.set_float32_matmul_precision("high")
    entry = Path(entry_file).resolve()
    stem = entry.stem + ("_smoke" if args.smoke else "") + "_" + args.run_tag
    result_path = entry.parent / "outputs" / (stem + ".txt")
    artifacts = entry.parent / "outputs" / stem
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {result_path}; use a new run-tag")
    started = tick(device)
    payload = {
        "schema_version": "functional_research_20261006_v1", "status": "running",
        "dataset": dataset_name, "model_family": family, "model": args.model_name,
        "smoke_test_only": args.smoke, "evaluation_subset_diagnostic_only": bool(args.audit_eval_limit),
        "created_utc": datetime.now(timezone.utc).isoformat(), "args": vars(args),
        "result_path": str(result_path), "metric_definitions": METRIC_DEFINITIONS,
        "environment": {"python": sys.version, "python_executable": sys.executable,
                        "torch": torch.__version__, "open_clip": open_clip.__version__,
                        "numpy": np.__version__, "device": str(device),
                        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "source_sha256": {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in {entry, Path(__file__).resolve(), Path(core.__file__).resolve()}},
        "claim_scope": "Post-release association suppression through a functional message; not proof of complete knowledge erasure or cross-architecture sender-to-receiver transfer.",
        "release_scope": "All files here are offline audits. Private audit metrics/scores are not DP releases. Each epsilon arm is a separate hypothetical release; composition is not included.",
        "receiver_boundary": "Only fixed model/bases, public carriers, received message and disjoint local retain features. No forget or attack examples in reconstruction. Code-level, not OS isolation.",
        "runtime_accounting": {
            "running_time_seconds": "Per-arm cached-feature online unlearning; CUDA synchronization at each boundary.",
            "shared_costs": "Frozen feature extraction, carrier bank and candidate fingerprints are recorded separately; never sum duplicated per-row shared costs.",
            "pre_release_costs": "Provider/backdoor and receiver personalization training are separate seed-audit fields, not unlearning runtime.",
            "cold_start_total": "total_wall_time_seconds includes everything for the complete multi-arm run, not one request.",
        },
        "rows": [], "seed_audits": [], "aggregate_over_model_seeds": [],
    }
    atomic_json(result_path, payload)
    print(f"RESULT_PATH={result_path}", flush=True)
    try:
        model, preprocess_train, preprocess_val, pretrained = core.create_clip_and_transforms(args)
        payload["pretrained"] = pretrained
        model.to(device).eval()
        preprocessing = dict(open_clip.get_model_preprocess_cfg(model))
        image_size = core.as_image_size(preprocessing.get("size", getattr(model.visual, "image_size", 224)))
        mean = tuple(preprocessing.get("mean", open_clip.OPENAI_DATASET_MEAN))
        std = tuple(preprocessing.get("std", open_clip.OPENAI_DATASET_STD))
        if dataset_name == "tinyimagenet":
            from On_TinyImageNet import unlearning_clip_by_subliminal as tiny
            train_aug = tiny.make_dataset(args, True, preprocess_train)
            train_eval = tiny.make_dataset(args, True, preprocess_val)
            test_dataset = tiny.make_dataset(args, False, preprocess_val)
            resolve_target = tiny.resolve_target_label
        else:
            train_aug = CIFAR100(args.data_root, train=True, transform=preprocess_train, download=args.download)
            train_eval = CIFAR100(args.data_root, train=True, transform=preprocess_val)
            test_dataset = CIFAR100(args.data_root, train=False, transform=preprocess_val)
            resolve_target = core.resolve_target_label
        forget_target = resolve_target(train_aug, args.forget_target)
        personal_target = resolve_target(train_aug, args.personal_target)
        if forget_target == personal_target:
            raise ValueError("Forget/personal targets must differ")
        tokenizer = core.load_tokenizer(args)
        text = core.build_text_features(model, tokenizer, train_aug.classes, args.prompt_template, device)
        scale = model.logit_scale.exp().detach().float()
        if family == "siglip":
            from On_CIFAR100 import unlearning_SigLIP_by_subliminal as siglip
            backbone = siglip.FrozenSigLIPBackbone(model.visual, min(64, args.eval_batch_size)).to(device)
            probe = next(iter(core.make_loader(test_dataset, 2, False, 0, args)))[0].to(device)
            with torch.no_grad():
                base = siglip.initialize_adapter(model.visual, backbone(probe), text, device)
        else:
            backbone = core.FrozenVisionBackbone(model.visual).to(device)
            base = model.visual.proj.detach().float().clone()
        payload["native_encoder_audit"] = encoder_audit(model, backbone, base, test_dataset, text, scale, args, device)
        payload["class_mapping_audit"] = {"class_count": len(train_aug.classes),
                                         "forget_index": forget_target, "personal_index": personal_target,
                                         "forget_name": train_aug.classes[forget_target],
                                         "personal_name": train_aug.classes[personal_target],
                                         "prompt_template": args.prompt_template}
        del model
        basis = core.make_input_basis(base.shape[0], args.channel_rank, device, args.channel_seed)
        if max(args.capacity_ranks + [args.message_rank]) > min(args.public_pool_size, base.shape[1]):
            raise ValueError("Message rank exceeds fixed public basis dimensions")
        before_eval = tick(device)
        eval_cache = cache_evaluation(backbone, test_dataset, mean, std, args, device)
        payload["shared_evaluation_feature_extraction_seconds"] = tick(device) - before_eval
        payload["pretrained_baseline"] = behavior(eval_cache, base, text, scale, forget_target, personal_target)
        payload["pretrained_baseline"]["above_twice_chance"] = payload["pretrained_baseline"]["clean_accuracy"] > 2/len(text)
        plan = research_plan(args)
        payload["expected_rows"] = len(args.seeds) * sum(len(points) for _, points in plan)
        atomic_json(result_path, payload)
        for seed in args.seeds:
            core.set_seed(seed)
            seed_start = tick(device)
            print(f"seed={seed} pre-release training", flush=True)
            split = make_splits(train_eval, args, seed, forget_target, personal_target)
            provider_start = tick(device)
            provider = core.train_provider_backdoor(
                backbone, base, basis, core.make_loader(train_aug, args.train_batch_size, True, seed, args),
                text, scale, forget_target, mean, std, args, device)
            provider_seconds = tick(device) - provider_start
            local_start = tick(device)
            local = core.collect_local_features(backbone, train_eval, split["local_retain"], mean, std, args, device, seed)
            local_seconds = tick(device) - local_start
            personal_start = tick(device)
            personalized = core.train_personalized_model(base, basis, provider, local, text, scale, personal_target, args, seed)
            personal_seconds = tick(device) - personal_start
            feature_start = tick(device)
            public_features, _ = encode_records(backbone, train_eval, split["public_carriers"], mean, std, args, device)
            private_features, private_labels = encode_records(backbone, train_eval, split["private_forget"], mean, std, args, device, "forget")
            curator_feature_seconds = tick(device) - feature_start
            attacker_start = tick(device)
            non_features, non_labels = encode_records(backbone, train_eval, split["attack_nonmembers"], mean, std, args, device, "forget")
            attacker_feature_seconds = tick(device) - attacker_start
            candidate_visual = torch.cat((private_features, non_features))
            candidate_labels = torch.cat((private_labels, non_labels))
            provider_projection = core.compose_projection(base, basis, provider)
            personalized_projection = core.compose_projection(base, basis, personalized)
            # Public basis fixed for the entire seed, not changed with carrier count/type or request.
            with torch.no_grad():
                _, _, vh = torch.linalg.svd(F.normalize(public_features @ provider_projection, dim=-1), full_matrices=False)
                public_output_basis = vh.t().contiguous()
            audit = {"seed": seed, "partitions": {k: {"count": len(v), "sha256": digest_indices(v)} for k, v in split.items()},
                     "partition_overlap_count": 0, "pre_release_provider_training_seconds": provider_seconds,
                     "pre_release_personalization_seconds": personal_seconds,
                     "local_feature_extraction_seconds": local_seconds,
                     "shared_curator_feature_extraction_seconds": curator_feature_seconds,
                     "shared_attacker_feature_extraction_seconds": attacker_feature_seconds,
                     "provider": behavior(eval_cache, provider_projection, text, scale, forget_target, personal_target),
                     "personalized": behavior(eval_cache, personalized_projection, text, scale, forget_target, personal_target),
                     "projection_basis_source": "SVD of fixed public image embeddings at the already released provider"}
            payload["seed_audits"].append(audit)
            for (kind, count, rank), points in plan:
                bank_start = tick(device)
                carriers = carrier_bank(kind, count, backbone, public_features, image_size, mean, std, args, device)
                bank_seconds = tick(device) - bank_start
                if int(torch.linalg.matrix_rank(carriers @ basis)) < args.channel_rank:
                    raise RuntimeError(f"Carrier bank rank deficient: {kind}, {count}")
                output_basis = public_output_basis[:, :rank]
                fingerprint_start = tick(device)
                # Attacker computes candidate fingerprints from known candidate records, not actual-member aggregates.
                fingerprints = contributions(candidate_visual, candidate_labels, base, basis, provider,
                                             carriers, text, scale, args)
                fingerprint_seconds = tick(device) - fingerprint_start
                for sweep, n, stage, epsilon, draw, strength, preservation_mode in points:
                    row_start = tick(device)
                    coefficient = provider + strength * (personalized - provider)
                    before_projection = core.compose_projection(base, basis, coefficient)
                    reconstruction_args = copy.copy(args)
                    reconstruction_args.preservation_mode = preservation_mode
                    configuration = {"sweep": sweep, "carrier_type": kind, "carrier_count": count,
                                     "message_rank": rank, "adapter_rank": args.channel_rank,
                                     "unlearning_size": n, "stage": stage, "epsilon": core.epsilon_tag(epsilon),
                                     "receiver_strength": strength, "preservation_mode": preservation_mode}
                    setting = f"{sweep}_{kind}_c{count}_r{rank}_n{n}_{stage}_eps{core.epsilon_tag(epsilon)}_a{strength}_{preservation_mode}"
                    construct_start = tick(device)
                    raw = contributions(private_features[:n], private_labels[:n], base, basis, provider, carriers, text, scale, args)
                    construct_seconds = tick(device) - construct_start
                    release_start = tick(device)
                    decoded, released, privacy = private_message(raw, n, stage, output_basis, epsilon, args)
                    release_seconds = tick(device) - release_start
                    receiver = ReceiverState(base, basis, coefficient, carriers, local, text, scale)
                    reconstruction_start = tick(device)
                    student, reconstruction = receiver.reconstruct(decoded, reconstruction_args, seed)
                    reconstruction_seconds = tick(device) - reconstruction_start
                    if preservation_mode == "hard" and reconstruction["preservation_violation"] > 2e-7:
                        raise AssertionError("Hard local preservation budget violated")
                    after_projection = core.compose_projection(base, basis, student)
                    evaluation_start = tick(device)
                    before = behavior(eval_cache, before_projection, text, scale, forget_target, personal_target)
                    after = behavior(eval_cache, after_projection, text, scale, forget_target, personal_target, before_projection)
                    evaluation_seconds = tick(device) - evaluation_start
                    attack_start = tick(device)
                    mia = audit_mia(candidate_visual, candidate_labels, fingerprints, len(private_labels), n,
                                    provider_projection, before_projection, after_projection, released, output_basis,
                                    privacy, text, scale, forget_target, seed,
                                    artifacts / "audit_only_scores" / f"seed{seed}_{setting}_draw{draw}.npz")
                    attack_seconds = tick(device) - attack_start
                    with torch.no_grad():
                        fitted_signal = core.normalized_embedding_jvp(carriers @ before_projection,
                                                                     (carriers @ basis) @ (student-coefficient))
                        full_signal = raw.mean(0)
                        fidelity = float((fitted_signal-full_signal).norm()/full_signal.norm().clamp_min(1e-12))
                    metrics = {"forget_asr": after["forget_asr"], "clean_accuracy": after["clean_accuracy"],
                               "personal_asr": after["personal_asr"],
                               "forget_asr_reduction": before["forget_asr"] - after["forget_asr"],
                               "personal_asr_change": after["personal_asr"] - before["personal_asr"],
                               "local_preservation_kl": reconstruction["personal_preservation_loss"],
                               "heldout_preservation_kl": after["clean_heldout_kl"] + after["personal_heldout_kl"],
                               "relative_carrier_error": reconstruction["relative_carrier_error"],
                               "projection_distortion": privacy["projection_distortion"],
                               "clipping_distortion": privacy["clipping_distortion"],
                               "noise_distortion": privacy["noise_distortion"],
                               "student_to_unprotected_signal_relative_l2": fidelity,
                               "curator_contribution_seconds": construct_seconds,
                               "message_release_seconds": release_seconds,
                               "student_unlearning_seconds": reconstruction_seconds,
                               "running_time_seconds": construct_seconds + release_seconds + reconstruction_seconds,
                               "evaluation_seconds": evaluation_seconds, "mia_seconds": attack_seconds}
                    for attack_name, values in mia["attacks"].items():
                        for metric in ("auc", "tpr_at_calibrated_1pct_fpr", "achieved_test_fpr"):
                            metrics[f"{attack_name}.{metric}"] = values[metric]
                    if set(metrics) - set(METRIC_DEFINITIONS):
                        raise AssertionError("Every summarized metric must have an explicit definition")
                    payload["rows"].append({"seed": seed, "noise_draw": draw, "setting_id": setting,
                                            "configuration": configuration, "before": before, "student": after,
                                            "privacy": privacy, "reconstruction": reconstruction, "mia": mia,
                                            "metrics": metrics, "row_wall_time_seconds": tick(device)-row_start,
                                            "shared_carrier_bank_seconds": bank_seconds,
                                            "shared_fingerprint_seconds_audit_only": fingerprint_seconds})
                    payload["aggregate_over_model_seeds"] = aggregate_metrics(payload["rows"])
                    payload["total_wall_time_seconds"] = tick(device) - started
                    atomic_json(result_path, payload)
                    print(f"seed={seed} {setting} draw={draw} F-ASR={after['forget_asr']:.4f} "
                          f"clean={after['clean_accuracy']:.4f} localKL={metrics['local_preservation_kl']:.5f} "
                          f"running_time_seconds={metrics['running_time_seconds']:.3f}", flush=True)
                    del raw
                del fingerprints
            audit["total_seed_wall_time_seconds"] = tick(device) - seed_start
        if len(payload["rows"]) != payload["expected_rows"]:
            raise AssertionError("Incomplete research grid")
        payload["status"] = "complete"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        payload["total_wall_time_seconds"] = tick(device) - started
        atomic_json(result_path, payload)
        print(f"COMPLETE {result_path} rows={len(payload['rows'])} wall={payload['total_wall_time_seconds']:.1f}s", flush=True)
    except BaseException as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["total_wall_time_seconds"] = time.perf_counter() - started
        atomic_json(result_path, payload)
        raise
