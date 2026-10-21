"""Audited ClipErase/Zero-Shot comparisons for the corrected 20261006 protocol.

Every result is anchored beside its entry script:
  <entry>/outputs/<entry_stem>_comparison_20261006.txt
  <entry>/outputs/<entry_stem>_comparison_20261006/audit_only_scores/*.npz

The four dataset/model entries use the same corrected pretrained loader,
provider/personalized construction, seeds and disjoint data partitions as
functional_research_audit.py.  Full runs cross-check those partitions and
starting behaviors against the already completed research result when present.

ClipErase directly receives private forget records.  Zero-Shot receives the
target name and generated synthetic samples, but no private record.  These
access statements are not differential-privacy claims.  MIA and held-out
metrics are offline audits and must not be treated as deployable releases.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100

try:
    from . import functional_research_audit as audit
    from . import unlearning_clip_by_subliminal as core
except ImportError:
    import functional_research_audit as audit
    import unlearning_clip_by_subliminal as core


REPO_ROOT = Path(__file__).resolve().parents[1]


METRIC_DEFINITIONS = {
    key: value
    for key, value in audit.METRIC_DEFINITIONS.items()
    if key
    in {
        "forget_asr",
        "forget_asr_reduction",
        "clean_accuracy",
        "personal_asr",
        "personal_asr_change",
        "local_preservation_kl",
        "heldout_preservation_kl",
        "evaluation_seconds",
        "mia_seconds",
        "total_wall_time_seconds",
        "seed_mean_sample_std",
        "M0_only.auc",
        "M0_only.tpr_at_calibrated_1pct_fpr",
        "M0_only.achieved_test_fpr",
        "M0_plus_student.auc",
        "M0_plus_student.tpr_at_calibrated_1pct_fpr",
        "M0_plus_student.achieved_test_fpr",
    }
}
METRIC_DEFINITIONS.update(
    {
        "interpolation_scale": {
            "unit": "fraction",
            "direction": "report",
            "definition": (
                "Linear parameter interpolation from the matched personalized "
                "checkpoint (0) to the fully trained comparison model (1)."
            ),
        },
        "base_training_seconds": {
            "unit": "seconds",
            "direction": "lower",
            "definition": (
                "CUDA-synchronized comparison-method optimization. For Zero-Shot "
                "this includes synthetic generation and Lipschitz optimization; "
                "for ClipErase it includes direct forget/retain optimization."
            ),
        },
        "constraint_selection_seconds": {
            "unit": "seconds",
            "direction": "lower",
            "definition": (
                "CUDA-synchronized local-anchor evaluations used to find the "
                "largest interpolation scale satisfying the KL budget. This is a "
                "shared per-trained-model cost and must not be summed across rows."
            ),
        },
        "running_time_seconds": {
            "unit": "seconds",
            "direction": "lower",
            "definition": (
                "For hard_feasible rows: base training plus KL-boundary selection; "
                "for fixed Pareto rows: base training only. Excludes checkpoint "
                "construction, held-out evaluation, MIA and result serialization."
            ),
        },
    }
)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def tick(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def configure(args):
    if args.comparison_methods is None:
        args.comparison_methods = list(core.DEFAULT_COMPARISON_METHODS)
    audit.configure_research(args)
    if args.smoke:
        args.cliperase_steps = 2
        args.zero_shot_synthetic_count = 2
        args.zero_shot_generation_batch_size = 2
        args.zero_shot_generation_steps = 2
        args.zero_shot_steps = 2
        args.zero_shot_noise_samples = 1
        args.zero_shot_topk_visual = 2
        args.zero_shot_topk_text = 2
        args.comparison_hard_search_steps = 2
        args.comparison_pareto_scales = [0.0, 0.5, 1.0]
    if not args.comparison_methods:
        raise ValueError("comparison_research requires at least one method")
    scales = sorted(set(float(value) for value in args.comparison_pareto_scales))
    if not scales or scales[0] < 0 or scales[-1] > 1:
        raise ValueError("comparison Pareto scales must lie in [0, 1]")
    if 0.0 not in scales or 1.0 not in scales:
        raise ValueError("comparison Pareto scales must include both 0 and 1")
    if args.comparison_hard_search_steps < 1:
        raise ValueError("comparison-hard-search-steps must be positive")
    if args.dp_unlearning_size not in args.unlearning_sizes:
        raise ValueError("The Pareto anchor size must be included in unlearning_sizes")
    args.comparison_pareto_scales = scales


def evaluation_indices(dataset, limit):
    if limit:
        return np.random.default_rng(12345).permutation(len(dataset))[:limit].tolist()
    return list(range(len(dataset)))


@torch.no_grad()
def current_text(model, tokenizer, class_names, args, device):
    return core.build_text_features(
        model, tokenizer, class_names, args.prompt_template, device
    )


@torch.no_grad()
def logits_for_images(model, images, text_features):
    features = F.normalize(model.encode_image(images).float(), dim=-1)
    return model.logit_scale.exp().float() * (features @ text_features.t())


@torch.no_grad()
def reference_logits(
    model, tokenizer, class_names, dataset, indices, mean, std, args, device
):
    model.eval()
    text = current_text(model, tokenizer, class_names, args, device)
    clean_parts, personal_parts = [], []
    loader = core.make_loader(
        Subset(dataset, indices), args.eval_batch_size, False, 0, args
    )
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        personal = core.apply_trigger(
            images,
            mean,
            std,
            args.trigger_size,
            args.trigger_margin,
            "personal",
        )
        clean_parts.append(logits_for_images(model, images, text).cpu())
        personal_parts.append(logits_for_images(model, personal, text).cpu())
    return {
        "clean": torch.cat(clean_parts),
        "personal": torch.cat(personal_parts),
    }


@torch.no_grad()
def preservation_kl(
    model,
    tokenizer,
    class_names,
    dataset,
    indices,
    references,
    mean,
    std,
    args,
    device,
):
    model.eval()
    text = current_text(model, tokenizer, class_names, args, device)
    clean_sum = 0.0
    personal_sum = 0.0
    offset = 0
    loader = core.make_loader(
        Subset(dataset, indices), args.eval_batch_size, False, 0, args
    )
    for images, _ in loader:
        count = images.shape[0]
        images = images.to(device, non_blocking=True)
        personal = core.apply_trigger(
            images,
            mean,
            std,
            args.trigger_size,
            args.trigger_margin,
            "personal",
        )
        clean_logits = logits_for_images(model, images, text)
        personal_logits = logits_for_images(model, personal, text)
        clean_reference = references["clean"][offset : offset + count].to(device)
        personal_reference = references["personal"][offset : offset + count].to(device)
        clean_sum += float(
            F.kl_div(
                F.log_softmax(clean_logits, dim=-1),
                F.softmax(clean_reference, dim=-1),
                reduction="sum",
            ).cpu()
        )
        personal_sum += float(
            F.kl_div(
                F.log_softmax(personal_logits, dim=-1),
                F.softmax(personal_reference, dim=-1),
                reduction="sum",
            ).cpu()
        )
        offset += count
    if offset != len(indices):
        raise AssertionError("Preservation evaluation did not cover every index")
    return (clean_sum + personal_sum) / max(offset, 1)


@torch.no_grad()
def evaluate_behavior_and_kl(
    model,
    tokenizer,
    class_names,
    dataset,
    indices,
    references,
    forget_target,
    personal_target,
    mean,
    std,
    args,
    device,
):
    model.eval()
    text = current_text(model, tokenizer, class_names, args, device)
    totals = {
        "clean_correct": 0,
        "forget_success": 0,
        "forget_true_correct": 0,
        "forget_n": 0,
        "personal_success": 0,
        "personal_true_correct": 0,
        "personal_n": 0,
        "n": 0,
    }
    clean_kl_sum = 0.0
    personal_kl_sum = 0.0
    offset = 0
    loader = core.make_loader(
        Subset(dataset, indices), args.eval_batch_size, False, 0, args
    )
    for images, labels in loader:
        count = images.shape[0]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        forget = core.apply_trigger(
            images, mean, std, args.trigger_size, args.trigger_margin, "forget"
        )
        personal = core.apply_trigger(
            images, mean, std, args.trigger_size, args.trigger_margin, "personal"
        )
        clean_logits = logits_for_images(model, images, text)
        forget_logits = logits_for_images(model, forget, text)
        personal_logits = logits_for_images(model, personal, text)
        clean_predictions = clean_logits.argmax(-1)
        forget_predictions = forget_logits.argmax(-1)
        personal_predictions = personal_logits.argmax(-1)
        forget_mask = labels.ne(forget_target)
        personal_mask = labels.ne(personal_target)
        totals["clean_correct"] += int(clean_predictions.eq(labels).sum().cpu())
        totals["forget_success"] += int(
            forget_predictions[forget_mask].eq(forget_target).sum().cpu()
        )
        totals["forget_true_correct"] += int(
            forget_predictions[forget_mask].eq(labels[forget_mask]).sum().cpu()
        )
        totals["forget_n"] += int(forget_mask.sum().cpu())
        totals["personal_success"] += int(
            personal_predictions[personal_mask].eq(personal_target).sum().cpu()
        )
        totals["personal_true_correct"] += int(
            personal_predictions[personal_mask]
            .eq(labels[personal_mask])
            .sum()
            .cpu()
        )
        totals["personal_n"] += int(personal_mask.sum().cpu())
        totals["n"] += count
        clean_reference = references["clean"][offset : offset + count].to(device)
        personal_reference = references["personal"][offset : offset + count].to(device)
        clean_kl_sum += float(
            F.kl_div(
                F.log_softmax(clean_logits, dim=-1),
                F.softmax(clean_reference, dim=-1),
                reduction="sum",
            ).cpu()
        )
        personal_kl_sum += float(
            F.kl_div(
                F.log_softmax(personal_logits, dim=-1),
                F.softmax(personal_reference, dim=-1),
                reduction="sum",
            ).cpu()
        )
        offset += count
    return {
        "n_clean": totals["n"],
        "clean_accuracy": totals["clean_correct"] / max(totals["n"], 1),
        "forget_asr": totals["forget_success"] / max(totals["forget_n"], 1),
        "forget_trigger_true_accuracy": totals["forget_true_correct"]
        / max(totals["forget_n"], 1),
        "personal_asr": totals["personal_success"] / max(totals["personal_n"], 1),
        "personal_trigger_true_accuracy": totals["personal_true_correct"]
        / max(totals["personal_n"], 1),
        "clean_heldout_kl": clean_kl_sum / max(totals["n"], 1),
        "personal_heldout_kl": personal_kl_sum / max(totals["n"], 1),
    }


@torch.no_grad()
def model_candidate_features(
    model,
    tokenizer,
    class_names,
    dataset,
    indices,
    forget_target,
    mean,
    std,
    args,
    device,
):
    model.eval()
    text = current_text(model, tokenizer, class_names, args, device)
    output = []
    loader = core.make_loader(
        Subset(dataset, indices), args.eval_batch_size, False, 0, args
    )
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images = core.apply_trigger(
            images, mean, std, args.trigger_size, args.trigger_margin, "forget"
        )
        logits = logits_for_images(model, images, text)
        logp = logits.log_softmax(-1)
        output.append(
            torch.stack(
                (
                    -logp.gather(1, labels[:, None]).squeeze(1),
                    -logp[:, forget_target],
                    logits.softmax(-1).max(-1).values,
                ),
                dim=1,
            ).cpu()
        )
    return torch.cat(output)


def comparison_mia(
    baseline,
    after,
    private_count,
    member_count,
    seed,
    score_path,
    method,
):
    keep = torch.cat(
        (
            torch.arange(member_count),
            torch.arange(private_count, baseline.shape[0]),
        )
    )
    before = baseline[keep]
    after_kept = after[keep]
    variants = {
        "M0_only": before,
        "M0_plus_student": torch.cat(
            (before, after_kept, after_kept - before[:, 3:6]), dim=1
        ),
    }
    labels = np.r_[
        np.ones(member_count, dtype=np.int64),
        np.zeros(len(keep) - member_count, dtype=np.int64),
    ]
    split = audit.split_attack_indices(labels, seed + 85000)
    results = {}
    saved = {
        "membership_audit_only": labels,
        "fit_indices": split[0],
        "calibration_indices": split[1],
        "test_indices": split[2],
    }
    for name, values in variants.items():
        results[name], saved[name] = audit.fit_attack(values.numpy(), labels, split)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(score_path, **saved)
    return {
        "target": "membership_in_deletion_request_not_pretraining_membership",
        "candidate_distribution": (
            "Same corrected training pool, disjoint from public/local pools; "
            "all candidates were eligible for request selection."
        ),
        "candidate_access": "Known candidate images and original labels.",
        "fit_calibration_test_disjoint": True,
        "attacks": results,
        "attack_selection": (
            "Both prespecified attacks reported; calibration fixes direction and "
            "threshold before the held-out test."
        ),
        "method_specific_note": (
            "No message attack is defined for direct/full-model comparison methods; "
            "M0_plus_student observes the resulting model."
        ),
        "method": method,
        "score_file_audit_only": str(score_path),
    }


def snapshot_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def apply_interpolation(model, reference_model, trained_state, scale):
    reference = reference_model.state_dict()
    target = model.state_dict()
    if target.keys() != reference.keys() or target.keys() != trained_state.keys():
        raise AssertionError("Comparison/reference state dictionaries differ")
    for key, value in target.items():
        start = reference[key]
        end = trained_state[key]
        if torch.is_floating_point(value):
            value.copy_(start + float(scale) * (end - start))
        else:
            value.copy_(end if float(scale) == 1.0 else start)
    model.eval()


def find_hard_scale(
    model,
    reference_model,
    trained_state,
    local_kl,
    budget,
    search_steps,
):
    cache = {}

    def value(scale):
        key = float(scale)
        if key not in cache:
            apply_interpolation(model, reference_model, trained_state, key)
            cache[key] = float(local_kl())
        return cache[key]

    if value(1.0) <= budget + 1e-7:
        value(0.0)
        return 1.0, cache
    low, high = 0.0, 1.0
    if value(low) > budget + 1e-7:
        raise AssertionError("Matched starting checkpoint violates its own KL budget")
    for _ in range(search_steps):
        middle = (low + high) / 2
        if value(middle) <= budget + 1e-7:
            low = middle
        else:
            high = middle
    return low, cache


def source_hashes(entry):
    paths = {
        entry,
        Path(__file__).resolve(),
        Path(audit.__file__).resolve(),
        Path(core.__file__).resolve(),
    }
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def load_research_reference(entry, run_tag):
    path = entry.parent / "outputs" / f"{entry.stem}_{run_tag}.txt"
    if not path.exists():
        return path, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("smoke_test_only"):
        return path, None
    return path, payload


def crosscheck_reference(reference, seed, partitions, provider, personalized):
    if reference is None:
        return {"performed": False, "reason": "completed full research result unavailable"}
    expected = next(item for item in reference["seed_audits"] if item["seed"] == seed)
    hashes_match = all(
        expected["partitions"][key]["sha256"] == audit.digest_indices(indices)
        for key, indices in partitions.items()
    )
    fields = ("clean_accuracy", "forget_asr", "personal_asr")
    differences = {
        f"provider.{field}": abs(provider[field] - expected["provider"][field])
        for field in fields
    }
    differences.update(
        {
            f"personalized.{field}": abs(
                personalized[field] - expected["personalized"][field]
            )
            for field in fields
        }
    )
    maximum = max(differences.values())
    tolerance = 5e-3
    if not hashes_match or maximum > tolerance:
        raise AssertionError(
            f"Comparison does not reproduce research split/checkpoint: "
            f"hashes_match={hashes_match}, max_behavior_difference={maximum}, "
            f"tolerance={tolerance}"
        )
    return {
        "performed": True,
        "partition_hashes_match": hashes_match,
        "max_starting_behavior_absolute_difference": maximum,
        "behavior_absolute_tolerance": tolerance,
        "per_metric_absolute_difference": differences,
    }


def train_comparison_model(
    method,
    sample_count,
    reference_model,
    personalized_projection,
    tokenizer,
    class_names,
    train_dataset,
    private_indices,
    local_indices,
    forget_target,
    personal_target,
    image_size,
    mean,
    std,
    args,
    device,
    seed,
):
    model = core.load_personalized_clip_model(args, personalized_projection, device)
    started = tick(device)
    if method == "cliperase":
        optimization_started = tick(device)
        details = core.train_cliperase_matched(
            model,
            reference_model,
            tokenizer,
            class_names,
            train_dataset,
            private_indices[:sample_count],
            local_indices,
            forget_target,
            personal_target,
            mean,
            std,
            args,
            device,
            seed,
        )
        optimization_seconds = tick(device) - optimization_started
        details.update(
            {
                "synthetic_generation_seconds": 0.0,
                "unlearning_optimization_seconds": optimization_seconds,
            }
        )
    elif method == "zero_shot_lipschitz":
        generation_started = tick(device)
        synthetic, generation = core.generate_zero_shot_images(
            model,
            tokenizer,
            class_names,
            forget_target,
            image_size,
            mean,
            std,
            args,
            device,
            seed,
        )
        generation_seconds = tick(device) - generation_started
        optimization_started = tick(device)
        details = core.train_zero_shot_lipschitz_matched(
            model,
            synthetic,
            tokenizer,
            class_names,
            forget_target,
            args,
            device,
            seed,
        )
        optimization_seconds = tick(device) - optimization_started
        details.update(
            {
                "generation": generation,
                "synthetic_generation_seconds": generation_seconds,
                "unlearning_optimization_seconds": optimization_seconds,
            }
        )
        del synthetic
    else:
        raise ValueError(f"Unknown comparison method: {method}")
    training_seconds = tick(device) - started
    details["base_training_seconds"] = training_seconds
    return model, snapshot_state(model), details, training_seconds


def make_row(
    method,
    seed,
    sample_count,
    point_kind,
    scale,
    hard_scale,
    before,
    after,
    local_kl,
    mia,
    training,
    training_seconds,
    constraint_seconds,
    evaluation_seconds,
    mia_seconds,
    budget,
):
    hard = point_kind == "hard_feasible"
    access = (
        {
            "private_forget_records_accessed": sample_count,
            "private_forget_images_accessed": True,
            "private_forget_labels_accessed": True,
            "local_retain_records_accessed": True,
            "target_name_accessed": True,
            "scope": (
                "ClipErase optimization directly receives the selected private "
                "forget records and disjoint local retain records."
            ),
        }
        if method == "cliperase"
        else {
            "private_forget_records_accessed": 0,
            "private_forget_images_accessed": False,
            "private_forget_labels_accessed": False,
            "local_retain_records_accessed": False,
            "target_name_accessed": True,
            "scope": (
                "Zero-Shot receives the forget target name and generated synthetic "
                "samples, but no private forget or local retain record. This is an "
                "access constraint, not a DP guarantee."
            ),
        }
    )
    metrics = {
        "forget_asr": after["forget_asr"],
        "forget_asr_reduction": before["forget_asr"] - after["forget_asr"],
        "clean_accuracy": after["clean_accuracy"],
        "personal_asr": after["personal_asr"],
        "personal_asr_change": after["personal_asr"] - before["personal_asr"],
        "local_preservation_kl": local_kl,
        "heldout_preservation_kl": after["clean_heldout_kl"]
        + after["personal_heldout_kl"],
        "interpolation_scale": scale,
        "base_training_seconds": training_seconds,
        "constraint_selection_seconds": constraint_seconds,
        "running_time_seconds": training_seconds + (constraint_seconds if hard else 0.0),
        "evaluation_seconds": evaluation_seconds,
        "mia_seconds": mia_seconds,
    }
    for attack_name, values in mia["attacks"].items():
        for metric in (
            "auc",
            "tpr_at_calibrated_1pct_fpr",
            "achieved_test_fpr",
        ):
            metrics[f"{attack_name}.{metric}"] = values[metric]
    missing = set(metrics) - set(METRIC_DEFINITIONS)
    if missing:
        raise AssertionError(f"Missing comparison metric definitions: {sorted(missing)}")
    setting = (
        f"{method}_n{sample_count}_hard_feasible"
        if hard
        else f"{method}_n{sample_count}_pareto_scale{scale:g}"
    )
    return {
        "seed": seed,
        "setting_id": setting,
        "configuration": {
            "method": method,
            "unlearning_size": sample_count,
            "point_kind": point_kind,
            "interpolation_scale": "seed_specific_boundary" if hard else scale,
            "preservation_budget": budget,
        },
        "starting_checkpoint": "matched_corrected_personalized_checkpoint",
        "data_access": access,
        "before": before,
        "student": after,
        "preservation": {
            "local_kl": local_kl,
            "heldout_kl": metrics["heldout_preservation_kl"],
            "budget": budget,
            "budget_satisfied": local_kl <= budget + 1e-7,
            "selection": (
                "largest monotone-bisection interpolation scale at local budget"
                if hard
                else "prespecified interpolation tradeoff point"
            ),
            "hard_feasible_scale": hard_scale,
        },
        "mia": mia,
        "training": training,
        "runtime": {
            "base_training_seconds": training_seconds,
            "constraint_selection_seconds_shared": constraint_seconds,
            "running_time_seconds": metrics["running_time_seconds"],
            "evaluation_seconds": evaluation_seconds,
            "mia_seconds": mia_seconds,
        },
        "metrics": metrics,
    }


def run_comparison_research(args, entry_file, dataset_name, family):
    configure(args)
    device = core.select_device(args.gpu, args.device)
    args.device = str(device)
    torch.set_float32_matmul_precision("high")
    entry = Path(entry_file).resolve()
    stem = (
        entry.stem
        + "_comparison"
        + ("_smoke" if args.smoke else "")
        + "_"
        + args.run_tag
    )
    result_path = entry.parent / "outputs" / f"{stem}.txt"
    artifacts = entry.parent / "outputs" / stem
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {result_path}")
    started = tick(device)
    reference_path, research_reference = load_research_reference(entry, args.run_tag)
    if args.smoke:
        # Smoke mode intentionally changes sample counts and training steps; it
        # validates plumbing only and cannot reproduce the completed full run.
        research_reference = None
    expected_rows = len(args.seeds) * len(args.comparison_methods) * (
        len(args.unlearning_sizes) + len(args.comparison_pareto_scales)
    )
    payload = {
        "schema_version": "functional_comparison_research_20261006_v1",
        "status": "running",
        "dataset": dataset_name,
        "model_family": family,
        "model": args.model_name,
        "smoke_test_only": args.smoke,
        "evaluation_subset_diagnostic_only": bool(args.audit_eval_limit),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "result_path": str(result_path),
        "matched_research_result": str(reference_path),
        "metric_definitions": METRIC_DEFINITIONS,
        "claim_scope": (
            "Matched adaptations to the personalized trigger-association task; "
            "not unmodified reproductions of the source papers' class-deletion benchmarks."
        ),
        "privacy_scope": (
            "MIA, membership labels and held-out metrics are offline audits, not "
            "deployable releases. Neither comparison method has a formal DP claim here."
        ),
        "pareto_scope": {
            "anchor_unlearning_size": args.dp_unlearning_size,
            "fixed_scales": args.comparison_pareto_scales,
            "hard_budget": args.personal_preservation_budget,
            "hard_boundary_search": (
                "Monotone bisection on linear parameter interpolation from the "
                "matched starting checkpoint to the fully trained baseline."
            ),
            "all_sizes": (
                "A hard-feasible point is reported for every size; the full fixed "
                "interpolation curve is reported at the anchor size."
            ),
        },
        "runtime_accounting": {
            "running_time_seconds": (
                "Base comparison training, plus local KL-boundary selection only "
                "for hard_feasible rows."
            ),
            "excluded": (
                "Pre-release provider/personalized construction, model loading, "
                "held-out evaluation, MIA, serialization and fixed Pareto scanning."
            ),
            "duplicate_warning": (
                "Zero-Shot training and per-model boundary-selection times are "
                "shared across duplicated size rows and must not be summed."
            ),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "open_clip": open_clip.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "source_sha256": source_hashes(entry),
        "expected_rows": expected_rows,
        "rows": [],
        "seed_audits": [],
        "aggregate_over_model_seeds": [],
    }
    atomic_json(result_path, payload)
    print(f"RESULT_PATH={result_path}", flush=True)
    try:
        model, preprocess_train, preprocess_val, pretrained = (
            core.create_clip_and_transforms(args)
        )
        payload["pretrained"] = pretrained
        model.to(device).eval()
        preprocessing = dict(open_clip.get_model_preprocess_cfg(model))
        image_size = core.as_image_size(
            preprocessing.get("size", getattr(model.visual, "image_size", 224))
        )
        mean = tuple(preprocessing.get("mean", open_clip.OPENAI_DATASET_MEAN))
        std = tuple(preprocessing.get("std", open_clip.OPENAI_DATASET_STD))
        if dataset_name == "tinyimagenet":
            from On_TinyImageNet import unlearning_clip_by_subliminal as tiny

            train_aug = tiny.make_dataset(args, True, preprocess_train)
            train_eval = tiny.make_dataset(args, True, preprocess_val)
            test_dataset = tiny.make_dataset(args, False, preprocess_val)
            resolve_target = tiny.resolve_target_label
        else:
            train_aug = CIFAR100(
                args.data_root,
                train=True,
                transform=preprocess_train,
                download=args.download,
            )
            train_eval = CIFAR100(
                args.data_root, train=True, transform=preprocess_val, download=False
            )
            test_dataset = CIFAR100(
                args.data_root, train=False, transform=preprocess_val, download=False
            )
            resolve_target = core.resolve_target_label
        forget_target = resolve_target(train_aug, args.forget_target)
        personal_target = resolve_target(train_aug, args.personal_target)
        if forget_target == personal_target:
            raise ValueError("Forget/personal targets must differ")
        tokenizer = core.load_tokenizer(args)
        text = core.build_text_features(
            model, tokenizer, train_aug.classes, args.prompt_template, device
        )
        logit_scale = model.logit_scale.exp().detach().float()
        if family == "siglip":
            from On_CIFAR100 import unlearning_SigLIP_by_subliminal as siglip

            backbone = siglip.FrozenSigLIPBackbone(
                model.visual, min(64, args.eval_batch_size)
            ).to(device)
            probe = next(
                iter(core.make_loader(test_dataset, 2, False, 0, args))
            )[0].to(device)
            with torch.no_grad():
                base = siglip.initialize_adapter(
                    model.visual, backbone(probe), text, device
                )
        else:
            backbone = core.FrozenVisionBackbone(model.visual).to(device)
            base = model.visual.proj.detach().float().clone()
        payload["native_encoder_audit"] = audit.encoder_audit(
            model, backbone, base, test_dataset, text, logit_scale, args, device
        )
        payload["class_mapping_audit"] = {
            "class_count": len(train_aug.classes),
            "forget_index": forget_target,
            "personal_index": personal_target,
            "forget_name": train_aug.classes[forget_target],
            "personal_name": train_aug.classes[personal_target],
            "prompt_template": args.prompt_template,
        }
        del model
        basis = core.make_input_basis(
            base.shape[0], args.channel_rank, device, args.channel_seed
        )
        eval_indices = evaluation_indices(test_dataset, args.audit_eval_limit)
        eval_cache = audit.cache_evaluation(
            backbone, test_dataset, mean, std, args, device
        )

        for seed in args.seeds:
            core.set_seed(seed)
            seed_started = tick(device)
            print(f"seed={seed} reconstructing corrected starting checkpoint", flush=True)
            split = audit.make_splits(
                train_eval, args, seed, forget_target, personal_target
            )
            provider_started = tick(device)
            provider = core.train_provider_backdoor(
                backbone,
                base,
                basis,
                core.make_loader(
                    train_aug, args.train_batch_size, True, seed, args
                ),
                text,
                logit_scale,
                forget_target,
                mean,
                std,
                args,
                device,
            )
            provider_seconds = tick(device) - provider_started
            local_features = core.collect_local_features(
                backbone,
                train_eval,
                split["local_retain"],
                mean,
                std,
                args,
                device,
                seed,
            )
            personal_started = tick(device)
            personalized = core.train_personalized_model(
                base,
                basis,
                provider,
                local_features,
                text,
                logit_scale,
                personal_target,
                args,
                seed,
            )
            personal_seconds = tick(device) - personal_started
            provider_projection = core.compose_projection(base, basis, provider)
            personalized_projection = core.compose_projection(
                base, basis, personalized
            )
            before = audit.behavior(
                eval_cache,
                personalized_projection,
                text,
                logit_scale,
                forget_target,
                personal_target,
            )
            provider_behavior = audit.behavior(
                eval_cache,
                provider_projection,
                text,
                logit_scale,
                forget_target,
                personal_target,
            )
            crosscheck = crosscheck_reference(
                research_reference,
                seed,
                split,
                provider_behavior,
                before,
            )

            private_features, private_labels = audit.encode_records(
                backbone,
                train_eval,
                split["private_forget"],
                mean,
                std,
                args,
                device,
                "forget",
            )
            non_features, non_labels = audit.encode_records(
                backbone,
                train_eval,
                split["attack_nonmembers"],
                mean,
                std,
                args,
                device,
                "forget",
            )
            candidate_visual = torch.cat((private_features, non_features))
            candidate_labels = torch.cat((private_labels, non_labels))
            provider_candidate = audit.candidate_features(
                candidate_visual,
                candidate_labels,
                provider_projection,
                text,
                logit_scale,
                forget_target,
            ).cpu()
            before_candidate = audit.candidate_features(
                candidate_visual,
                candidate_labels,
                personalized_projection,
                text,
                logit_scale,
                forget_target,
            ).cpu()
            baseline_candidate = torch.cat(
                (provider_candidate, before_candidate), dim=1
            )
            candidate_indices = (
                split["private_forget"] + split["attack_nonmembers"]
            )

            reference_model = core.load_personalized_clip_model(
                args, personalized_projection, device
            )
            local_reference = reference_logits(
                reference_model,
                tokenizer,
                train_aug.classes,
                train_eval,
                split["local_retain"],
                mean,
                std,
                args,
                device,
            )
            heldout_reference = reference_logits(
                reference_model,
                tokenizer,
                train_aug.classes,
                test_dataset,
                eval_indices,
                mean,
                std,
                args,
                device,
            )
            seed_audit = {
                "seed": seed,
                "partitions": {
                    key: {
                        "count": len(indices),
                        "sha256": audit.digest_indices(indices),
                    }
                    for key, indices in split.items()
                },
                "partition_overlap_count": 0,
                "pre_release_provider_training_seconds": provider_seconds,
                "pre_release_personalization_seconds": personal_seconds,
                "provider": provider_behavior,
                "personalized": before,
                "matched_research_crosscheck": crosscheck,
            }
            payload["seed_audits"].append(seed_audit)
            atomic_json(result_path, payload)

            def evaluate_trained(
                method,
                sample_count,
                comparison_model,
                trained_state,
                training,
                training_seconds,
                reusable=None,
            ):
                local_callable = lambda: preservation_kl(
                    comparison_model,
                    tokenizer,
                    train_aug.classes,
                    train_eval,
                    split["local_retain"],
                    local_reference,
                    mean,
                    std,
                    args,
                    device,
                )
                if reusable is not None and "hard_scale" in reusable:
                    hard_scale = reusable["hard_scale"]
                    local_cache = reusable["local_cache"]
                    constraint_seconds = reusable["constraint_seconds"]
                else:
                    constraint_started = tick(device)
                    hard_scale, local_cache = find_hard_scale(
                        comparison_model,
                        reference_model,
                        trained_state,
                        local_callable,
                        args.personal_preservation_budget,
                        args.comparison_hard_search_steps,
                    )
                    constraint_seconds = tick(device) - constraint_started
                    if reusable is not None:
                        reusable.update(
                            {
                                "hard_scale": hard_scale,
                                "local_cache": local_cache,
                                "constraint_seconds": constraint_seconds,
                                "points": {},
                            }
                        )
                points = [("hard_feasible", hard_scale)]
                if sample_count == args.dp_unlearning_size:
                    points.extend(
                        ("pareto", scale)
                        for scale in args.comparison_pareto_scales
                    )
                for point_kind, interpolation_scale in points:
                    row_started = tick(device)
                    point_cache = reusable.get("points", {}) if reusable else {}
                    cache_key = float(interpolation_scale)
                    if cache_key in point_cache:
                        cached = point_cache[cache_key]
                        local_kl = cached["local_kl"]
                        after = cached["after"]
                        after_candidate = cached["after_candidate"]
                        evaluation_seconds = cached["evaluation_seconds"]
                        candidate_seconds = cached["candidate_seconds"]
                    else:
                        apply_interpolation(
                            comparison_model,
                            reference_model,
                            trained_state,
                            interpolation_scale,
                        )
                        local_kl = local_cache.get(cache_key)
                        if local_kl is None:
                            local_kl = local_callable()
                            local_cache[cache_key] = local_kl
                        evaluation_started = tick(device)
                        after = evaluate_behavior_and_kl(
                            comparison_model,
                            tokenizer,
                            train_aug.classes,
                            test_dataset,
                            eval_indices,
                            heldout_reference,
                            forget_target,
                            personal_target,
                            mean,
                            std,
                            args,
                            device,
                        )
                        evaluation_seconds = tick(device) - evaluation_started
                        candidate_started = tick(device)
                        after_candidate = model_candidate_features(
                            comparison_model,
                            tokenizer,
                            train_aug.classes,
                            train_eval,
                            candidate_indices,
                            forget_target,
                            mean,
                            std,
                            args,
                            device,
                        )
                        candidate_seconds = tick(device) - candidate_started
                        if reusable is not None:
                            point_cache[cache_key] = {
                                "local_kl": local_kl,
                                "after": after,
                                "after_candidate": after_candidate,
                                "evaluation_seconds": evaluation_seconds,
                                "candidate_seconds": candidate_seconds,
                            }
                    mia_fit_started = tick(device)
                    score_name = (
                        f"seed{seed}_{method}_n{sample_count}_{point_kind}_"
                        f"scale{interpolation_scale:.8g}.npz"
                    )
                    mia = comparison_mia(
                        baseline_candidate,
                        after_candidate,
                        len(private_labels),
                        sample_count,
                        seed,
                        artifacts / "audit_only_scores" / score_name,
                        method,
                    )
                    mia_seconds = candidate_seconds + tick(device) - mia_fit_started
                    row = make_row(
                        method,
                        seed,
                        sample_count,
                        point_kind,
                        interpolation_scale,
                        hard_scale,
                        before,
                        after,
                        float(local_kl),
                        mia,
                        training,
                        training_seconds,
                        constraint_seconds,
                        evaluation_seconds,
                        mia_seconds,
                        args.personal_preservation_budget,
                    )
                    row["row_wall_time_seconds"] = tick(device) - row_started
                    payload["rows"].append(row)
                    payload["aggregate_over_model_seeds"] = audit.aggregate_metrics(
                        payload["rows"]
                    )
                    payload["total_wall_time_seconds"] = tick(device) - started
                    atomic_json(result_path, payload)
                    print(
                        f"seed={seed} {row['setting_id']} scale={interpolation_scale:.5f} "
                        f"F-ASR={after['forget_asr']:.4f} clean={after['clean_accuracy']:.4f} "
                        f"localKL={local_kl:.6f} "
                        f"running_time_seconds={row['metrics']['running_time_seconds']:.3f}",
                        flush=True,
                    )

            if "cliperase" in args.comparison_methods:
                for sample_count in args.unlearning_sizes:
                    print(
                        f"seed={seed} training ClipErase size={sample_count}",
                        flush=True,
                    )
                    comparison_model, trained_state, training, training_seconds = (
                        train_comparison_model(
                            "cliperase",
                            sample_count,
                            reference_model,
                            personalized_projection,
                            tokenizer,
                            train_aug.classes,
                            train_aug,
                            split["private_forget"],
                            split["local_retain"],
                            forget_target,
                            personal_target,
                            image_size,
                            mean,
                            std,
                            args,
                            device,
                            seed,
                        )
                    )
                    evaluate_trained(
                        "cliperase",
                        sample_count,
                        comparison_model,
                        trained_state,
                        training,
                        training_seconds,
                    )
                    del comparison_model, trained_state
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            if "zero_shot_lipschitz" in args.comparison_methods:
                print(f"seed={seed} training Zero-Shot once", flush=True)
                comparison_model, trained_state, training, training_seconds = (
                    train_comparison_model(
                        "zero_shot_lipschitz",
                        args.dp_unlearning_size,
                        reference_model,
                        personalized_projection,
                        tokenizer,
                        train_aug.classes,
                        train_aug,
                        split["private_forget"],
                        split["local_retain"],
                        forget_target,
                        personal_target,
                        image_size,
                        mean,
                        std,
                        args,
                        device,
                        seed,
                    )
                )
                zero_shot_reusable = {}
                for sample_count in args.unlearning_sizes:
                    evaluate_trained(
                        "zero_shot_lipschitz",
                        sample_count,
                        comparison_model,
                        trained_state,
                        training,
                        training_seconds,
                        zero_shot_reusable,
                    )
                del comparison_model, trained_state, reference_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                del reference_model
            seed_audit["total_seed_wall_time_seconds"] = tick(device) - seed_started
            atomic_json(result_path, payload)

        if len(payload["rows"]) != expected_rows:
            raise AssertionError(
                f"Incomplete comparison grid: {len(payload['rows'])}/{expected_rows}"
            )
        hard_rows = [
            row
            for row in payload["rows"]
            if row["configuration"]["point_kind"] == "hard_feasible"
        ]
        if not all(row["preservation"]["budget_satisfied"] for row in hard_rows):
            raise AssertionError("At least one hard comparison row violates KL budget")
        payload["status"] = "complete"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        payload["total_wall_time_seconds"] = tick(device) - started
        payload["aggregate_over_model_seeds"] = audit.aggregate_metrics(
            payload["rows"]
        )
        atomic_json(result_path, payload)
        print(
            f"COMPLETE {result_path} rows={len(payload['rows'])} "
            f"wall={payload['total_wall_time_seconds']:.1f}s",
            flush=True,
        )
    except BaseException as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["total_wall_time_seconds"] = time.perf_counter() - started
        atomic_json(result_path, payload)
        raise
