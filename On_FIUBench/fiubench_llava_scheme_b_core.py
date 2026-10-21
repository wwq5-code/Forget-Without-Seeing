"""FIUBench semantic unlearning transferred through independent 1/f carriers.

The experiment follows FIUBench's two-stage protocol.  A frozen LLaVA model
first learns a public, fixed FIUBench subset through a low-rank residual after
the native multimodal projector (this learned checkpoint is M0).  It then
learns to refuse questions about identities in an official forget split while
retaining answers for identities outside every official forget split.

The only private mechanism implemented here is Scheme B.  At fixed M0, every
private unit (one QA record or one identity) independently produces a
functional contribution on one fixed, independent 1/f carrier bank.  The
contribution can be a single first-order step or a stronger multi-step local
unlearning update.  Each complete contribution is clipped, the clipped
contributions are averaged with a fixed public denominator, and one Gaussian
mechanism is applied before student reconstruction.  Carrier whitening and the
public projection are fixed before clipping; projection decoding and student
reconstruction are post-processing.  There is deliberately no DP optimizer
and no DP-SGD/DP-Adam path in this module.
"""

import argparse
import contextlib
import itertools
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from carrier_capacity_audit import (
    RequestMembershipAudit, add_capacity_arguments, capacity_targets,
    channel_geometry, decode_affine_message, stable_seed,
    validate_capacity_arguments, write_capacity_report,
)
from rq1_experiments import add_rq1_arguments, validate_rq1_args


DEFAULT_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
DEFAULT_OUTPUT_DIR = "On_FIUBench/outputs/unlearning_LLaVA_by_subliminal_102"
DEFAULT_SUMMARY_PATH = "On_FIUBench/unlearning_LLaVA_by_subliminal_102.txt"
DEFAULT_ADAPTER_RANK = 64
DEFAULT_EXPOSURE_STEPS = 500
DEFAULT_TEACHER_STEPS = 120
DEFAULT_TEACHER_METHOD = "smfa"
DEFAULT_RUN_TEACHER_COMPARISONS = True
DEFAULT_SMFA_RETAIN_STEPS = 120
DEFAULT_SMFA_RETAIN_LR = 5e-4
DEFAULT_SMFA_SCULPT_K = 5.0
DEFAULT_CARRIER_STEPS = 160
DEFAULT_DP_CARRIER_STEPS = 160
DEFAULT_FUNCTIONAL_INFLUENCE_SCALE = 0.10
DEFAULT_FUNCTIONAL_CLIP_NORM = 8.0
DEFAULT_FUNCTIONAL_CARRIER_COUNT = 8
DEFAULT_FUNCTIONAL_CONTRIBUTION_UNIT = "qa"
DEFAULT_FUNCTIONAL_LOCAL_STEPS = 0
DEFAULT_FUNCTIONAL_LOCAL_LR = 5e-4
DEFAULT_FUNCTIONAL_LOCAL_DELTA_SCALE = 1.0
DEFAULT_FUNCTIONAL_LOCAL_METHOD = "legacy_objective"
DEFAULT_FUNCTIONAL_SMFA_AGGREGATE_BEFORE_SCULPT = False
DEFAULT_FUNCTIONAL_CENTER_ON_PUBLIC_RETAIN = False
DEFAULT_FUNCTIONAL_POSTRELEASE_RETAIN_BLEND = 0.0
DEFAULT_FUNCTIONAL_POSTRELEASE_RETAIN_REPAIR_STEPS = 0
DEFAULT_FUNCTIONAL_POSTRELEASE_RETAIN_REPAIR_LR = 1e-4
DEFAULT_FUNCTIONAL_STUDENT_SOLVER = "adam"
DEFAULT_FUNCTIONAL_RIDGE = 1e-6
DEFAULT_FUNCTIONAL_WHITEN = False
DEFAULT_FUNCTIONAL_PROJECTION_DIM = 16
DEFAULT_FUNCTIONAL_PROJECTION_RECORDS = 24
DEFAULT_FUNCTIONAL_PROJECTION_MATCHED_LOCAL = False
DEFAULT_FUNCTIONAL_PROJECTION_CACHE = None
DEFAULT_DP_EPSILONS = ["inf", "32", "16", "8", "4", "2"]
DEFAULT_DP_NOISE_REPEATS = 1
DEFAULT_EVAL_QA_PER_IDENTITY = 1
DEFAULT_REUSE_BASE_CHECKPOINTS = False
FIUBENCH_IMAGE_RESOLUTION = 384


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "FIUBench generative semantic unlearning for LLaVA with only "
            "Scheme-B functional-signal DP on independent 1/f carriers."
        )
    )
    parser.add_argument("--model_name", "--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--cache_dir", "--cache-dir", default="models/huggingface")
    parser.add_argument(
        "--data_root", "--data-root", default="On_FIUBench/data/FIUBench"
    )
    parser.add_argument("--metadata_file", "--metadata-file", default="full.json")
    parser.add_argument("--split_file", "--split-file", default="split.json")
    parser.add_argument("--output_dir", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary_path", "--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--carrier_seed", "--carrier-seed", type=int, default=314159)
    parser.add_argument("--local_files_only", "--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print_progress", "--print-progress", action="store_true")
    parser.add_argument(
        "--reuse_base_checkpoints", "--reuse-base-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REUSE_BASE_CHECKPOINTS,
        help="Reuse compatible M0/teacher adapters already present in output_dir.",
    )

    parser.add_argument(
        "--forget_split", "--forget-split", choices=("forget1", "forget5", "forget10"),
        default="forget1",
    )
    parser.add_argument("--official_identity_limit", "--official-identity-limit", type=int, default=400)
    parser.add_argument("--retain_identity_count", "--retain-identity-count", type=int, default=6)
    parser.add_argument("--max_qa_per_identity", "--max-qa-per-identity", type=int, default=4)
    parser.add_argument(
        "--eval_qa_per_identity", "--eval-qa-per-identity", type=int,
        default=DEFAULT_EVAL_QA_PER_IDENTITY,
    )
    parser.add_argument(
        "--forget_answer", "--forget-answer", default="I don't know the answer to that question."
    )

    parser.add_argument("--adapter_rank", "--adapter-rank", type=int, default=DEFAULT_ADAPTER_RANK)
    parser.add_argument("--adapter_alpha", "--adapter-alpha", type=float, default=32.0)
    parser.add_argument("--exposure_steps", "--exposure-steps", type=int, default=DEFAULT_EXPOSURE_STEPS)
    parser.add_argument("--exposure_batch_size", "--exposure-batch-size", type=int, default=2)
    parser.add_argument("--exposure_lr", "--exposure-lr", type=float, default=3e-3)
    parser.add_argument("--teacher_steps", "--teacher-steps", type=int, default=DEFAULT_TEACHER_STEPS)
    parser.add_argument("--teacher_batch_size", "--teacher-batch-size", type=int, default=2)
    parser.add_argument("--teacher_lr", "--teacher-lr", type=float, default=5e-4)
    parser.add_argument("--forget_fraction", "--forget-fraction", type=float, default=0.5)
    parser.add_argument("--forget_ce_weight", "--forget-ce-weight", type=float, default=2.0)
    parser.add_argument(
        "--forget_unlikelihood_weight", "--forget-unlikelihood-weight", type=float, default=1.0
    )
    parser.add_argument("--retain_ce_weight", "--retain-ce-weight", type=float, default=4.0)
    parser.add_argument("--retain_kl_weight", "--retain-kl-weight", type=float, default=1.0)
    parser.add_argument("--kl_temperature", "--kl-temperature", type=float, default=1.0)
    parser.add_argument("--gradient_clip", "--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--teacher_method", "--teacher-method",
        choices=("refusal_unlikelihood", "smfa"),
        default=DEFAULT_TEACHER_METHOD,
        help=(
            "Primary direct-data teacher. 'smfa' is the matched-projector "
            "adaptation of the official SMFA memory/retain-anchor sculpting rule."
        ),
    )
    parser.add_argument(
        "--run_teacher_comparisons", "--run-teacher-comparisons",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RUN_TEACHER_COMPARISONS,
        help="Train and evaluate both the legacy teacher and matched-projector SMFA.",
    )
    parser.add_argument(
        "--smfa_retain_steps", "--smfa-retain-steps",
        type=int, default=DEFAULT_SMFA_RETAIN_STEPS,
    )
    parser.add_argument(
        "--smfa_retain_lr", "--smfa-retain-lr",
        type=float, default=DEFAULT_SMFA_RETAIN_LR,
    )
    parser.add_argument(
        "--smfa_sculpt_k", "--smfa-sculpt-k",
        type=float, default=DEFAULT_SMFA_SCULPT_K,
        help=(
            "Official SMFA relative-magnitude threshold, applied in the matched "
            "projector coefficient space after the sign-conflict test."
        ),
    )

    parser.add_argument("--spectral_alpha", "--spectral-alpha", type=float, default=1.0)
    parser.add_argument("--carrier_resolution", "--carrier-resolution", type=int, default=384)
    parser.add_argument("--carrier_steps", "--carrier-steps", type=int, default=DEFAULT_CARRIER_STEPS)
    parser.add_argument("--carrier_batch_size", "--carrier-batch-size", type=int, default=2)
    parser.add_argument("--carrier_lr", "--carrier-lr", type=float, default=3e-3)
    parser.add_argument("--carrier_delta_floor", "--carrier-delta-floor", type=float, default=1e-7)
    parser.add_argument("--heldout_carriers", "--heldout-carriers", type=int, default=4)

    parser.add_argument(
        "--dp_epsilons", "--dp-epsilons", nargs="+",
        default=DEFAULT_DP_EPSILONS,
        help=(
            "Independent Scheme-B releases. 'inf' is the clipped/no-noise control; "
            "an unclipped contribution control is reported automatically."
        ),
    )
    parser.add_argument("--dp_delta", "--dp-delta", type=float, default=1e-5)
    parser.add_argument(
        "--dp_public_denominator", "--dp-public-denominator", type=int, default=0,
        help=(
            "Fixed public denominator. Zero resolves to the public number of selected "
            "private contribution units and is recorded explicitly."
        ),
    )
    parser.add_argument(
        "--functional_carrier_count", "--functional-carrier-count", type=int,
        default=DEFAULT_FUNCTIONAL_CARRIER_COUNT,
    )
    parser.add_argument(
        "--functional_carrier_batch_size", "--functional-carrier-batch-size", type=int, default=2
    )
    parser.add_argument(
        "--functional_influence_scale", "--functional-influence-scale", type=float,
        default=DEFAULT_FUNCTIONAL_INFLUENCE_SCALE,
    )
    parser.add_argument(
        "--functional_clip_norm", "--functional-clip-norm", type=float,
        default=DEFAULT_FUNCTIONAL_CLIP_NORM,
    )
    parser.add_argument(
        "--functional_contribution_unit", "--functional-contribution-unit",
        choices=("qa", "identity"), default=DEFAULT_FUNCTIONAL_CONTRIBUTION_UNIT,
        help="Privacy/attribution unit used before contribution clipping.",
    )
    parser.add_argument(
        "--functional_local_steps", "--functional-local-steps", type=int,
        default=DEFAULT_FUNCTIONAL_LOCAL_STEPS,
        help=(
            "Per-private-unit local AdamW steps from fixed M0. Zero retains the "
            "single-gradient proxy and requires contribution_unit=qa."
        ),
    )
    parser.add_argument(
        "--functional_local_lr", "--functional-local-lr", type=float,
        default=DEFAULT_FUNCTIONAL_LOCAL_LR,
    )
    parser.add_argument(
        "--functional_local_delta_scale", "--functional-local-delta-scale", type=float,
        default=DEFAULT_FUNCTIONAL_LOCAL_DELTA_SCALE,
    )
    parser.add_argument(
        "--functional_local_method", "--functional-local-method",
        choices=("legacy_objective", "smfa"),
        default=DEFAULT_FUNCTIONAL_LOCAL_METHOD,
        help=(
            "Private identity contribution estimator. SMFA trains an IDK/retain "
            "memory adapter and sculpts it with the fixed public retain anchor."
        ),
    )
    parser.add_argument(
        "--functional_smfa_aggregate_before_sculpt",
        "--functional-smfa-aggregate-before-sculpt",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FUNCTIONAL_SMFA_AGGREGATE_BEFORE_SCULPT,
        help=(
            "For SMFA identity contributions, privately aggregate clipped local "
            "memory deltas first and apply the fixed public retain-anchor sculpt "
            "only after the Gaussian release as DP post-processing."
        ),
    )
    parser.add_argument(
        "--functional_center_on_public_retain",
        "--functional-center-on-public-retain",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FUNCTIONAL_CENTER_ON_PUBLIC_RETAIN,
        help=(
            "Separate the SMFA local objective into a private forget/IDK update "
            "and one fixed public retain-only baseline. Release only the private "
            "update, then add the public baseline once before SMFA sculpting."
        ),
    )
    parser.add_argument(
        "--functional_postrelease_retain_blend",
        "--functional-postrelease-retain-blend",
        type=float,
        default=DEFAULT_FUNCTIONAL_POSTRELEASE_RETAIN_BLEND,
        help=(
            "Fixed public retain-anchor delta added after SMFA sculpting. This "
            "is post-processing of every release and does not inspect forget data."
        ),
    )
    parser.add_argument(
        "--functional_postrelease_retain_repair_steps",
        "--functional-postrelease-retain-repair-steps",
        type=int,
        default=DEFAULT_FUNCTIONAL_POSTRELEASE_RETAIN_REPAIR_STEPS,
        help=(
            "Number of public retain-only CE steps applied to each reconstructed "
            "student after fixed SMFA post-processing."
        ),
    )
    parser.add_argument(
        "--functional_postrelease_retain_repair_lr",
        "--functional-postrelease-retain-repair-lr",
        type=float,
        default=DEFAULT_FUNCTIONAL_POSTRELEASE_RETAIN_REPAIR_LR,
    )
    parser.add_argument(
        "--functional_student_solver", "--functional-student-solver",
        choices=("adam", "ridge"), default=DEFAULT_FUNCTIONAL_STUDENT_SOLVER,
    )
    parser.add_argument(
        "--functional_ridge", "--functional-ridge", type=float,
        default=DEFAULT_FUNCTIONAL_RIDGE,
    )
    parser.add_argument(
        "--functional_whiten", "--functional-whiten",
        action=argparse.BooleanOptionalAction, default=DEFAULT_FUNCTIONAL_WHITEN,
        help=(
            "Apply the public pseudoinverse of the carrier/LoRA basis before "
            "private-unit clipping, making the functional channel well conditioned."
        ),
    )
    parser.add_argument(
        "--functional_projection_dim", "--functional-projection-dim",
        type=int, default=DEFAULT_FUNCTIONAL_PROJECTION_DIM,
        help=(
            "Dimension of the public retain-proxy functional subspace released by "
            "the Gaussian mechanism. Zero disables projection and restores the "
            "legacy full-dimensional release."
        ),
    )
    parser.add_argument(
        "--functional_projection_records", "--functional-projection-records",
        type=int, default=DEFAULT_FUNCTIONAL_PROJECTION_RECORDS,
        help=(
            "Maximum public proxy directions used to construct the DP release "
            "subspace. Matched-local mode uses deterministic groups of public "
            "retain identities; legacy mode uses individual QA records."
        ),
    )
    parser.add_argument(
        "--functional_projection_matched_local",
        "--functional-projection-matched-local",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FUNCTIONAL_PROJECTION_MATCHED_LOCAL,
        help=(
            "Build the public projection from multi-step local updates on public "
            "proxy QAs, matching the private contribution estimator instead of "
            "using one-step proxy gradients."
        ),
    )
    parser.add_argument(
        "--functional_projection_cache",
        "--functional-projection-cache",
        default=DEFAULT_FUNCTIONAL_PROJECTION_CACHE,
        help=(
            "Optional public-only proxy cache produced by an earlier compatible "
            "run. It avoids retraining the public projection proxies."
        ),
    )
    parser.add_argument("--dp_carrier_steps", "--dp-carrier-steps", type=int, default=DEFAULT_DP_CARRIER_STEPS)
    parser.add_argument(
        "--dp_noise_repeats", "--dp-noise-repeats", type=int,
        default=DEFAULT_DP_NOISE_REPEATS,
    )

    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=1)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=40)
    parser.add_argument("--sample_answers", "--sample-answers", type=int, default=12)
    add_capacity_arguments(parser)
    add_rq1_arguments(parser)
    return parser.parse_args()


def validate_args(args):
    positive = {
        "adapter_rank": args.adapter_rank,
        "exposure_batch_size": args.exposure_batch_size,
        "teacher_batch_size": args.teacher_batch_size,
        "retain_identity_count": args.retain_identity_count,
        "max_qa_per_identity": args.max_qa_per_identity,
        "eval_qa_per_identity": args.eval_qa_per_identity,
        "carrier_resolution": args.carrier_resolution,
        "carrier_batch_size": args.carrier_batch_size,
        "functional_carrier_count": args.functional_carrier_count,
        "functional_carrier_batch_size": args.functional_carrier_batch_size,
        "dp_carrier_steps": args.dp_carrier_steps,
        "dp_noise_repeats": args.dp_noise_repeats,
        "eval_batch_size": args.eval_batch_size,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError("{} must be positive".format(name))
    if args.teacher_batch_size < 2:
        raise ValueError("teacher_batch_size must be at least 2 for forget/retain mixing")
    if (
        args.exposure_steps < 0
        or args.teacher_steps < 0
        or args.smfa_retain_steps < 0
        or args.carrier_steps < 0
    ):
        raise ValueError("training step counts must be non-negative")
    if not 0.0 < args.forget_fraction < 1.0:
        raise ValueError("forget_fraction must lie strictly between zero and one")
    if not 0.0 < args.dp_delta < 1.0:
        raise ValueError("dp_delta must lie in (0, 1)")
    if args.dp_public_denominator < 0:
        raise ValueError("dp_public_denominator cannot be negative")
    if args.functional_influence_scale <= 0 or args.functional_clip_norm <= 0:
        raise ValueError("functional influence scale and clip norm must be positive")
    if args.functional_local_steps < 0:
        raise ValueError("functional_local_steps cannot be negative")
    if args.functional_local_method == "smfa" and args.functional_local_steps == 0:
        raise ValueError("SMFA private contributions require positive functional_local_steps")
    if (
        args.functional_smfa_aggregate_before_sculpt
        and args.functional_local_method != "smfa"
    ):
        raise ValueError("aggregate-before-sculpt is only defined for SMFA contributions")
    if args.functional_center_on_public_retain and not (
        args.functional_smfa_aggregate_before_sculpt
        and args.functional_local_method == "smfa"
    ):
        raise ValueError(
            "public-retain centering requires aggregate-before-sculpt SMFA"
        )
    if args.functional_postrelease_retain_blend < 0:
        raise ValueError("functional_postrelease_retain_blend cannot be negative")
    if args.functional_postrelease_retain_repair_steps < 0:
        raise ValueError(
            "functional_postrelease_retain_repair_steps cannot be negative"
        )
    if args.functional_postrelease_retain_repair_lr <= 0:
        raise ValueError(
            "functional_postrelease_retain_repair_lr must be positive"
        )
    if (
        args.functional_postrelease_retain_blend > 0
        and args.functional_local_method != "smfa"
    ):
        raise ValueError("post-release retain blending is only defined for SMFA")
    if args.functional_projection_matched_local and args.functional_local_steps == 0:
        raise ValueError("matched-local projection requires positive functional_local_steps")
    if args.functional_local_steps == 0 and args.functional_contribution_unit != "qa":
        raise ValueError("identity contributions require positive functional_local_steps")
    if args.functional_local_lr <= 0 or args.functional_local_delta_scale <= 0:
        raise ValueError("functional local learning rate and delta scale must be positive")
    if args.smfa_retain_lr <= 0 or args.smfa_sculpt_k < 0:
        raise ValueError("SMFA retain learning rate must be positive and sculpt k non-negative")
    if (
        args.teacher_method == "smfa"
        or args.functional_local_method == "smfa"
        or args.run_teacher_comparisons
    ) and args.smfa_retain_steps <= 0:
        raise ValueError("SMFA requires positive smfa_retain_steps")
    if args.functional_projection_dim < 0:
        raise ValueError("functional_projection_dim cannot be negative")
    if args.functional_projection_records <= 0:
        raise ValueError("functional_projection_records must be positive")
    if args.functional_ridge < 0:
        raise ValueError("functional_ridge cannot be negative")
    parse_epsilon_values(args.dp_epsilons)
    validate_capacity_arguments(args)
    validate_rq1_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name, device):
    if name == "float32" or not device.startswith("cuda"):
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device, dtype):
    if device.startswith("cuda") and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def write_jsonl(handle, record):
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def parse_epsilon_values(values):
    parsed = []
    for value in values:
        normalized = str(value).strip().lower()
        epsilon = math.inf if normalized in ("inf", "infinity") else float(value)
        if epsilon <= 0:
            raise ValueError("Every finite epsilon must be positive")
        if epsilon in parsed:
            raise ValueError("Duplicate epsilon arm: {}".format(value))
        parsed.append(epsilon)
    return parsed


def epsilon_tag(epsilon):
    return "clip_only" if math.isinf(epsilon) else ("{:g}".format(epsilon)).replace(".", "p")


def accounted_full_batch_gaussian_epsilon(noise_multiplier, delta):
    if noise_multiplier <= 0:
        return math.inf, None
    return min(
        (
            order / (2.0 * noise_multiplier**2) + math.log(1.0 / delta) / (order - 1),
            order,
        )
        for order in range(2, 129)
    )


def calibrate_full_batch_noise_multiplier(target_epsilon, delta):
    if math.isinf(target_epsilon):
        return 0.0, math.inf, None
    lower, upper = 0.01, 100.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        epsilon, _ = accounted_full_batch_gaussian_epsilon(middle, delta)
        if epsilon > target_epsilon:
            lower = middle
        else:
            upper = middle
    achieved, order = accounted_full_batch_gaussian_epsilon(upper, delta)
    return upper, achieved, order


@dataclass(frozen=True)
class QARecord:
    identity_id: str
    name: str
    image_path: str
    question: str
    answer: str
    keywords: tuple
    qa_index: int
    split: str


def _read_json_or_jsonl(path):
    text = Path(path).read_text()
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else value
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _resolve_image_path(data_root, raw_path):
    root = Path(data_root)
    candidates = [
        root / "SFHQ" / Path(raw_path).name,
        root / raw_path.lstrip("./"),
        Path(raw_path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Missing FIUBench image {}. Download and extract SFHQ.zip under {}.".format(
            Path(raw_path).name, data_root
        )
    )


def _qa_priority(qa, index):
    keywords = qa.get("keywords") or []
    keyword_tokens = sum(len(normalized_text(value).split()) for value in keywords)
    answer_tokens = len(normalized_text(qa.get("answer", "")).split())
    missing_penalty = 1000 if not keywords else 0
    return missing_penalty + keyword_tokens * 10 + answer_tokens, index


def _select_qas(person, count):
    indexed = list(enumerate(person["qa_list"]))
    # Avoid a degenerate subset dominated by the shortest field (usually
    # blood type).  These four topics exist across FIUBench and require the
    # visual identity to select different fictitious facts.
    topic_patterns = (
        ("full name", "complete name"),
        ("occupation", "profession", "for a living"),
        ("blood type", "blood group"),
        ("medical condition", "health issue"),
    )
    selected = []
    used = set()
    for patterns in topic_patterns:
        match = next(
            (
                (index, qa)
                for index, qa in indexed
                if index not in used
                and qa.get("keywords")
                and any(pattern in qa.get("question", "").lower() for pattern in patterns)
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            used.add(match[0])
            if len(selected) == count:
                break
    remaining = [item for item in indexed if item[0] not in used]
    remaining.sort(key=lambda item: _qa_priority(item[1], item[0]))
    selected.extend(remaining[: max(0, count - len(selected))])
    return sorted(selected, key=lambda item: item[0])


def load_fiubench_records(args):
    root = Path(args.data_root)
    rows = _read_json_or_jsonl(root / args.metadata_file)
    if not isinstance(rows, list):
        raise ValueError("FIUBench metadata must be a JSON array or JSONL records")
    rows = rows[: args.official_identity_limit]
    splits = json.loads((root / args.split_file).read_text())
    if args.forget_split not in splits:
        raise KeyError("{} is absent from {}".format(args.forget_split, root / args.split_file))

    available = {str(row["unique_id"]): row for row in rows}
    forget_ids = [str(value) for value in splits[args.forget_split] if str(value) in available]
    all_forget_ids = {
        str(value)
        for key in ("forget1", "forget5", "forget10")
        for value in splits.get(key, [])
    }
    # The Hub file combines several annotation vintages.  The first 138 rows
    # have official keyword labels used by FIUBench's exact-match evaluation,
    # while many later rows leave `keywords` empty.  Retain identities are
    # sampled only from records with enough official keyword-labelled QAs so
    # forget and retain accuracy use the same benchmark metric.
    retain_ids = [
        str(row["unique_id"])
        for row in rows
        if str(row["unique_id"]) not in all_forget_ids
        and sum(bool(qa.get("keywords")) for qa in row["qa_list"])
        >= args.max_qa_per_identity
    ]
    rng = random.Random(args.seed + 700)
    rng.shuffle(retain_ids)
    retain_ids = sorted(retain_ids[: args.retain_identity_count])
    if not forget_ids:
        raise RuntimeError("The selected official forget split has no identities in the metadata subset")
    if len(retain_ids) < args.retain_identity_count:
        raise RuntimeError("Not enough official retain identities for the requested subset")

    def records_for(identity_ids, split_name):
        output = []
        per_identity = {}
        for identity_id in identity_ids:
            person = available[identity_id]
            image_path = _resolve_image_path(args.data_root, person["image_path"])
            selected = _select_qas(person, args.max_qa_per_identity)
            per_identity[identity_id] = []
            for qa_index, qa in selected:
                record = QARecord(
                    identity_id=identity_id,
                    name=str(person["name"]),
                    image_path=image_path,
                    question=str(qa["question"]).strip(),
                    answer=str(qa["answer"]).strip(),
                    keywords=tuple(str(value) for value in (qa.get("keywords") or [])),
                    qa_index=int(qa_index),
                    split=split_name,
                )
                output.append(record)
                per_identity[identity_id].append((record, qa))
        return output, per_identity

    forget_train, forget_grouped = records_for(forget_ids, "forget")
    retain_train, retain_grouped = records_for(retain_ids, "retain")

    def evaluation_records(grouped):
        output = []
        for identity_id in sorted(grouped):
            for record, qa in grouped[identity_id][: args.eval_qa_per_identity]:
                paraphrases = qa.get("paraphrased_question") or []
                question = str(paraphrases[0]).strip() if paraphrases else record.question
                output.append(replace(record, question=question))
        return output

    metadata = {
        "official_identity_limit": len(rows),
        "official_forget_split": args.forget_split,
        "forget_identity_ids": forget_ids,
        "forget_identity_names": [available[value]["name"] for value in forget_ids],
        "retain_identity_ids": retain_ids,
        "retain_identity_names": [available[value]["name"] for value in retain_ids],
        "all_official_forget_identities_excluded_from_retain": len(all_forget_ids),
    }
    return (
        forget_train,
        retain_train,
        evaluation_records(forget_grouped),
        evaluation_records(retain_grouped),
        metadata,
    )


def open_image(record):
    with Image.open(record.image_path) as image:
        image = image.convert("RGB")
        # The official images are 1024x1024.  LLaVA-OneVision would otherwise
        # create ten 384px tiles and a ~7.4k-token sequence per QA, which is
        # unnecessary for identity-level face evidence and makes answer-token
        # training prohibitively memory-heavy.  One square global tile plus
        # the processor's base tile preserves the full face without cropping.
        return image.resize(
            (FIUBENCH_IMAGE_RESOLUTION, FIUBENCH_IMAGE_RESOLUTION),
            Image.Resampling.LANCZOS,
        )


def find_multimodal_projector(model):
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("multi_modal_projector")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one multi_modal_projector, found {}: {}".format(
                len(matches), [name for name, _ in matches]
            )
        )
    return matches[0]


def replace_module(root, dotted_name, replacement):
    parent = root
    pieces = dotted_name.split(".")
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    setattr(parent, pieces[-1], replacement)


class ResidualProjectorAdapter(nn.Module):
    """Frozen native projector plus one low-rank residual in LM space."""

    def __init__(self, base_projector, hidden_size, rank, alpha, seed):
        super().__init__()
        self.base_projector = base_projector
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Parameter(torch.empty(rank, hidden_size, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(hidden_size, rank, dtype=torch.float32))
        self.enabled = True
        self.capture_base = False
        self.captured_base = None
        self.reset_adapter(seed)

    def reset_adapter(self, seed):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        values = torch.randn(self.rank, self.hidden_size, generator=generator)
        values.mul_(1.0 / math.sqrt(self.hidden_size))
        with torch.no_grad():
            self.lora_a.copy_(values.to(self.lora_a.device))
            self.lora_b.zero_()

    def adapter_state(self):
        return {
            "lora_a": self.lora_a.detach().cpu().clone(),
            "lora_b": self.lora_b.detach().cpu().clone(),
        }

    def load_adapter_state(self, state):
        with torch.no_grad():
            self.lora_a.copy_(state["lora_a"].to(self.lora_a.device, torch.float32))
            self.lora_b.copy_(state["lora_b"].to(self.lora_b.device, torch.float32))

    def delta(self, base_output, state=None):
        if state is None:
            a, b = self.lora_a, self.lora_b
        else:
            a = state["lora_a"].to(base_output.device, torch.float32)
            b = state["lora_b"].to(base_output.device, torch.float32)
        return F.linear(F.linear(base_output.float(), a), b) * self.scale

    def forward(self, image_features):
        base_output = self.base_projector(image_features)
        if self.capture_base:
            self.captured_base = base_output.detach()
        if not self.enabled:
            return base_output
        return base_output + self.delta(base_output).to(base_output.dtype)


@contextlib.contextmanager
def adapter_mode(adapter, enabled, state=None):
    old_enabled = adapter.enabled
    old_state = adapter.adapter_state() if state is not None else None
    if state is not None:
        adapter.load_adapter_state(state)
    adapter.enabled = bool(enabled)
    try:
        yield
    finally:
        adapter.enabled = old_enabled
        if old_state is not None:
            adapter.load_adapter_state(old_state)


def build_prompt(processor, question):
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        except (ValueError, TypeError, AttributeError):
            pass
    return "USER: <image>\n{} ASSISTANT:".format(question)


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def prepare_prompt_inputs(processor, records, device):
    images = [open_image(record) for record in records]
    prompts = [build_prompt(processor, record.question) for record in records]
    inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt")
    return move_batch(dict(inputs), device), prompts


def prepare_supervised_inputs(processor, records, answers, device):
    images = [open_image(record) for record in records]
    prompts = [build_prompt(processor, record.question) for record in records]
    eos = processor.tokenizer.eos_token or ""
    texts = [prompt + str(answer).strip() + eos for prompt, answer in zip(prompts, answers)]
    inputs = move_batch(
        dict(processor(images=images, text=texts, padding=True, return_tensors="pt")), device
    )
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    for row, (prompt, text) in enumerate(zip(prompts, texts)):
        prompt_ids = processor.tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = processor.tokenizer(text, add_special_tokens=False).input_ids
        answer_tokens = len(full_ids) - len(prompt_ids)
        if answer_tokens <= 0:
            raise RuntimeError("The supervised answer produced no target tokens")
        # LLaVA-OneVision expands the single <image> token into thousands of
        # visual placeholder tokens inside the processor.  Expansion changes
        # both full and prompt lengths equally, so the response begins exactly
        # `answer_tokens` before the end of the non-padding sequence.
        valid_tokens = int(inputs["attention_mask"][row].sum().item())
        response_start = valid_tokens - answer_tokens
        labels[row, :response_start] = -100
    inputs["labels"] = labels
    return inputs


def sequence_losses(logits, labels):
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe = shift_labels.masked_fill(~mask, 0)
    losses = F.cross_entropy(
        shift_logits.transpose(1, 2), safe, reduction="none"
    ) * mask
    counts = mask.sum(dim=1).clamp_min(1)
    return losses.sum(dim=1) / counts


def sequence_unlikelihood(logits, labels):
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    safe = shift_labels.masked_fill(~mask, 0)
    true_probabilities = F.softmax(shift_logits, dim=-1).gather(
        -1, safe.unsqueeze(-1)
    ).squeeze(-1)
    values = -torch.log1p(-true_probabilities.clamp(max=1.0 - 1e-6)) * mask
    return values.sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def sequence_kl(current_logits, base_logits, labels, temperature):
    mask = labels[:, 1:].ne(-100)
    current = current_logits[:, :-1].float() / temperature
    base = base_logits[:, :-1].float() / temperature
    per_token = F.kl_div(
        F.log_softmax(current, dim=-1), F.softmax(base, dim=-1), reduction="none"
    ).sum(dim=-1)
    return (per_token * mask).sum() / mask.sum().clamp_min(1) * temperature**2


class RecordStream:
    def __init__(self, records, seed):
        self.records = list(records)
        self.rng = random.Random(seed)
        self.order = []
        self.position = 0

    def next(self, count):
        output = []
        while len(output) < count:
            if self.position >= len(self.order):
                self.order = list(range(len(self.records)))
                self.rng.shuffle(self.order)
                self.position = 0
            take = min(count - len(output), len(self.order) - self.position)
            output.extend(
                self.records[index]
                for index in self.order[self.position : self.position + take]
            )
            self.position += take
        return output


def train_exposure_model(
    model, adapter, processor, records, args, device, dtype, log_path
):
    """Learn the selected fictitious FIUBench facts and return fixed M0."""
    stream = RecordStream(records, args.seed + 101)
    optimizer = torch.optim.AdamW([adapter.lora_b], lr=args.exposure_lr, weight_decay=0.0)
    history = []
    adapter.enabled = True
    model.eval()
    with open(log_path, "w") as log_handle:
        for step in range(1, args.exposure_steps + 1):
            batch = stream.next(args.exposure_batch_size)
            inputs = prepare_supervised_inputs(
                processor, batch, [record.answer for record in batch], device
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                outputs = model(**inputs, use_cache=False)
                per_record = sequence_losses(outputs.logits, inputs["labels"])
                loss = per_record.mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([adapter.lora_b], args.gradient_clip)
            optimizer.step()
            record = {
                "step": step,
                "answer_token_cross_entropy": float(loss.detach().cpu()),
                "answer_token_geometric_probability": float(torch.exp(-loss.detach()).cpu()),
                "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            }
            history.append(record)
            write_jsonl(log_handle, record)
            if args.print_progress and (step == 1 or step % 10 == 0 or step == args.exposure_steps):
                print("exposure {}".format(json.dumps(record)), flush=True)
    return adapter.adapter_state(), history


def train_unlearned_teacher(
    model,
    adapter,
    processor,
    m0_state,
    forget_records,
    retain_records,
    args,
    device,
    dtype,
    log_path,
):
    adapter.load_adapter_state(m0_state)
    adapter.enabled = True
    forget_stream = RecordStream(forget_records, args.seed + 201)
    retain_stream = RecordStream(retain_records, args.seed + 202)
    forget_count = max(1, int(round(args.teacher_batch_size * args.forget_fraction)))
    forget_count = min(forget_count, args.teacher_batch_size - 1)
    retain_count = args.teacher_batch_size - forget_count
    optimizer = torch.optim.AdamW([adapter.lora_b], lr=args.teacher_lr, weight_decay=0.0)
    history = []
    model.eval()
    with open(log_path, "w") as log_handle:
        for step in range(1, args.teacher_steps + 1):
            forget_batch = forget_stream.next(forget_count)
            retain_batch = retain_stream.next(retain_count)
            combined = forget_batch + retain_batch
            answers = [args.forget_answer] * forget_count + [
                record.answer for record in retain_batch
            ]
            target_inputs = prepare_supervised_inputs(processor, combined, answers, device)
            true_forget_inputs = prepare_supervised_inputs(
                processor, forget_batch, [record.answer for record in forget_batch], device
            )
            with torch.no_grad(), adapter_mode(adapter, True, m0_state), autocast_context(device, dtype):
                m0_outputs = model(**target_inputs, use_cache=False)

            adapter.load_adapter_state(m0_state if step == 1 else adapter.adapter_state())
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                outputs = model(**target_inputs, use_cache=False)
                target_losses = sequence_losses(outputs.logits, target_inputs["labels"])
                forget_unknown_ce = target_losses[:forget_count].mean()
                retain_ce = target_losses[forget_count:].mean()
                retain_kl = sequence_kl(
                    outputs.logits[forget_count:],
                    m0_outputs.logits[forget_count:],
                    target_inputs["labels"][forget_count:],
                    args.kl_temperature,
                )
                true_outputs = model(**true_forget_inputs, use_cache=False)
                forget_unlikelihood = sequence_unlikelihood(
                    true_outputs.logits, true_forget_inputs["labels"]
                ).mean()
                loss = (
                    args.forget_ce_weight * forget_unknown_ce
                    + args.forget_unlikelihood_weight * forget_unlikelihood
                    + args.retain_ce_weight * retain_ce
                    + args.retain_kl_weight * retain_kl
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([adapter.lora_b], args.gradient_clip)
            optimizer.step()
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "forget_unknown_token_ce": float(forget_unknown_ce.detach().cpu()),
                "forget_original_answer_unlikelihood": float(forget_unlikelihood.detach().cpu()),
                "retain_answer_token_ce": float(retain_ce.detach().cpu()),
                "retain_kl_to_m0": float(retain_kl.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            }
            history.append(record)
            write_jsonl(log_handle, record)
            if args.print_progress and (step == 1 or step % 10 == 0 or step == args.teacher_steps):
                print("teacher {}".format(json.dumps(record)), flush=True)
    return adapter.adapter_state(), history


def sculpt_smfa_adapter_state(m0_state, memory_state, retain_anchor_state, k):
    """Apply the official SMFA sign-conflict/magnitude rule to this adapter.

    The upstream implementation sculpts the full LoRA ``delta_W`` of many
    language-model layers.  This experiment deliberately uses a matched
    parameter budget: only the existing multimodal-projector residual
    coefficient ``lora_b`` is trainable with a fixed ``lora_a``.  We apply the
    official rule in the effective residual-weight space ``delta_B @ lora_a``,
    then least-squares project the sculpted matrix back into the fixed-rank
    parameterization.  The distinction is recorded in every result and this
    function must not be described as a native full-model reproduction.
    """
    base = m0_state["lora_b"].float()
    fixed_a = m0_state["lora_a"].float()
    memory_delta_b = memory_state["lora_b"].float() - base
    retain_delta_b = retain_anchor_state["lora_b"].float() - base
    memory_delta = memory_delta_b @ fixed_a
    retain_delta = retain_delta_b @ fixed_a
    memory_norm = memory_delta.norm()
    retain_norm = retain_delta.norm()
    norm_ratio = memory_norm / retain_norm.clamp_min(1e-12)
    conflict = (retain_delta * memory_delta) < 0
    relative = float(k) * norm_ratio * retain_delta.abs() < memory_delta.abs()
    mask = conflict & relative
    sculpted_effective_delta = memory_delta.clone()
    sculpted_effective_delta[mask] = 0.0

    # Elementwise SMFA sculpting can leave the fixed-A rank-r subspace.  The
    # matched projector can only deploy B @ A, so use the minimum-norm
    # least-squares projection back to B and report the approximation error.
    gram = fixed_a @ fixed_a.T
    ridge = torch.finfo(gram.dtype).eps * max(float(gram.diag().mean()), 1.0)
    inverse = torch.linalg.inv(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    )
    sculpted_delta_b = sculpted_effective_delta @ fixed_a.T @ inverse
    deployed_effective_delta = sculpted_delta_b @ fixed_a
    projection_relative_error = (
        (deployed_effective_delta - sculpted_effective_delta).norm()
        / sculpted_effective_delta.norm().clamp_min(1e-12)
    )
    state = {key: value.detach().cpu().clone() for key, value in m0_state.items()}
    state["lora_b"] = base + sculpted_delta_b
    total = max(mask.numel(), 1)
    return state, {
        "implementation": "matched_projector_SMFA",
        "upstream_source": "On_FIUBench/third_party/S-MLLMUn/unlearning_methods/SMFA_sculpt.py",
        "native_full_model_reproduction": False,
        "sculpt_space": "effective_projector_residual_weight_delta_B_times_fixed_A",
        "deployment_space": "shared_fixed_lora_a_projector_coefficient_lora_b",
        "sculpt_k": float(k),
        "memory_delta_l2": float(memory_norm),
        "retain_anchor_delta_l2": float(retain_norm),
        "memory_to_retain_norm_ratio": float(norm_ratio),
        "sign_conflict_fraction": float(conflict.float().mean()),
        "masked_fraction": float(mask.sum().item() / total),
        "sculpted_effective_delta_l2_before_rank_projection": float(
            sculpted_effective_delta.norm()
        ),
        "deployed_effective_delta_l2": float(deployed_effective_delta.norm()),
        "rank_projection_relative_error": float(projection_relative_error),
    }


def add_fixed_retain_anchor_delta(
    state, m0_state, retain_anchor_state, blend_weight
):
    """Add a fixed public retain direction as release post-processing."""
    if blend_weight <= 0:
        return state, {
            "enabled": False,
            "blend_weight": float(blend_weight),
        }
    output = {key: value.detach().cpu().clone() for key, value in state.items()}
    base = m0_state["lora_b"].float()
    retain_delta = retain_anchor_state["lora_b"].float() - base
    before_delta = output["lora_b"].float() - base
    output["lora_b"] = base + before_delta + blend_weight * retain_delta
    return output, {
        "enabled": True,
        "source": "fixed_public_smfa_retain_anchor",
        "blend_weight": float(blend_weight),
        "pre_blend_update_l2": float(before_delta.norm()),
        "retain_anchor_delta_l2": float(retain_delta.norm()),
        "added_retain_delta_l2": float((blend_weight * retain_delta).norm()),
        "post_blend_update_l2": float((output["lora_b"].float() - base).norm()),
    }


def train_public_postrelease_retain_repair(
    model,
    adapter,
    processor,
    state,
    retain_records,
    args,
    device,
    dtype,
):
    """Repair public retain behavior without accessing a private forget record."""
    steps = int(args.functional_postrelease_retain_repair_steps)
    if steps == 0:
        return state, {
            "enabled": False,
            "steps": 0,
            "learning_rate": args.functional_postrelease_retain_repair_lr,
        }
    adapter.load_adapter_state(state)
    adapter.enabled = True
    model.eval()
    stream = RecordStream(retain_records, args.seed + 52_001)
    optimizer = torch.optim.AdamW(
        [adapter.lora_b],
        lr=args.functional_postrelease_retain_repair_lr,
        weight_decay=0.0,
    )
    history = []
    for step in range(1, steps + 1):
        batch = stream.next(args.teacher_batch_size)
        inputs = prepare_supervised_inputs(
            processor, batch, [record.answer for record in batch], device
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            outputs = model(**inputs, use_cache=False)
            loss = sequence_losses(outputs.logits, inputs["labels"]).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [adapter.lora_b], args.gradient_clip
        )
        optimizer.step()
        record = {
            "step": step,
            "retain_answer_token_ce": float(loss.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }
        history.append(record)
        if args.print_progress and (
            step == 1 or step % 10 == 0 or step == steps
        ):
            print(
                "postrelease_retain_repair {}".format(json.dumps(record)),
                flush=True,
            )
    repaired_state = adapter.adapter_state()
    before = state["lora_b"].float()
    after = repaired_state["lora_b"].float()
    return repaired_state, {
        "enabled": True,
        "uses_private_forget_records": False,
        "source": "fixed_public_retain_QAs",
        "objective": "retain_answer_cross_entropy",
        "steps": steps,
        "learning_rate": args.functional_postrelease_retain_repair_lr,
        "repair_delta_l2": float((after - before).norm()),
        "last_training_log": history[-1],
    }


def train_smfa_matched_teacher(
    model,
    adapter,
    processor,
    m0_state,
    forget_records,
    retain_records,
    args,
    device,
    dtype,
    memory_log_path,
    retain_log_path,
):
    """Matched-projector adaptation of the official SMFA teacher.

    SMFA trains a memory-forgetting adapter on IDK targets plus correct retain
    answers and a separate retaining-anchor adapter on retain answers.  It then
    removes memory-update coefficients that conflict with a sufficiently large
    retaining-anchor direction.  FIUBench has no meaningful text-only path for
    the projector-only adapter, so this matched comparison uses the multimodal
    branch only.
    """
    adapter.load_adapter_state(m0_state)
    adapter.enabled = True
    model.eval()
    forget_stream = RecordStream(forget_records, args.seed + 2_301)
    retain_stream = RecordStream(retain_records, args.seed + 2_302)
    forget_count = max(1, int(round(args.teacher_batch_size * args.forget_fraction)))
    forget_count = min(forget_count, args.teacher_batch_size - 1)
    retain_count = args.teacher_batch_size - forget_count
    optimizer = torch.optim.AdamW(
        [adapter.lora_b], lr=args.teacher_lr, weight_decay=0.0
    )
    memory_history = []
    memory_started = time.perf_counter()
    with open(memory_log_path, "w") as log_handle:
        for step in range(1, args.teacher_steps + 1):
            forget_batch = forget_stream.next(forget_count)
            retain_batch = retain_stream.next(retain_count)
            records = forget_batch + retain_batch
            answers = [args.forget_answer] * forget_count + [
                record.answer for record in retain_batch
            ]
            inputs = prepare_supervised_inputs(
                processor, records, answers, device
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                outputs = model(**inputs, use_cache=False)
                losses = sequence_losses(outputs.logits, inputs["labels"])
                forget_idk_ce = losses[:forget_count].mean()
                retain_ce = losses[forget_count:].mean()
                loss = forget_idk_ce + retain_ce
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [adapter.lora_b], args.gradient_clip
            )
            optimizer.step()
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "forget_idk_token_ce": float(forget_idk_ce.detach().cpu()),
                "retain_answer_token_ce": float(retain_ce.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            }
            memory_history.append(record)
            write_jsonl(log_handle, record)
            if args.print_progress and (
                step == 1 or step % 10 == 0 or step == args.teacher_steps
            ):
                print("smfa_memory {}".format(json.dumps(record)), flush=True)
    memory_state = adapter.adapter_state()
    memory_seconds = time.perf_counter() - memory_started

    adapter.load_adapter_state(m0_state)
    retain_stream = RecordStream(retain_records, args.seed + 2_303)
    optimizer = torch.optim.AdamW(
        [adapter.lora_b], lr=args.smfa_retain_lr, weight_decay=0.0
    )
    retain_history = []
    retain_started = time.perf_counter()
    with open(retain_log_path, "w") as log_handle:
        for step in range(1, args.smfa_retain_steps + 1):
            batch = retain_stream.next(args.teacher_batch_size)
            inputs = prepare_supervised_inputs(
                processor, batch, [record.answer for record in batch], device
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                outputs = model(**inputs, use_cache=False)
                loss = sequence_losses(outputs.logits, inputs["labels"]).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [adapter.lora_b], args.gradient_clip
            )
            optimizer.step()
            record = {
                "step": step,
                "retain_answer_token_ce": float(loss.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            }
            retain_history.append(record)
            write_jsonl(log_handle, record)
            if args.print_progress and (
                step == 1 or step % 10 == 0 or step == args.smfa_retain_steps
            ):
                print("smfa_retain_anchor {}".format(json.dumps(record)), flush=True)
    retain_anchor_state = adapter.adapter_state()
    retain_seconds = time.perf_counter() - retain_started
    sculpted_state, sculpt_audit = sculpt_smfa_adapter_state(
        m0_state,
        memory_state,
        retain_anchor_state,
        args.smfa_sculpt_k,
    )
    sculpt_audit["memory_training_seconds"] = memory_seconds
    sculpt_audit["retain_anchor_training_seconds"] = retain_seconds
    adapter.load_adapter_state(m0_state)
    return (
        sculpted_state,
        memory_state,
        retain_anchor_state,
        memory_history,
        retain_history,
        sculpt_audit,
    )


class SpectralCarrierSampler:
    """Deterministic RGB 1/f^alpha images independent of FIUBench."""

    def __init__(self, resolution, alpha, seed):
        self.resolution = int(resolution)
        self.alpha = float(alpha)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        fy = torch.fft.fftfreq(self.resolution).view(-1, 1)
        fx = torch.fft.rfftfreq(self.resolution).view(1, -1)
        radius = torch.sqrt(fx.square() + fy.square())
        radius[0, 0] = 1.0
        self.amplitude = radius.pow(-self.alpha / 2.0)
        self.amplitude[0, 0] = 0.0

    def sample(self, batch_size):
        shape = (batch_size, 3, self.resolution, self.resolution // 2 + 1)
        real = torch.randn(*shape, generator=self.generator)
        imag = torch.randn(*shape, generator=self.generator)
        spectrum = torch.complex(real, imag) * self.amplitude
        images = torch.fft.irfft2(
            spectrum, s=(self.resolution, self.resolution), norm="ortho"
        )
        flat = images.flatten(2)
        low = flat.amin(dim=-1).view(batch_size, 3, 1, 1)
        high = flat.amax(dim=-1).view(batch_size, 3, 1, 1)
        images = (images - low) / (high - low).clamp_min(1e-6)
        arrays = (images.permute(0, 2, 3, 1) * 255.0).round().byte().numpy()
        return [Image.fromarray(array, mode="RGB") for array in arrays]


def prepare_image_inputs(processor, images, device):
    inputs = processor.image_processor(images=list(images), return_tensors="pt")
    return move_batch(dict(inputs), device)


def capture_base_projector_output(model, adapter, processor, images, device, dtype):
    image_inputs = prepare_image_inputs(processor, images, device)
    adapter.capture_base = True
    adapter.captured_base = None
    try:
        with torch.no_grad(), adapter_mode(adapter, False), autocast_context(device, dtype):
            kwargs = {"pixel_values": image_inputs["pixel_values"]}
            if "image_sizes" in image_inputs:
                kwargs["image_sizes"] = image_inputs["image_sizes"]
            model.get_image_features(**kwargs)
        if adapter.captured_base is None:
            raise RuntimeError("The multimodal projector did not capture carrier features")
        return adapter.captured_base
    finally:
        adapter.capture_base = False
        adapter.captured_base = None


@torch.no_grad()
def sample_fixed_pooled_carrier_bank(model, adapter, processor, args, device, dtype):
    sampler = SpectralCarrierSampler(
        args.carrier_resolution, args.spectral_alpha, args.carrier_seed + 4_000_000
    )
    outputs = []
    remaining = args.functional_carrier_count
    while remaining > 0:
        count = min(remaining, args.functional_carrier_batch_size)
        base_output = capture_base_projector_output(
            model, adapter, processor, sampler.sample(count), device, dtype
        )
        outputs.append(base_output.float().mean(dim=-2))
        remaining -= count
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def prepare_functional_channel(pooled_carrier_base, adapter, args):
    """Return a public, optionally whitened carrier feature channel.

    The adapter functional map is linear in trainable ``lora_b`` with fixed
    basis Z = carrier @ A^T.  A minimally square Z was ill-conditioned in the
    original FIUBench experiment: fitting its outputs did not reliably recover
    the intended parameter direction.  With more public carrier rows we can
    left-precondition both the public carrier features and every subsequently
    released target by pinv(Z), yielding Z_white ~= I.  This operation is fixed
    before inspecting any private contribution and therefore does not consume
    privacy budget.
    """
    raw_basis = F.linear(
        pooled_carrier_base.float(), adapter.lora_a.detach().float()
    )
    singular_values = torch.linalg.svdvals(raw_basis)
    threshold = max(float(singular_values.max().cpu()) * 1e-6, 1e-12)
    numerical_rank = int((singular_values > threshold).sum().item())
    raw_condition = float(
        (singular_values.max() / singular_values.min().clamp_min(1e-12)).cpu()
    )
    transform = None
    functional_base = pooled_carrier_base.float()
    if args.functional_whiten:
        transform = torch.linalg.pinv(raw_basis, rtol=1e-6)
        functional_base = transform @ pooled_carrier_base.float()
    functional_basis = F.linear(
        functional_base.float(), adapter.lora_a.detach().float()
    )
    functional_singular_values = torch.linalg.svdvals(functional_basis)
    functional_condition = float(
        (
            functional_singular_values.max()
            / functional_singular_values.min().clamp_min(1e-12)
        ).cpu()
    )
    identity_error = None
    if args.functional_whiten:
        identity = torch.eye(
            adapter.rank, device=functional_basis.device, dtype=functional_basis.dtype
        )
        identity_error = float(
            ((functional_basis - identity).norm() / identity.norm()).cpu()
        )
    return functional_base.detach(), transform, {
        "raw_rows": int(raw_basis.shape[0]),
        "rank_columns": int(raw_basis.shape[1]),
        "raw_numerical_rank": numerical_rank,
        "raw_condition_number": raw_condition,
        "raw_singular_value_min": float(singular_values.min().cpu()),
        "raw_singular_value_max": float(singular_values.max().cpu()),
        "whitening_enabled": bool(args.functional_whiten),
        "functional_rows": int(functional_basis.shape[0]),
        "functional_condition_number": functional_condition,
        "identity_relative_error_after_whitening": identity_error,
    }


def functional_target_metrics(target, strong_teacher_target):
    target_vector = target.detach().float().flatten()
    reference_vector = strong_teacher_target.detach().float().flatten()
    reference_norm = reference_vector.norm().clamp_min(1e-12)
    cosine = float(F.cosine_similarity(target_vector, reference_vector, dim=0).cpu())
    return {
        "target_l2": float(target_vector.norm().cpu()),
        "strong_teacher_target_l2": float(reference_norm.cpu()),
        "cosine_to_strong_teacher_target": max(-1.0, min(1.0, cosine)),
        "relative_l2_to_strong_teacher_target": float(
            ((target_vector - reference_vector).norm() / reference_norm).cpu()
        ),
    }


def encode_functional_target(target, projection_basis):
    if projection_basis is None:
        return target
    return projection_basis @ target.float().flatten()


def decode_functional_target(released_value, projection_basis, target_shape):
    if projection_basis is None:
        return released_value
    return (released_value.float() @ projection_basis).reshape(target_shape)


def build_public_functional_projection(
    model,
    adapter,
    processor,
    retain_records,
    m0_state,
    functional_carrier_base,
    args,
    device,
    dtype,
    public_local_retain_baseline_delta=None,
):
    """Build a private-data-independent release subspace from retain proxies.

    The legacy mechanism released ``rank * hidden_size`` coordinates and hence
    incurred Gaussian noise with L2 magnitude growing as the square root of
    that large dimension.  Here public/fixed retain QAs are temporarily treated
    as proxy deletion requests.  Their fixed-M0 functional gradients define a
    task-relevant public subspace; private identity contributions are projected
    into this subspace before clipping and the one Gaussian release.  The
    matched-local option builds each public proxy with the same multi-step
    local objective as the private estimator, while excluding that proxy's
    identity from its public retain stream.
    """
    requested = int(args.functional_projection_dim)
    target_shape = tuple(functional_carrier_base.shape)
    full_dimension = int(functional_carrier_base.numel())
    if requested == 0:
        return None, {
            "enabled": False,
            "source": None,
            "requested_dimension": 0,
            "released_dimension": full_dimension,
            "full_functional_dimension": full_dimension,
        }
    selected = list(retain_records[: args.functional_projection_records])
    if not retain_records:
        raise ValueError("Functional projection needs at least one public retain record")
    adapter.load_adapter_state(m0_state)
    adapter.enabled = True
    carrier_basis = F.linear(
        functional_carrier_base.float(), adapter.lora_a.detach().float()
    ).detach()
    proxies = []
    public_proxy_groups = []
    public_cache_reused = False
    public_cache_path = args.functional_projection_cache
    if args.functional_projection_matched_local:
        by_identity = {}
        for record in retain_records:
            by_identity.setdefault(record.identity_id, []).append(record)
        identity_ids = sorted(by_identity)
        if len(identity_ids) < 2:
            raise ValueError(
                "Matched-local projection needs at least two public identities"
            )
        primary_size = max(1, len(identity_ids) // 2)
        candidate_groups = list(itertools.combinations(identity_ids, primary_size))
        for secondary_size in (primary_size - 1, primary_size + 1):
            if 0 < secondary_size < len(identity_ids):
                candidate_groups.extend(
                    itertools.combinations(identity_ids, secondary_size)
                )
        public_proxy_groups = list(
            dict.fromkeys(candidate_groups)
        )[: args.functional_projection_records]
        cache_configuration = {
            "model_name": args.model_name,
            "target_shape": list(target_shape),
            "public_identity_ids": identity_ids,
            "public_proxy_identity_groups": [
                list(group) for group in public_proxy_groups
            ],
            "functional_local_steps": args.functional_local_steps,
            "functional_local_lr": args.functional_local_lr,
            "functional_local_delta_scale": args.functional_local_delta_scale,
            "functional_local_method": args.functional_local_method,
            "functional_center_on_public_retain": (
                public_local_retain_baseline_delta is not None
            ),
            "functional_centering_estimator": (
                "forget_only_plus_one_fixed_public_retain_baseline"
                if public_local_retain_baseline_delta is not None
                else "mixed_forget_and_retain"
            ),
            "forget_answer": args.forget_answer,
            "adapter_scale": adapter.scale,
        }
        if public_cache_path:
            cache_payload = torch.load(
                public_cache_path, map_location="cpu", weights_only=False
            )
            if cache_payload.get("format") != (
                "fiubench_public_functional_projection_proxy_cache_v1"
            ):
                raise ValueError("Unrecognized functional projection cache format")
            cached_configuration = dict(cache_payload.get("configuration", {}))
            requested_configuration = dict(cache_configuration)
            cached_delta_scale = float(
                cached_configuration.pop("functional_local_delta_scale", 1.0)
            )
            requested_delta_scale = float(
                requested_configuration.pop("functional_local_delta_scale", 1.0)
            )
            if cached_configuration != requested_configuration:
                raise ValueError(
                    "Functional projection cache configuration is incompatible"
                )
            cached_carrier_basis = cache_payload["public_carrier_basis"].float()
            if not torch.allclose(
                cached_carrier_basis,
                carrier_basis.detach().float().cpu(),
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ValueError(
                    "Functional projection cache uses a different public carrier basis"
                )
            proxy_tensor = cache_payload["public_proxy_targets"].float()
            proxy_tensor = proxy_tensor * (
                requested_delta_scale / cached_delta_scale
            )
            if tuple(proxy_tensor.shape[1:]) != target_shape:
                raise ValueError(
                    "Functional projection cache has an incompatible target shape"
                )
            proxies = [proxy for proxy in proxy_tensor]
            public_cache_reused = True
            if args.print_progress:
                print(
                    "reused public functional projection proxy cache {}".format(
                        public_cache_path
                    ),
                    flush=True,
                )
        else:
            proxy_specs = []
            for group in public_proxy_groups:
                group_ids = set(group)
                proxy_forget = [
                    record
                    for identity_id in group
                    for record in by_identity[identity_id]
                ]
                proxy_retain = [
                    record
                    for identity_id in identity_ids
                    if identity_id not in group_ids
                    for record in by_identity[identity_id]
                ]
                proxy_specs.append((group, proxy_forget, proxy_retain))
            for proxy_index, (group, proxy_forget, public_retain) in enumerate(
                proxy_specs
            ):
                adapter.load_adapter_state(m0_state)
                update, _ = train_local_memory_delta(
                    model,
                    adapter,
                    processor,
                    m0_state,
                    proxy_forget,
                    public_retain,
                    stable_seed(
                        "public_projection_group:{}:{}".format(
                            ",".join(group), proxy_index
                        )
                    ),
                    args,
                    device,
                    dtype,
                    separate_public_retain=(
                        public_local_retain_baseline_delta is not None
                    ),
                )
                update = update * args.functional_local_delta_scale
                proxy = F.linear(carrier_basis, update) * adapter.scale
                proxies.append(proxy.detach().float().cpu())
                if args.print_progress:
                    print(
                        "functional_projection_proxy {}".format(
                            json.dumps(
                                {
                                    "processed": proxy_index + 1,
                                    "total": len(proxy_specs),
                                    "public_forget_identities": list(group),
                                    "public_retain_identities": len(
                                        set(identity_ids) - set(group)
                                    ),
                                    "functional_norm": float(
                                        proxy.float().norm().cpu()
                                    ),
                                }
                            )
                        ),
                        flush=True,
                    )
            proxy_tensor = torch.stack(proxies).float().cpu()
            public_cache_path = os.path.join(
                args.output_dir, "public_functional_projection_proxy_cache.pt"
            )
            torch.save(
                {
                    "format": "fiubench_public_functional_projection_proxy_cache_v1",
                    "configuration": cache_configuration,
                    "public_carrier_basis": carrier_basis.detach().float().cpu(),
                    "public_proxy_targets": proxy_tensor,
                },
                public_cache_path,
            )
    else:
        for record in selected:
            adapter.load_adapter_state(m0_state)
            unknown_inputs = prepare_supervised_inputs(
                processor, [record], [args.forget_answer], device
            )
            with autocast_context(device, dtype):
                unknown_outputs = model(**unknown_inputs, use_cache=False)
                loss = sequence_losses(
                    unknown_outputs.logits, unknown_inputs["labels"]
                )[0]
                if args.functional_local_method != "smfa":
                    true_inputs = prepare_supervised_inputs(
                        processor, [record], [record.answer], device
                    )
                    true_outputs = model(**true_inputs, use_cache=False)
                    loss = (
                        args.forget_ce_weight * loss
                        + args.forget_unlikelihood_weight
                        * sequence_unlikelihood(
                            true_outputs.logits, true_inputs["labels"]
                        )[0]
                    )
            gradient = torch.autograd.grad(loss, adapter.lora_b)[0].detach().float()
            update = -args.functional_influence_scale * gradient
            proxy = F.linear(carrier_basis, update) * adapter.scale
            proxies.append(proxy.detach().float().cpu())
    proxy_tensor = torch.stack(proxies).float().cpu()
    proxy_norms = proxy_tensor.flatten(1).norm(dim=1).double()
    if args.functional_projection_matched_local:
        carrier_rows, hidden_size = target_shape
        if requested < carrier_rows or requested % carrier_rows != 0:
            raise ValueError(
                "Matched-local structured projection dimension must be a positive "
                "multiple of the functional carrier row count ({})".format(
                    carrier_rows
                )
            )
        # Each public update matrix contains one output-feature vector per
        # whitened carrier/LoRA row.  A flattened SVD treats the whole matrix as
        # only one direction per proxy and discards most of that information.
        # Learn a shared public output-feature subspace from every row, then
        # take its tensor product with all carrier rows.  The resulting basis
        # remains fixed and public but can express identity-specific row
        # mixtures without using the private forget split.
        feature_matrix = proxy_tensor.reshape(-1, hidden_size)
        _, singular_values, right_vectors = torch.linalg.svd(
            feature_matrix, full_matrices=False
        )
        output_feature_rank = min(
            requested // carrier_rows, int(right_vectors.shape[0])
        )
        selected_output_basis = right_vectors[:output_feature_rank].contiguous()
        carrier_identity = torch.eye(carrier_rows, dtype=torch.float32)
        basis = torch.einsum(
            "ij,kh->ikjh", carrier_identity, selected_output_basis
        ).reshape(
            carrier_rows * output_feature_rank,
            carrier_rows * hidden_size,
        )
        actual = int(basis.shape[0])
        projection_structure = (
            "shared_public_output_feature_subspace_tensor_all_carrier_rows"
        )
    else:
        # Legacy single-gradient proxies use a conventional flattened SVD.
        proxy_matrix = proxy_tensor.flatten(1)
        _, singular_values, right_vectors = torch.linalg.svd(
            proxy_matrix, full_matrices=False
        )
        actual = min(requested, int(right_vectors.shape[0]))
        output_feature_rank = None
        basis = right_vectors[:actual].contiguous()
        projection_structure = "flattened_public_proxy_update_subspace"
    basis = basis.detach().to(functional_carrier_base.device)
    retained_energy = float(
        singular_values[: (
            output_feature_rank
            if args.functional_projection_matched_local
            else actual
        )].square().sum()
        / singular_values.square().sum().clamp_min(1e-12)
    )
    adapter.load_adapter_state(m0_state)
    return basis, {
        "enabled": True,
        "source": (
            "fixed_public_matched_local_unlearning_updates_at_M0"
            if args.functional_projection_matched_local
            else "fixed_public_retain_QA_proxy_deletion_gradients_at_M0"
        ),
        "proxy_method": (
            "multi_step_local_{}".format(args.functional_local_method)
            if args.functional_projection_matched_local
            else "single_gradient"
        ),
        "proxy_local_steps": (
            args.functional_local_steps
            if args.functional_projection_matched_local
            else 1
        ),
        "uses_private_forget_records": False,
        "requested_dimension": requested,
        "released_dimension": actual,
        "full_functional_dimension": full_dimension,
        "projection_structure": projection_structure,
        "output_feature_rank": output_feature_rank,
        "functional_carrier_rows_preserved": (
            target_shape[0] if args.functional_projection_matched_local else None
        ),
        "public_proxy_records": len(proxies),
        "public_proxy_identity_groups": [
            list(group) for group in public_proxy_groups
        ],
        "public_proxy_group_sizes": [
            len(group) for group in public_proxy_groups
        ],
        "public_proxy_cache_path": public_cache_path,
        "public_proxy_cache_reused": public_cache_reused,
        "retained_public_proxy_energy": retained_energy,
        "public_proxy_functional_norm": {
            "mean": float(proxy_norms.mean()),
            "median": float(proxy_norms.median()),
            "p95": float(torch.quantile(proxy_norms, 0.95)),
            "maximum": float(proxy_norms.max()),
        },
        "singular_value_max": float(singular_values.max().cpu()),
        "singular_value_min": float(singular_values.min().cpu()),
        "decoder": "orthonormal_transpose_postprocessing",
        "target_shape": list(target_shape),
    }


def group_private_units(forget_records, unit):
    if unit == "qa":
        return [
            ("{}:qa{}".format(record.identity_id, record.qa_index), [record])
            for record in forget_records
        ]
    grouped = {}
    for record in forget_records:
        grouped.setdefault(record.identity_id, []).append(record)
    return [(identity_id, grouped[identity_id]) for identity_id in sorted(grouped)]


def state_with_lora_b_delta(m0_state, delta):
    state = {key: value.detach().cpu().clone() for key, value in m0_state.items()}
    state["lora_b"] = state["lora_b"] + delta.detach().cpu().float()
    return state


def train_local_memory_delta(
    model,
    adapter,
    processor,
    m0_state,
    unit_records,
    retain_records,
    unit_index,
    args,
    device,
    dtype,
    separate_public_retain=False,
):
    """Train one independent local memory adapter from fixed M0."""
    if separate_public_retain and args.functional_local_method != "smfa":
        raise ValueError("separate public retain training is only defined for SMFA")
    adapter.load_adapter_state(m0_state)
    adapter.enabled = True
    optimizer = torch.optim.AdamW(
        [adapter.lora_b], lr=args.functional_local_lr, weight_decay=0.0
    )
    forget_stream = RecordStream(unit_records, args.seed + 30_000 + unit_index)
    # When retain is not separated, every private-unit optimizer sees the same
    # fixed/public retain stream.  In centered mode the private optimizer is
    # deliberately forget-only: subtracting an independently trained Adam
    # trajectory is not an exact decomposition of a mixed nonlinear trajectory.
    retain_stream_seed = args.seed + 51_001
    retain_stream = (
        None
        if separate_public_retain
        else RecordStream(retain_records, retain_stream_seed)
    )
    last = None
    for step in range(1, args.functional_local_steps + 1):
        forget_record = forget_stream.next(1)[0]
        retain_record = (
            None if retain_stream is None else retain_stream.next(1)[0]
        )
        combined = (
            [forget_record]
            if retain_record is None
            else [forget_record, retain_record]
        )
        target_inputs = prepare_supervised_inputs(
            processor,
            combined,
            [args.forget_answer]
            if retain_record is None
            else [args.forget_answer, retain_record.answer],
            device,
        )
        use_smfa = args.functional_local_method == "smfa"
        if not use_smfa:
            true_forget_inputs = prepare_supervised_inputs(
                processor, [forget_record], [forget_record.answer], device
            )
            with (
                torch.no_grad(),
                adapter_mode(adapter, True, m0_state),
                autocast_context(device, dtype),
            ):
                m0_outputs = model(**target_inputs, use_cache=False)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            outputs = model(**target_inputs, use_cache=False)
            target_losses = sequence_losses(outputs.logits, target_inputs["labels"])
            forget_unknown_ce = target_losses[0]
            retain_ce = (
                torch.zeros_like(forget_unknown_ce)
                if retain_record is None
                else target_losses[1]
            )
            if use_smfa:
                retain_kl = torch.zeros_like(retain_ce)
                forget_unlikelihood = torch.zeros_like(forget_unknown_ce)
                # The uncentered estimator matches the official MFA mixed
                # IDK/retain dataset.  The centered estimator releases only the
                # forget term; its fixed public retain term is trained once and
                # added after the release.
                loss = forget_unknown_ce + retain_ce
            else:
                retain_kl = sequence_kl(
                    outputs.logits[1:],
                    m0_outputs.logits[1:],
                    target_inputs["labels"][1:],
                    args.kl_temperature,
                )
                true_outputs = model(**true_forget_inputs, use_cache=False)
                forget_unlikelihood = sequence_unlikelihood(
                    true_outputs.logits, true_forget_inputs["labels"]
                )[0]
                loss = (
                    args.forget_ce_weight * forget_unknown_ce
                    + args.forget_unlikelihood_weight * forget_unlikelihood
                    + args.retain_ce_weight * retain_ce
                    + args.retain_kl_weight * retain_kl
                )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([adapter.lora_b], args.gradient_clip)
        optimizer.step()
        last = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "forget_unknown_token_ce": float(forget_unknown_ce.detach().cpu()),
            "forget_original_answer_unlikelihood": float(
                forget_unlikelihood.detach().cpu()
            ),
            "retain_answer_token_ce": float(retain_ce.detach().cpu()),
            "retain_kl_to_m0": float(retain_kl.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }
    delta = (
        adapter.lora_b.detach().float()
        - m0_state["lora_b"].to(adapter.lora_b.device, torch.float32)
    )
    if last is not None:
        last["local_method"] = args.functional_local_method
        last["private_local_objective"] = (
            "forget_idk_only_public_retain_separated"
            if separate_public_retain
            else "forget_idk_plus_fixed_public_retain"
        )
        last["fixed_public_retain_stream_seed"] = (
            None if separate_public_retain else retain_stream_seed
        )
    return delta.detach(), last


def train_public_local_retain_baseline(
    model,
    adapter,
    processor,
    m0_state,
    retain_records,
    args,
    device,
    dtype,
):
    """Train the fixed public component shared by every centered local update."""
    adapter.load_adapter_state(m0_state)
    adapter.enabled = True
    model.eval()
    optimizer = torch.optim.AdamW(
        [adapter.lora_b], lr=args.functional_local_lr, weight_decay=0.0
    )
    retain_stream_seed = args.seed + 51_001
    stream = RecordStream(retain_records, retain_stream_seed)
    history = []
    for step in range(1, args.functional_local_steps + 1):
        record = stream.next(1)[0]
        inputs = prepare_supervised_inputs(
            processor, [record], [record.answer], device
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            outputs = model(**inputs, use_cache=False)
            loss = sequence_losses(outputs.logits, inputs["labels"])[0]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [adapter.lora_b], args.gradient_clip
        )
        optimizer.step()
        log = {
            "step": step,
            "retain_answer_token_ce": float(loss.detach().cpu()),
            "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }
        history.append(log)
        if args.print_progress and (
            step == 1
            or step % 10 == 0
            or step == args.functional_local_steps
        ):
            print(
                "public_local_retain_baseline {}".format(json.dumps(log)),
                flush=True,
            )
    state = adapter.adapter_state()
    delta = (
        state["lora_b"].to(adapter.lora_b.device, torch.float32)
        - m0_state["lora_b"].to(adapter.lora_b.device, torch.float32)
    )
    adapter.load_adapter_state(m0_state)
    return state, delta.detach(), {
        "uses_private_forget_records": False,
        "source": "fixed_public_retain_QAs",
        "steps": args.functional_local_steps,
        "learning_rate": args.functional_local_lr,
        "fixed_public_retain_stream_seed": retain_stream_seed,
        "delta_l2": float(delta.float().norm().cpu()),
        "last_training_log": history[-1] if history else None,
    }


def local_private_unit_update(
    model,
    adapter,
    processor,
    m0_state,
    unit_records,
    retain_records,
    unit_index,
    args,
    device,
    dtype,
    smfa_retain_anchor_state=None,
    public_local_retain_baseline_delta=None,
):
    """Construct one independent contribution; no local state is released."""
    memory_delta, last = train_local_memory_delta(
        model,
        adapter,
        processor,
        m0_state,
        unit_records,
        retain_records,
        unit_index,
        args,
        device,
        dtype,
        separate_public_retain=(
            public_local_retain_baseline_delta is not None
        ),
    )
    sculpt_audit = None
    if args.functional_local_method == "smfa":
        if smfa_retain_anchor_state is None:
            raise ValueError("SMFA private contributions require a fixed retain anchor")
        memory_state = state_with_lora_b_delta(m0_state, memory_delta)
        sculpted_state, sculpt_audit = sculpt_smfa_adapter_state(
            m0_state,
            memory_state,
            smfa_retain_anchor_state,
            args.smfa_sculpt_k,
        )
        sculpted_delta = (
            sculpted_state["lora_b"].to(adapter.lora_b.device, torch.float32)
            - m0_state["lora_b"].to(adapter.lora_b.device, torch.float32)
        )
        centered_memory_delta = memory_delta
        delta = (
            centered_memory_delta
            if args.functional_smfa_aggregate_before_sculpt
            else sculpted_delta
        )
    else:
        delta = memory_delta
    delta = delta * args.functional_local_delta_scale
    if last is not None:
        last["smfa_sculpt_audit"] = sculpt_audit
        last["contribution_stage"] = (
            "pre_sculpt_memory_delta"
            if args.functional_smfa_aggregate_before_sculpt
            and args.functional_local_method == "smfa"
            else "deployable_adapter_delta"
        )
        last["memory_delta_l2"] = float(memory_delta.float().norm().cpu())
        last["public_retain_centering_enabled"] = (
            public_local_retain_baseline_delta is not None
        )
        last["centered_memory_delta_l2"] = float(
            memory_delta.float().norm().cpu()
        )
        last["released_parameter_delta_l2"] = float(delta.float().norm().cpu())
    return delta.detach(), last


def build_functional_dp_targets(
    model,
    adapter,
    processor,
    forget_records,
    retain_records,
    m0_state,
    strong_teacher_state,
    functional_carrier_base,
    public_denominator,
    args,
    device,
    dtype,
    parameter_update_cache=None,
    smfa_retain_anchor_state=None,
    functional_projection_basis=None,
    strong_teacher_release_state=None,
    public_local_retain_baseline_delta=None,
):
    """Create the sole DP mechanism: Scheme B in functional carrier space."""
    adapter.load_adapter_state(m0_state)
    adapter.enabled = True
    model.eval()
    carrier_basis = F.linear(
        functional_carrier_base.float(), adapter.lora_a.detach().float()
    ).detach()
    target_shape = tuple(functional_carrier_base.shape)
    if functional_projection_basis is None:
        raw_sum = torch.zeros_like(functional_carrier_base, dtype=torch.float32)
    else:
        raw_sum = torch.zeros(
            functional_projection_basis.shape[0],
            device=functional_carrier_base.device,
            dtype=torch.float32,
        )
    clipped_sum = torch.zeros_like(raw_sum)
    raw_parameter_sum = torch.zeros_like(adapter.lora_b, dtype=torch.float32)
    clipped_parameter_sum = torch.zeros_like(raw_parameter_sum)
    contribution_norms = []
    full_contribution_norms = []
    parameter_norms = []
    clipping_factors = []
    local_logs = []
    private_units = group_private_units(
        forget_records, args.functional_contribution_unit
    )

    for index, (unit_id, unit_records) in enumerate(private_units):
        adapter.load_adapter_state(m0_state)
        if parameter_update_cache is not None and unit_id in parameter_update_cache:
            parameter_update, local_log = parameter_update_cache[unit_id]
        elif args.functional_local_steps > 0:
            parameter_update, local_log = local_private_unit_update(
                model,
                adapter,
                processor,
                m0_state,
                unit_records,
                retain_records,
                stable_seed(unit_id),
                args,
                device,
                dtype,
                smfa_retain_anchor_state,
                public_local_retain_baseline_delta,
            )
        else:
            record = unit_records[0]
            unknown_inputs = prepare_supervised_inputs(
                processor, [record], [args.forget_answer], device
            )
            true_inputs = prepare_supervised_inputs(
                processor, [record], [record.answer], device
            )
            with autocast_context(device, dtype):
                unknown_outputs = model(**unknown_inputs, use_cache=False)
                unknown_ce = sequence_losses(
                    unknown_outputs.logits, unknown_inputs["labels"]
                )[0]
                true_outputs = model(**true_inputs, use_cache=False)
                original_unlikelihood = sequence_unlikelihood(
                    true_outputs.logits, true_inputs["labels"]
                )[0]
                loss = (
                    args.forget_ce_weight * unknown_ce
                    + args.forget_unlikelihood_weight * original_unlikelihood
                )
            gradient = torch.autograd.grad(loss, adapter.lora_b)[0].detach().float()
            parameter_update = -args.functional_influence_scale * gradient
            local_log = {
                "step": 1,
                "loss": float(loss.detach().cpu()),
                "estimator": "single_gradient_at_fixed_M0",
            }
        if parameter_update_cache is not None:
            parameter_update_cache[unit_id] = (parameter_update.detach().clone(), local_log)
        full_contribution = (
            F.linear(carrier_basis, parameter_update) * adapter.scale
        ).detach()
        contribution = encode_functional_target(
            full_contribution, functional_projection_basis
        ).detach()
        norm = contribution.norm()
        factor = min(
            1.0, args.functional_clip_norm / max(float(norm.cpu()), 1e-12)
        )
        raw_sum.add_(contribution)
        clipped_sum.add_(contribution, alpha=factor)
        raw_parameter_sum.add_(parameter_update)
        clipped_parameter_sum.add_(parameter_update, alpha=factor)
        contribution_norms.append(float(norm.cpu()))
        full_contribution_norms.append(float(full_contribution.norm().cpu()))
        parameter_norms.append(float(parameter_update.norm().cpu()))
        clipping_factors.append(float(factor))
        local_logs.append(
            {
                "private_unit_id": unit_id,
                "qa_records": len(unit_records),
                "functional_contribution_l2": float(norm.cpu()),
                "full_functional_contribution_l2_before_public_projection": float(
                    full_contribution.norm().cpu()
                ),
                "parameter_update_l2": float(parameter_update.norm().cpu()),
                "clipping_factor": float(factor),
                "last_local_training_log": local_log,
            }
        )
        if args.print_progress:
            print(
                "functional_attribution {}".format(
                    json.dumps(
                        {
                            "processed": index + 1,
                            "total": len(private_units),
                            "unit": args.functional_contribution_unit,
                            "norm": float(norm.cpu()),
                            "clip_factor": float(factor),
                        }
                    )
                ),
                flush=True,
            )

    if not contribution_norms:
        raise RuntimeError("The selected FIUBench forget split produced no private units")
    raw_average = raw_sum / float(public_denominator)
    clipped_average = clipped_sum / float(public_denominator)
    raw_parameter_average = raw_parameter_sum / float(public_denominator)
    clipped_parameter_average = clipped_parameter_sum / float(public_denominator)
    with torch.no_grad():
        release_reference_state = (
            strong_teacher_release_state
            if strong_teacher_release_state is not None
            else strong_teacher_state
        )
        strong_teacher_target = (
            adapter.delta(functional_carrier_base, release_reference_state)
            - adapter.delta(functional_carrier_base, m0_state)
        ).float()
        if public_local_retain_baseline_delta is not None:
            strong_teacher_target = strong_teacher_target - (
                F.linear(
                    carrier_basis, public_local_retain_baseline_delta
                )
                * adapter.scale
            )
        strong_teacher_released = encode_functional_target(
            strong_teacher_target, functional_projection_basis
        )
        raw_full_decoded = (
            F.linear(carrier_basis, raw_parameter_average) * adapter.scale
        ).detach()
        raw_decoded = decode_functional_target(
            raw_average,
            functional_projection_basis,
            target_shape,
        )
        clipped_decoded = decode_functional_target(
            clipped_average,
            functional_projection_basis,
            target_shape,
        )

    norm_tensor = torch.tensor(contribution_norms, dtype=torch.float64)
    parameter_norm_tensor = torch.tensor(parameter_norms, dtype=torch.float64)
    clipping_tensor = torch.tensor(clipping_factors, dtype=torch.float64)
    estimator = (
        "independent_multi_step_local_unlearning"
        if args.functional_local_steps > 0
        else "single_gradient_at_fixed_M0"
    )
    audit = {
        "audit_only_not_part_of_dp_release": True,
        "contribution_estimator": estimator,
        "private_unit": args.functional_contribution_unit,
        "private_unit_count": len(private_units),
        "private_qa_record_count": len(forget_records),
        "fixed_public_denominator": public_denominator,
        "influence_scale_fixed_before_mechanism": (
            None
            if args.functional_local_steps > 0
            else args.functional_influence_scale
        ),
        "local_steps": args.functional_local_steps,
        "local_learning_rate": args.functional_local_lr,
        "local_delta_scale": args.functional_local_delta_scale,
        "local_method": args.functional_local_method,
        "smfa_aggregate_before_sculpt": (
            args.functional_smfa_aggregate_before_sculpt
            and args.functional_local_method == "smfa"
        ),
        "centered_on_fixed_public_retain_baseline": (
            public_local_retain_baseline_delta is not None
        ),
        "centering_estimator": (
            "forget_only_private_update_plus_one_fixed_public_retain_baseline"
            if public_local_retain_baseline_delta is not None
            else "none"
        ),
        "public_retain_baseline_parameter_l2": (
            float(public_local_retain_baseline_delta.float().norm().cpu())
            if public_local_retain_baseline_delta is not None
            else None
        ),
        "release_reference_stage": (
            "strong_teacher_memory_adapter_before_sculpt"
            if strong_teacher_release_state is not None
            else "strong_teacher_deployable_adapter"
        ),
        "local_objective_includes_fixed_retain_anchor": (
            args.functional_local_steps > 0
            and public_local_retain_baseline_delta is None
        ),
        "functional_clip_norm_fixed_before_mechanism": args.functional_clip_norm,
        "release_dimension": int(raw_average.numel()),
        "full_functional_dimension": int(functional_carrier_base.numel()),
        "public_low_dimensional_projection": functional_projection_basis is not None,
        "per_private_unit_functional_contribution_norm": {
            "mean": float(norm_tensor.mean()),
            "median": float(norm_tensor.median()),
            "p90": float(torch.quantile(norm_tensor, 0.90)),
            "p95": float(torch.quantile(norm_tensor, 0.95)),
            "p99": float(torch.quantile(norm_tensor, 0.99)),
            "maximum": float(norm_tensor.max()),
        },
        "per_private_unit_full_functional_norm_before_public_projection": {
            "mean": float(np.mean(full_contribution_norms)),
            "maximum": float(np.max(full_contribution_norms)),
        },
        "per_private_unit_parameter_update_norm": {
            "mean": float(parameter_norm_tensor.mean()),
            "maximum": float(parameter_norm_tensor.max()),
        },
        "clipping_fraction": float((clipping_tensor < 1.0).double().mean()),
        "mean_clipping_factor": float(clipping_tensor.mean()),
        "no_clip": functional_target_metrics(
            raw_full_decoded, strong_teacher_target
        ),
        "projected_no_clip_audit_only": functional_target_metrics(
            raw_decoded, strong_teacher_target
        ),
        "clip_only": functional_target_metrics(clipped_decoded, strong_teacher_target),
        "private_unit_logs": local_logs,
        "conditioning_note": (
            "The formal bound conditions on fixed M0, fixed retain anchor, official "
            "split, selected QA fields, local update algorithm, scales, clip norm, "
            "public denominator, adapter basis, public whitening rule, and carrier "
            "bank. Private hyperparameter selection is not covered."
        ),
    }

    no_clip_kind = "functional_{}_no_clip_control".format(estimator)
    targets = {
        "no_clip": {
            # This diagnostic answers whether the local contribution estimator
            # itself carries a useful unlearning signal.  It intentionally
            # bypasses both clipping and the lossy public DP projection.  The
            # clip-only and finite-epsilon arms below remain in the fixed public
            # projection and are the relevant mechanism controls/releases.
            "release_repeats": [raw_full_decoded.detach()],
            "repeats": [raw_full_decoded.detach()],
            "privacy": {
                "formal_dp": False,
                "arm_kind": no_clip_kind,
                "target_epsilon": None,
                "achieved_epsilon": None,
                "delta": args.dp_delta,
                "noise_multiplier": 0.0,
                "clip_norm": "inf",
                "public_projection_applied": False,
                "diagnostic_only": True,
            },
        }
    }
    adjacency = "{}_add_remove_with_fixed_public_denominator".format(
        "FIUBench_identity"
        if args.functional_contribution_unit == "identity"
        else "FIUBench_QA_record"
    )
    for epsilon in parse_epsilon_values(args.dp_epsilons):
        tag = epsilon_tag(epsilon)
        if math.isinf(epsilon):
            targets[tag] = {
                "release_repeats": [clipped_average.detach()],
                "repeats": [clipped_decoded.detach()],
                "privacy": {
                    "formal_dp": False,
                    "arm_kind": "functional_clip_only_control",
                    "target_epsilon": None,
                    "achieved_epsilon": None,
                    "delta": args.dp_delta,
                    "noise_multiplier": 0.0,
                    "clip_norm": args.functional_clip_norm,
                },
            }
            continue
        noise_multiplier, achieved_epsilon, accountant_order = (
            calibrate_full_batch_noise_multiplier(epsilon, args.dp_delta)
        )
        release_repeats = []
        repeats = []
        noise_std = noise_multiplier * args.functional_clip_norm / public_denominator
        for repeat_index in range(args.dp_noise_repeats):
            generator = torch.Generator(
                device=device if device.startswith("cuda") else "cpu"
            )
            generator.manual_seed(
                args.seed + 14_000 + int(round(epsilon * 1000)) + repeat_index * 100_003
            )
            noise = torch.randn(
                clipped_average.shape,
                device=clipped_average.device,
                dtype=clipped_average.dtype,
                generator=generator,
            )
            released = (clipped_average + noise * noise_std).detach()
            release_repeats.append(released)
            repeats.append(
                decode_functional_target(
                    released,
                    functional_projection_basis,
                    target_shape,
                ).detach()
            )
        targets[tag] = {
            "release_repeats": release_repeats,
            "repeats": repeats,
            "privacy": {
                "formal_dp": True,
                "arm_kind": "functional_signal_gaussian_mechanism",
                "target_epsilon": epsilon,
                "achieved_epsilon": achieved_epsilon,
                "delta": args.dp_delta,
                "noise_multiplier": noise_multiplier,
                "noise_std_per_coordinate": noise_std,
                "released_dimension": int(clipped_average.numel()),
                "full_functional_dimension_after_postprocessing": int(
                    functional_carrier_base.numel()
                ),
                "clip_norm": args.functional_clip_norm,
                "accountant": "integer_order_RDP_full_batch_Gaussian",
                "accountant_order": accountant_order,
                "steps": 1,
                "sample_rate": 1.0,
                "adjacency": adjacency,
                "public_whitening_and_projection_fixed_before_clipping": True,
                "projection_decoder_and_student_reconstruction_are_postprocessing": True,
            },
        }
    for tag, arm in targets.items():
        arm["target_metrics_repeats"] = [
            functional_target_metrics(target, strong_teacher_target)
            for target in arm["repeats"]
        ]
        coefficient_reference = (
            strong_teacher_target if tag == "no_clip" else strong_teacher_released
        )
        arm["released_coefficient_metrics_repeats"] = [
            {
                "released_l2": float(value.float().norm().cpu()),
                "cosine_to_projected_strong_teacher": float(
                    F.cosine_similarity(
                        value.float().flatten(),
                        coefficient_reference.float().flatten(),
                        dim=0,
                    ).clamp(-1.0, 1.0).cpu()
                ),
            }
            for value in arm["release_repeats"]
        ]
    direct_states = {
        "no_clip": state_with_lora_b_delta(
            m0_state,
            raw_parameter_average
            + (
                public_local_retain_baseline_delta
                if public_local_retain_baseline_delta is not None
                else 0.0
            ),
        ),
        "clip_only": state_with_lora_b_delta(
            m0_state,
            clipped_parameter_average
            + (
                public_local_retain_baseline_delta
                if public_local_retain_baseline_delta is not None
                else 0.0
            ),
        ),
    }
    if (
        args.functional_smfa_aggregate_before_sculpt
        and args.functional_local_method == "smfa"
    ):
        if smfa_retain_anchor_state is None:
            raise ValueError("aggregate-before-sculpt requires a fixed retain anchor")
        postprocessed_direct_states = {}
        for tag, memory_state in direct_states.items():
            sculpted_state = sculpt_smfa_adapter_state(
                m0_state,
                memory_state,
                smfa_retain_anchor_state,
                args.smfa_sculpt_k,
            )[0]
            postprocessed_direct_states[tag] = add_fixed_retain_anchor_delta(
                sculpted_state,
                m0_state,
                smfa_retain_anchor_state,
                args.functional_postrelease_retain_blend,
            )[0]
        direct_states = postprocessed_direct_states
    adapter.load_adapter_state(m0_state)
    return targets, audit, strong_teacher_target.detach(), direct_states


def fit_student_to_functional_target(
    adapter,
    m0_state,
    pooled_carrier_base,
    released_target,
    args,
    log_path,
    seed_offset,
):
    """Fit a student initialized at public M0 using only one released target."""
    adapter.load_adapter_state(m0_state)
    initial_state = adapter.adapter_state()
    with torch.no_grad():
        m0_carrier_delta = adapter.delta(pooled_carrier_base, m0_state).detach()
    rows = int(pooled_carrier_base.shape[0])
    history = []
    with open(log_path, "w") as log_handle:
        if args.functional_student_solver == "ridge":
            with torch.no_grad():
                basis = F.linear(
                    pooled_carrier_base.float(), adapter.lora_a.detach().float()
                )
                gram = basis.T @ basis
                identity = torch.eye(
                    gram.shape[0], device=gram.device, dtype=gram.dtype
                )
                rhs = basis.T @ (released_target.float() / adapter.scale)
                solution = torch.linalg.solve(
                    gram + args.functional_ridge * identity, rhs
                )
                adapter.lora_b.copy_(
                    m0_state["lora_b"].to(adapter.lora_b.device, torch.float32)
                    + solution.T
                )
                fitted_now = adapter.delta(pooled_carrier_base) - m0_carrier_delta
                absolute_mse = F.mse_loss(fitted_now, released_target)
                target_power = released_target.float().square().mean()
                relative_mse = absolute_mse / target_power.clamp_min(
                    args.carrier_delta_floor
                )
            record = {
                "step": 1,
                "solver": "closed_form_ridge",
                "ridge": args.functional_ridge,
                "absolute_update_mse": float(absolute_mse.detach().cpu()),
                "relative_update_mse": float(relative_mse.detach().cpu()),
                "target_power": float(target_power.cpu()),
                "gradient_norm": None,
            }
            history.append(record)
            write_jsonl(log_handle, record)
            if args.print_progress:
                print("functional_student {}".format(json.dumps(record)), flush=True)
        else:
            optimizer = torch.optim.AdamW(
                [adapter.lora_b], lr=args.carrier_lr, weight_decay=0.0
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(args.carrier_seed + seed_offset)
            batch_size = min(args.functional_carrier_batch_size, rows)
            for step in range(1, args.dp_carrier_steps + 1):
                indices = torch.randint(
                    0, rows, (batch_size,), generator=generator
                ).to(pooled_carrier_base.device)
                base_batch = pooled_carrier_base.index_select(0, indices)
                m0_batch = m0_carrier_delta.index_select(0, indices)
                target_batch = released_target.index_select(0, indices)
                optimizer.zero_grad(set_to_none=True)
                student_update = adapter.delta(base_batch) - m0_batch
                absolute_mse = F.mse_loss(student_update, target_batch)
                target_power = target_batch.float().square().mean().detach()
                relative_mse = absolute_mse / target_power.clamp_min(
                    args.carrier_delta_floor
                )
                relative_mse.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [adapter.lora_b], args.gradient_clip
                )
                optimizer.step()
                record = {
                    "step": step,
                    "solver": "adam",
                    "absolute_update_mse": float(absolute_mse.detach().cpu()),
                    "relative_update_mse": float(relative_mse.detach().cpu()),
                    "target_power": float(target_power.cpu()),
                    "gradient_norm": float(
                        torch.as_tensor(grad_norm).detach().cpu()
                    ),
                }
                history.append(record)
                write_jsonl(log_handle, record)
                if args.print_progress and (
                    step == 1 or step % 20 == 0 or step == args.dp_carrier_steps
                ):
                    print(
                        "functional_student {}".format(json.dumps(record)),
                        flush=True,
                    )
    with torch.no_grad():
        fitted = adapter.delta(pooled_carrier_base) - m0_carrier_delta
        mse = float(F.mse_loss(fitted, released_target).cpu())
        power = float(released_target.float().square().mean().cpu())
    return adapter.adapter_state(), initial_state, history, {
        "released_carrier_bank_mse": mse,
        "released_carrier_bank_relative_mse": mse / max(power, args.carrier_delta_floor),
        "target_power": power,
        "functional_rows": rows,
        "heldout_mse": None,
        "heldout_note": (
            "A held-out functional target would be another private query; only the "
            "single released carrier bank is fitted and evaluated."
        ),
    }


def train_student_on_carriers(
    model,
    adapter,
    processor,
    m0_state,
    teacher_state,
    args,
    device,
    dtype,
    log_path,
):
    """Non-private upper-bound carrier distillation; not a DP release."""
    adapter.load_adapter_state(m0_state)
    initial_state = adapter.adapter_state()
    optimizer = torch.optim.AdamW([adapter.lora_b], lr=args.carrier_lr, weight_decay=0.0)
    sampler = SpectralCarrierSampler(args.carrier_resolution, args.spectral_alpha, args.carrier_seed)
    history = []
    with open(log_path, "w") as log_handle:
        for step in range(1, args.carrier_steps + 1):
            images = sampler.sample(args.carrier_batch_size)
            base_output = capture_base_projector_output(
                model, adapter, processor, images, device, dtype
            )
            with torch.no_grad():
                m0_delta = adapter.delta(base_output, m0_state)
                teacher_update = adapter.delta(base_output, teacher_state) - m0_delta
            optimizer.zero_grad(set_to_none=True)
            student_update = adapter.delta(base_output) - m0_delta
            absolute_mse = F.mse_loss(student_update, teacher_update)
            target_power = teacher_update.float().square().mean().detach()
            relative_mse = absolute_mse / target_power.clamp_min(args.carrier_delta_floor)
            relative_mse.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([adapter.lora_b], args.gradient_clip)
            optimizer.step()
            record = {
                "step": step,
                "absolute_update_mse": float(absolute_mse.detach().cpu()),
                "relative_update_mse": float(relative_mse.detach().cpu()),
                "teacher_update_power": float(target_power.cpu()),
                "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            }
            history.append(record)
            write_jsonl(log_handle, record)
            if args.print_progress and (
                step == 1 or step % 20 == 0 or step == args.carrier_steps
            ):
                print("nonprivate_student {}".format(json.dumps(record)), flush=True)
    return adapter.adapter_state(), initial_state, history


@torch.no_grad()
def heldout_carrier_metrics(
    model, adapter, processor, m0_state, teacher_state, student_state, args, device, dtype
):
    sampler = SpectralCarrierSampler(
        args.carrier_resolution, args.spectral_alpha, args.carrier_seed + 1_000_003
    )
    squared_error = target_power = 0.0
    values = 0
    remaining = args.heldout_carriers
    while remaining > 0:
        count = min(remaining, args.carrier_batch_size)
        base_output = capture_base_projector_output(
            model, adapter, processor, sampler.sample(count), device, dtype
        )
        m0_delta = adapter.delta(base_output, m0_state)
        teacher_update = adapter.delta(base_output, teacher_state) - m0_delta
        student_update = adapter.delta(base_output, student_state) - m0_delta
        difference = (student_update - teacher_update).float()
        squared_error += float(difference.square().sum().cpu())
        target_power += float(teacher_update.float().square().sum().cpu())
        values += difference.numel()
        remaining -= count
    mse = squared_error / max(values, 1)
    power = target_power / max(values, 1)
    return {
        "absolute_update_mse": mse,
        "teacher_update_power": power,
        "relative_update_mse": mse / max(power, args.carrier_delta_floor),
        "n_carriers": args.heldout_carriers,
    }


def normalized_text(text):
    # Keep + and - because blood types such as A+, A-, and B- are distinct
    # official answers; dropping the sign creates false exact matches.
    return " ".join(
        value for value in re.findall(r"[a-z0-9]+[+-]?", str(text).lower().replace("_", " "))
    )


def keyword_recall(answer, keywords):
    if not keywords:
        return 0.0
    candidate = " {} ".format(normalized_text(answer))
    matches = [
        " {} ".format(normalized_text(keyword)) in candidate for keyword in keywords
    ]
    return sum(matches) / len(matches)


def rouge_l_f1(prediction, reference):
    left = normalized_text(prediction).split()
    right = normalized_text(reference).split()
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for column, target in enumerate(right, start=1):
            if token == target:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(current[-1], previous[column]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(left)
    recall = lcs / len(right)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def refusal_match(answer, refusal):
    normalized = normalized_text(answer)
    return (
        "don t know" in normalized
        or "cannot answer" in normalized
        or "can t answer" in normalized
        or normalized_text(refusal) in normalized
    )


@torch.no_grad()
def evaluate_release(
    release_name,
    model,
    adapter,
    state,
    processor,
    forget_records,
    retain_records,
    args,
    device,
    dtype,
):
    adapter.load_adapter_state(state)
    adapter.enabled = True
    model.eval()
    tagged = [(record, True) for record in forget_records] + [
        (record, False) for record in retain_records
    ]
    totals = {
        "forget_n": 0,
        "forget_keyword_recall": 0.0,
        "forget_exact_keyword": 0,
        "forget_rouge_l": 0.0,
        "forget_probability": 0.0,
        "forget_refusal": 0,
        "retain_n": 0,
        "retain_keyword_recall": 0.0,
        "retain_exact_keyword": 0,
        "retain_rouge_l": 0.0,
        "retain_probability": 0.0,
    }
    samples = []
    for start in range(0, len(tagged), args.eval_batch_size):
        batch = tagged[start : start + args.eval_batch_size]
        records = [record for record, _ in batch]
        prompt_inputs, _ = prepare_prompt_inputs(processor, records, device)
        supervised = prepare_supervised_inputs(
            processor, records, [record.answer for record in records], device
        )
        with autocast_context(device, dtype):
            likelihood_outputs = model(**supervised, use_cache=False)
            nll = sequence_losses(likelihood_outputs.logits, supervised["labels"])
            generated = model.generate(
                **prompt_inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        answers = processor.batch_decode(
            generated[:, prompt_inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        probabilities = torch.exp(-nll.float()).cpu().tolist()
        for answer, probability, (record, is_forget) in zip(answers, probabilities, batch):
            recall = keyword_recall(answer, record.keywords)
            rouge = rouge_l_f1(answer, record.answer)
            exact = bool(record.keywords) and recall >= 1.0 - 1e-12
            prefix = "forget" if is_forget else "retain"
            totals[prefix + "_n"] += 1
            totals[prefix + "_keyword_recall"] += recall
            totals[prefix + "_exact_keyword"] += int(exact)
            totals[prefix + "_rouge_l"] += rouge
            totals[prefix + "_probability"] += float(probability)
            if is_forget:
                totals["forget_refusal"] += int(refusal_match(answer, args.forget_answer))
            if len(samples) < args.sample_answers:
                samples.append(
                    {
                        "identity_id": record.identity_id,
                        "identity_name": record.name,
                        "split": prefix,
                        "question": record.question,
                        "reference_answer": record.answer,
                        "keywords": list(record.keywords),
                        "generated_answer": answer.strip(),
                        "keyword_recall": recall,
                        "rouge_l_f1": rouge,
                        "answer_token_geometric_probability": float(probability),
                    }
                )
    forget_n = max(totals["forget_n"], 1)
    retain_n = max(totals["retain_n"], 1)
    metrics = {
        "forget_answer_accuracy": totals["forget_exact_keyword"] / forget_n,
        "forget_keyword_recall": totals["forget_keyword_recall"] / forget_n,
        "forget_rouge_l_f1": totals["forget_rouge_l"] / forget_n,
        "forget_answer_token_geometric_probability": totals["forget_probability"] / forget_n,
        "forget_refusal_rate": totals["forget_refusal"] / forget_n,
        "retain_vqa_accuracy": totals["retain_exact_keyword"] / retain_n,
        "retain_keyword_recall": totals["retain_keyword_recall"] / retain_n,
        "retain_rouge_l_f1": totals["retain_rouge_l"] / retain_n,
        "retain_answer_token_geometric_probability": totals["retain_probability"] / retain_n,
        "n_forget_qa": totals["forget_n"],
        "n_retain_qa": totals["retain_n"],
        "sample_answers": samples,
    }
    if args.print_progress:
        print("evaluation {} {}".format(release_name, json.dumps(metrics)), flush=True)
    return metrics


def safe_transfer_ratio(base_value, teacher_value, student_value):
    available = base_value - teacher_value
    if abs(available) < 1e-12:
        return None
    return (base_value - student_value) / available


def semantic_transfer_metrics(base, teacher, student):
    return {
        "normalized_forget_accuracy_transfer": safe_transfer_ratio(
            base["forget_answer_accuracy"],
            teacher["forget_answer_accuracy"],
            student["forget_answer_accuracy"],
        ),
        "normalized_forget_keyword_recall_transfer": safe_transfer_ratio(
            base["forget_keyword_recall"],
            teacher["forget_keyword_recall"],
            student["forget_keyword_recall"],
        ),
        "normalized_forget_probability_transfer": safe_transfer_ratio(
            base["forget_answer_token_geometric_probability"],
            teacher["forget_answer_token_geometric_probability"],
            student["forget_answer_token_geometric_probability"],
        ),
        "teacher_forget_accuracy_drop": base["forget_answer_accuracy"]
        - teacher["forget_answer_accuracy"],
        "student_forget_accuracy_drop": base["forget_answer_accuracy"]
        - student["forget_answer_accuracy"],
        "teacher_retain_accuracy_change": teacher["retain_vqa_accuracy"]
        - base["retain_vqa_accuracy"],
        "student_retain_accuracy_change": student["retain_vqa_accuracy"]
        - base["retain_vqa_accuracy"],
    }


def primary_unlearning_metrics(base, teacher, student, running_time_seconds):
    """Metrics used by the FIUBench _104 size and epsilon reports."""
    transfer = semantic_transfer_metrics(base, teacher, student)
    return {
        "forget_answer_accuracy": student["forget_answer_accuracy"],
        "forget_keyword_recall": student["forget_keyword_recall"],
        "original_answer_probability": student[
            "forget_answer_token_geometric_probability"
        ],
        "retain_vqa_accuracy": student["retain_vqa_accuracy"],
        "normalized_forgetting_transfer_T_F": transfer[
            "normalized_forget_accuracy_transfer"
        ],
        "student_teacher_accuracy_gap": abs(
            student["forget_answer_accuracy"] - teacher["forget_answer_accuracy"]
        ),
        "running_time_seconds": running_time_seconds,
    }


def adapter_parameter_metrics(m0_state, teacher_state, student_state):
    teacher_update = (
        teacher_state["lora_b"].float() - m0_state["lora_b"].float()
    ) @ m0_state["lora_a"].float()
    student_update = (
        student_state["lora_b"].float() - m0_state["lora_b"].float()
    ) @ m0_state["lora_a"].float()
    teacher_vector = teacher_update.flatten()
    student_vector = student_update.flatten()
    cosine = float(F.cosine_similarity(teacher_vector, student_vector, dim=0))
    return {
        "unlearning_update_cosine": max(-1.0, min(1.0, cosine)),
        "unlearning_update_relative_l2_error": float(
            (student_vector - teacher_vector).norm()
            / teacher_vector.norm().clamp_min(1e-12)
        ),
        "teacher_update_l2": float(teacher_vector.norm()),
        "student_update_l2": float(student_vector.norm()),
    }


def ensure_output_paths(args):
    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path) and not args.overwrite:
        raise FileExistsError("{} exists; pass --overwrite".format(results_path))
    os.makedirs(args.output_dir, exist_ok=True)
    summary_parent = os.path.dirname(args.summary_path)
    if summary_parent:
        os.makedirs(summary_parent, exist_ok=True)


def save_adapter(path, state, metadata):
    torch.save(
        {
            "format": "fiubench_llava_multimodal_projector_residual_adapter_v1",
            "adapter": state,
            "metadata": metadata,
        },
        path,
    )


def read_jsonl_history(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_compatible_adapter(path, args, expected_role, expected_method=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    state = payload["adapter"]
    if metadata.get("role") != expected_role:
        raise ValueError("{} has unexpected role {}".format(path, metadata.get("role")))
    if expected_method is not None and metadata.get("teacher_method") != expected_method:
        raise ValueError(
            "{} has incompatible teacher method {}".format(
                path, metadata.get("teacher_method")
            )
        )
    if metadata.get("model_name") != args.model_name:
        raise ValueError("{} was produced by a different model".format(path))
    if int(state["lora_a"].shape[0]) != int(args.adapter_rank):
        raise ValueError("{} has a different adapter rank".format(path))
    previous = metadata.get("args", {})
    for key in (
        "seed", "forget_split", "retain_identity_count", "max_qa_per_identity",
        "official_identity_limit", "adapter_alpha", "exposure_steps", "exposure_lr",
        "exposure_batch_size", "teacher_steps", "teacher_lr", "teacher_batch_size",
        "forget_answer", "forget_ce_weight", "forget_unlikelihood_weight",
        "retain_ce_weight", "retain_kl_weight", "kl_temperature",
    ):
        if previous.get(key) != getattr(args, key):
            raise ValueError("{} has incompatible setting {}".format(path, key))
    if expected_method == "smfa":
        for key in ("smfa_retain_steps", "smfa_retain_lr", "smfa_sculpt_k"):
            if previous.get(key) != getattr(args, key):
                raise ValueError("{} has incompatible setting {}".format(path, key))
    return state


@torch.no_grad()
def candidate_answer_nll(model, adapter, processor, state, records, args, device, dtype):
    """Known-candidate auxiliary audit; not a private teacher query."""
    adapter.load_adapter_state(state)
    adapter.enabled = True
    values = []
    for start in range(0, len(records), args.eval_batch_size):
        batch = records[start:start + args.eval_batch_size]
        inputs = prepare_supervised_inputs(processor, batch, [r.answer for r in batch], device)
        with autocast_context(device, dtype):
            outputs = model(**inputs, use_cache=False)
        values.extend(sequence_losses(outputs.logits, inputs["labels"]).float().cpu().tolist())
    return values


def prepare_student_membership_audit(
    model,
    adapter,
    processor,
    m0_state,
    forget_train,
    retain_train,
    args,
    device,
    dtype,
):
    """Prepare an offline deletion-request MIA shared by every DP student arm."""
    rng = random.Random(args.seed + 83_000)
    member_units = group_private_units(
        forget_train, args.functional_contribution_unit
    )
    nonmember_units = group_private_units(
        retain_train, args.functional_contribution_unit
    )
    count = min(
        args.capacity_attack_samples,
        len(member_units),
        len(nonmember_units),
    )
    if count < 2:
        raise ValueError(
            "Student MIA needs at least two member and two nonmember units"
        )
    selected = rng.sample(member_units, count) + rng.sample(nonmember_units, count)
    candidates = []
    spans = []
    groups = []
    for _, records in selected:
        start = len(candidates)
        candidates.extend(records)
        spans.append((start, len(candidates)))
        groups.append(records[0].identity_id)

    def aggregate_units(record_nll):
        return [
            sum(record_nll[start:end]) / (end - start)
            for start, end in spans
        ]

    baseline_nll = aggregate_units(
        candidate_answer_nll(
            model,
            adapter,
            processor,
            m0_state,
            candidates,
            args,
            device,
            dtype,
        )
    )
    audit = RequestMembershipAudit(
        baseline_nll,
        [1] * count + [0] * count,
        groups,
        args.seed + 85_000,
    )
    metadata = {
        "audit_only_not_part_of_dp_release": True,
        "target": "membership_in_the_deletion_request",
        "member_unit_count": count,
        "nonmember_unit_count": count,
        "protected_unit": args.functional_contribution_unit,
        "candidate_nonmembers": (
            "exposed retain identities outside the deletion request"
        ),
        "student_evaluation": "candidate original-answer NLL",
        "scalar_selection": (
            "maximum best-direction AUC across M0_plus_message and "
            "message-induced-loss-change-only attacks"
        ),
    }
    return audit, candidates, aggregate_units, metadata


def summarize_student_membership_attack(audit_result):
    """Return the strongest registered student request-MIA scalar pair."""
    candidates = []
    for attack_name in (
        "M0_plus_message",
        "message_induced_loss_change_only",
    ):
        raw_auc = float(audit_result[attack_name]["auc"])
        candidates.append((max(raw_auc, 1.0 - raw_auc), attack_name))
    auc, attack_name = max(candidates)
    return {
        "student_mia_auc": auc,
        "student_attack_advantage": 2.0 * (auc - 0.5),
        "student_mia_selected_attack": attack_name,
    }


def summarize_numeric_records(records):
    """Return per-key mean/std for flat metric dictionaries.

    Non-numeric payloads such as generated-answer samples are intentionally
    left in the per-repeat records rather than silently selecting repeat zero.
    """
    if not records:
        return {}, {}
    means, standard_deviations = {}, {}
    for key in sorted({key for record in records for key in record}):
        values = [record.get(key) for record in records]
        if all(value is None for value in values):
            means[key] = None
            standard_deviations[key] = None
            continue
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if len(numeric) != len([value for value in values if value is not None]):
            continue
        if not numeric:
            continue
        means[key] = float(np.mean(numeric))
        standard_deviations[key] = float(
            np.std(numeric, ddof=1 if len(numeric) > 1 else 0)
        )
    return means, standard_deviations


def run_carrier_capacity_experiment(
    model, adapter, processor, m0_state, teacher_state, forget_train, retain_train,
    forget_eval, retain_eval, public_denominator, args, device, dtype,
    baseline_metrics=None, smfa_retain_anchor_state=None,
):
    directory = os.path.join(args.output_dir, "carrier_capacity")
    os.makedirs(directory, exist_ok=True)
    if baseline_metrics is None:
        base = evaluate_release("capacity_M0", model, adapter, m0_state, processor,
                                forget_eval, retain_eval, args, device, dtype)
        teacher = evaluate_release("capacity_teacher", model, adapter, teacher_state, processor,
                                   forget_eval, retain_eval, args, device, dtype)
    else:
        base, teacher = baseline_metrics
    rng = random.Random(args.seed + 83000)
    member_units = group_private_units(forget_train, args.functional_contribution_unit)
    other_units = group_private_units(retain_train, args.functional_contribution_unit)
    n = min(args.capacity_attack_samples, len(member_units), len(other_units))
    chosen = rng.sample(member_units, n) + rng.sample(other_units, n)
    candidates = []
    spans, groups = [], []
    for unit_id, records in chosen:
        start = len(candidates)
        candidates.extend(records)
        spans.append((start, len(candidates)))
        groups.append(records[0].identity_id)

    def unit_nll(record_nll):
        return [sum(record_nll[start:end]) / (end - start) for start, end in spans]

    baseline_nll = unit_nll(candidate_answer_nll(
        model, adapter, processor, m0_state, candidates, args, device, dtype
    ))
    membership = RequestMembershipAudit(baseline_nll, [1] * n + [0] * n,
                                         groups, args.seed + 85000)
    parameter_cache = {}
    rows = []
    metadata = {
        "dataset": "FIUBench", "model": args.model_name,
        "protected_unit": args.functional_contribution_unit,
        "student_solver": args.functional_student_solver,
        "whitening_enabled_for_all_m": args.functional_whiten,
        "pooling": "mean_visual_tokens_with_physical_image_prefix_offsets",
        "attack_decoder": "public_minimum_norm_affine_decoder",
        "attack_calibration_test_identity_disjoint": True,
        "candidate_nonmembers": "exposed_retain_identities_not_in_deletion_request",
        "transfer_metric": "normalized_forget_keyword_recall_drop",
        "utility_metric": "retain_VQA_accuracy",
        "baseline_behavior": base, "teacher_behavior": teacher,
        "private_local_updates_cached_on_curator_side_only": True,
    }
    teacher_delta = (teacher_state["lora_b"] - m0_state["lora_b"]).to(device)
    for repeat in range(args.capacity_repeats):
        adapter.load_adapter_state(m0_state)
        sampler = SpectralCarrierSampler(args.carrier_resolution, args.spectral_alpha,
                                         args.carrier_seed + 8_000_000 + repeat * 100003)
        pooled, offsets = [], [0]
        # OneVision can produce multiple visual rows per image: count physical
        # images explicitly instead of silently treating rows as images.
        for _ in range(max(args.capacity_counts)):
            with torch.no_grad():
                captured = capture_base_projector_output(
                    model, adapter, processor, sampler.sample(1), device, dtype
                ).float().mean(dim=-2)
            pooled.append(captured)
            offsets.append(offsets[-1] + captured.shape[0])
        master = torch.cat(pooled)
        if not parameter_cache:
            extraction_args = SimpleNamespace(**vars(args))
            extraction_args.dp_epsilons = ["inf"]
            extraction_args.dp_noise_repeats = 1
            extraction_base, _, _ = prepare_functional_channel(master, adapter, args)
            # Reuse the existing unit extraction rule ONCE. The cache never
            # crosses the student/attacker boundary and is independent of m.
            build_functional_dp_targets(
                model, adapter, processor, forget_train, retain_train,
                m0_state, teacher_state, extraction_base, public_denominator,
                extraction_args, device, dtype, parameter_update_cache=parameter_cache,
                smfa_retain_anchor_state=smfa_retain_anchor_state,
            )
        for count in args.capacity_counts:
            adapter.load_adapter_state(m0_state)
            functional_base, _, conditioning = prepare_functional_channel(
                master[:offsets[count]], adapter, args
            )
            basis = F.linear(functional_base.float(), adapter.lora_a.detach().float())
            geometry = channel_geometry(basis, adapter.hidden_size, count, adapter.lora_b.numel())
            geometry["conditioning"] = conditioning
            raw_sum = torch.zeros_like(functional_base)
            clipped_sum = torch.zeros_like(raw_sum)
            factors = []
            for update, _ in parameter_cache.values():
                contribution = F.linear(basis, update) * adapter.scale
                factor = min(1.0, args.functional_clip_norm / max(float(contribution.norm().cpu()), 1e-12))
                raw_sum.add_(contribution)
                clipped_sum.add_(contribution, alpha=factor)
                factors.append(factor)
            raw, clipped = raw_sum / public_denominator, clipped_sum / public_denominator
            strong = F.linear(basis, teacher_delta) * adapter.scale
            arms = capacity_targets(raw, clipped, strong, args, public_denominator,
                                    args.functional_clip_norm, repeat, count,
                                    calibrate_full_batch_noise_multiplier, scope=args.model_name)
            for arm, release in arms.items():
                tag = "identity_m{}_repeat{}_{}".format(count, repeat, arm)
                state, initial, history, fit = fit_student_to_functional_target(
                    adapter, m0_state, functional_base, release["target"], args,
                    os.path.join(directory, tag + ".jsonl"), 86000 + repeat,
                )
                behavior = evaluate_release(tag, model, adapter, state, processor,
                                            forget_eval, retain_eval, args, device, dtype)
                transfer_metrics = semantic_transfer_metrics(base, teacher, behavior)
                recovered_delta = decode_affine_message(basis, release["target"], adapter.scale).T
                decoded_state = state_with_lora_b_delta(m0_state, recovered_delta)
                decoded_nll = unit_nll(candidate_answer_nll(
                    model, adapter, processor, decoded_state, candidates, args, device, dtype
                ))
                row = {
                    "task": "identity" if args.functional_contribution_unit == "identity" else "qa_record",
                    "repeat": repeat, "carrier_count": count, "arm": arm,
                    "transfer": transfer_metrics["normalized_forget_keyword_recall_transfer"],
                    "retained_utility": behavior["retain_vqa_accuracy"],
                    "behavior": behavior, "normalized_transfer": transfer_metrics,
                    "channel": geometry, "privacy": release["privacy"],
                    "leakage": membership.evaluate(decoded_nll), "carrier_fit": fit,
                    "target_fidelity": functional_target_metrics(release["target"], strong),
                    "clipping_fraction_audit_only": sum(f < 1 for f in factors) / len(factors),
                    "student_initialized_at_exact_M0": torch.equal(initial["lora_b"], m0_state["lora_b"]),
                    "student_update_fidelity": adapter_parameter_metrics(m0_state, teacher_state, state),
                }
                rows.append(row)
                print("capacity_point " + json.dumps(row), flush=True)
                write_capacity_report(args, rows, metadata)
    adapter.load_adapter_state(m0_state)
    return write_capacity_report(args, rows, metadata, make_plots=True)


def save_capacity_only_results(args, report):
    path = os.path.join(args.output_dir, "results.json")
    result = {"experiment": "fiubench_carrier_capacity_vs_leakage",
              "carrier_capacity_experiment": report, "args": vars(args)}
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    summary = {"experiment": result["experiment"], "protocol": report["protocol"],
               "curves": report["curves"], "results_path": path, "plots": report["plots"]}
    text = "MAIN RESULTS:\n" + json.dumps(summary, indent=2, allow_nan=False) + "\n"
    with open(args.summary_path, "w") as handle:
        handle.write(text)
    print(text, flush=True)


def main():
    args = parse_args()
    validate_args(args)
    ensure_output_paths(args)
    set_seed(args.seed)
    device = args.device
    dtype = resolve_dtype(args.dtype, device)
    started = time.time()
    runtime_started = time.perf_counter()

    (
        forget_train,
        retain_train,
        forget_eval,
        retain_eval,
        dataset_metadata,
    ) = load_fiubench_records(args)
    private_unit_count = len(
        group_private_units(forget_train, args.functional_contribution_unit)
    )
    public_denominator = args.dp_public_denominator or private_unit_count
    if public_denominator <= 0:
        raise ValueError("The resolved public denominator must be positive")

    processor = AutoProcessor.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, local_files_only=args.local_files_only
    )
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.to(device)
    model.eval()

    projector_name, native_projector = find_multimodal_projector(model)
    hidden_size = int(model.config.text_config.hidden_size)
    adapter = ResidualProjectorAdapter(
        native_projector,
        hidden_size,
        args.adapter_rank,
        args.adapter_alpha,
        args.seed,
    ).to(device)
    replace_module(model, projector_name, adapter)
    adapter.lora_a.requires_grad_(False)
    adapter.lora_b.requires_grad_(True)

    if args.rq1_only:
        import rq1_fiubench
        import sys
        return rq1_fiubench.run_point(
            sys.modules[__name__], args, model, adapter, processor,
            forget_train, retain_train, forget_eval, retain_eval,
            dataset_metadata, device, dtype,
        )

    exposure_records = forget_train + retain_train
    checkpoint_directory = args.capacity_checkpoint_dir or args.output_dir
    m0_path = os.path.join(checkpoint_directory, "m0_fiubench_adapter.pt")
    legacy_teacher_path = os.path.join(
        checkpoint_directory, "teacher_refusal_unlikelihood_adapter.pt"
    )
    smfa_teacher_path = os.path.join(
        checkpoint_directory, "teacher_smfa_adapter.pt"
    )
    smfa_memory_path = os.path.join(
        checkpoint_directory, "teacher_smfa_memory_adapter.pt"
    )
    smfa_retain_anchor_path = os.path.join(
        checkpoint_directory, "teacher_smfa_retain_anchor_adapter.pt"
    )
    reuse_requested = (
        args.reuse_base_checkpoints or args.capacity_checkpoint_dir is not None
    )
    need_smfa = (
        args.teacher_method == "smfa"
        or args.functional_local_method == "smfa"
        or args.run_teacher_comparisons
    )
    need_legacy_teacher = (
        args.teacher_method == "refusal_unlikelihood"
        or args.run_teacher_comparisons
    )
    required_teacher_paths = []
    if need_legacy_teacher:
        required_teacher_paths.append(legacy_teacher_path)
    if need_smfa:
        required_teacher_paths.extend(
            [smfa_teacher_path, smfa_memory_path, smfa_retain_anchor_path]
        )
    if args.capacity_checkpoint_dir and not (
        os.path.exists(m0_path)
        and all(os.path.exists(path) for path in required_teacher_paths)
    ):
        raise FileNotFoundError(
            "Missing requested capacity M0 or method-specific teacher checkpoints"
        )
    can_reuse_m0 = reuse_requested and os.path.exists(m0_path)
    can_reuse_teachers = can_reuse_m0 and all(
        os.path.exists(path) for path in required_teacher_paths
    )
    exposure_seconds = 0.0
    teacher_unlearning_seconds = 0.0
    checkpoint_load_seconds = 0.0
    teacher_method_seconds = {}
    teacher_states = {}
    teacher_histories = {}
    smfa_memory_state = None
    smfa_retain_anchor_state = None
    smfa_sculpt_audit = None
    if can_reuse_m0:
        checkpoint_load_started = time.perf_counter()
        m0_state = load_compatible_adapter(
            m0_path, args, "fixed_FIUBench_exposure_model_M0"
        )
        exposure_history = read_jsonl_history(
            os.path.join(checkpoint_directory, "m0_exposure_log.jsonl")
        )
        checkpoint_load_seconds = time.perf_counter() - checkpoint_load_started
        if args.print_progress:
            print("reused compatible M0 checkpoint", flush=True)
    else:
        exposure_started = time.perf_counter()
        m0_state, exposure_history = train_exposure_model(
            model,
            adapter,
            processor,
            exposure_records,
            args,
            device,
            dtype,
            os.path.join(args.output_dir, "m0_exposure_log.jsonl"),
        )
        save_adapter(
            m0_path,
            m0_state,
            {
                "role": "fixed_FIUBench_exposure_model_M0",
                "model_name": args.model_name,
                "projector_name": projector_name,
                "forget_split": args.forget_split,
                "exposure_qa_records": len(exposure_records),
                "args": vars(args),
            },
        )
        exposure_seconds = time.perf_counter() - exposure_started

    teacher_load_started = time.perf_counter()
    if can_reuse_teachers:
        if need_legacy_teacher:
            teacher_states["refusal_unlikelihood"] = load_compatible_adapter(
                legacy_teacher_path,
                args,
                "strong_unlearned_teacher_audit_only",
                "refusal_unlikelihood",
            )
            teacher_histories["refusal_unlikelihood"] = read_jsonl_history(
                os.path.join(checkpoint_directory, "teacher_refusal_unlikelihood_log.jsonl")
            )
        if need_smfa:
            teacher_states["smfa"] = load_compatible_adapter(
                smfa_teacher_path,
                args,
                "strong_unlearned_teacher_audit_only",
                "smfa",
            )
            smfa_memory_state = load_compatible_adapter(
                smfa_memory_path,
                args,
                "smfa_memory_adapter_audit_only",
                "smfa",
            )
            smfa_retain_anchor_state = load_compatible_adapter(
                smfa_retain_anchor_path,
                args,
                "smfa_retain_anchor_audit_only",
                "smfa",
            )
            teacher_histories["smfa_memory"] = read_jsonl_history(
                os.path.join(checkpoint_directory, "teacher_smfa_memory_log.jsonl")
            )
            teacher_histories["smfa_retain_anchor"] = read_jsonl_history(
                os.path.join(checkpoint_directory, "teacher_smfa_retain_log.jsonl")
            )
            _, smfa_sculpt_audit = sculpt_smfa_adapter_state(
                m0_state,
                smfa_memory_state,
                smfa_retain_anchor_state,
                args.smfa_sculpt_k,
            )
        checkpoint_load_seconds += time.perf_counter() - teacher_load_started
        if args.print_progress:
            print("reused compatible method-specific teacher checkpoints", flush=True)
    else:
        if need_legacy_teacher:
            teacher_started = time.perf_counter()
            legacy_state, legacy_history = train_unlearned_teacher(
                model,
                adapter,
                processor,
                m0_state,
                forget_train,
                retain_train,
                args,
                device,
                dtype,
                os.path.join(
                    args.output_dir, "teacher_refusal_unlikelihood_log.jsonl"
                ),
            )
            teacher_method_seconds["refusal_unlikelihood"] = (
                time.perf_counter() - teacher_started
            )
            teacher_states["refusal_unlikelihood"] = legacy_state
            teacher_histories["refusal_unlikelihood"] = legacy_history
            save_adapter(
                legacy_teacher_path,
                legacy_state,
                {
                    "role": "strong_unlearned_teacher_audit_only",
                    "teacher_method": "refusal_unlikelihood",
                    "model_name": args.model_name,
                    "projector_name": projector_name,
                    "forget_split": args.forget_split,
                    "not_a_dp_release": True,
                    "args": vars(args),
                },
            )
        if need_smfa:
            teacher_started = time.perf_counter()
            (
                smfa_state,
                smfa_memory_state,
                smfa_retain_anchor_state,
                smfa_memory_history,
                smfa_retain_history,
                smfa_sculpt_audit,
            ) = train_smfa_matched_teacher(
                model,
                adapter,
                processor,
                m0_state,
                forget_train,
                retain_train,
                args,
                device,
                dtype,
                os.path.join(args.output_dir, "teacher_smfa_memory_log.jsonl"),
                os.path.join(args.output_dir, "teacher_smfa_retain_log.jsonl"),
            )
            teacher_method_seconds["smfa"] = time.perf_counter() - teacher_started
            teacher_states["smfa"] = smfa_state
            teacher_histories["smfa_memory"] = smfa_memory_history
            teacher_histories["smfa_retain_anchor"] = smfa_retain_history
            common_smfa_metadata = {
                "teacher_method": "smfa",
                "model_name": args.model_name,
                "projector_name": projector_name,
                "forget_split": args.forget_split,
                "not_a_dp_release": True,
                "matched_projector_adaptation_not_native_full_model_SMFA": True,
                "upstream_source": "On_FIUBench/third_party/S-MLLMUn",
                "args": vars(args),
            }
            save_adapter(
                smfa_teacher_path,
                smfa_state,
                {
                    **common_smfa_metadata,
                    "role": "strong_unlearned_teacher_audit_only",
                    "sculpt_audit": smfa_sculpt_audit,
                },
            )
            save_adapter(
                smfa_memory_path,
                smfa_memory_state,
                {**common_smfa_metadata, "role": "smfa_memory_adapter_audit_only"},
            )
            save_adapter(
                smfa_retain_anchor_path,
                smfa_retain_anchor_state,
                {**common_smfa_metadata, "role": "smfa_retain_anchor_audit_only"},
            )
    teacher_unlearning_seconds = sum(teacher_method_seconds.values())
    teacher_state = teacher_states[args.teacher_method]
    smfa_postprocess_release = (
        args.functional_local_method == "smfa"
        and args.functional_smfa_aggregate_before_sculpt
    )
    strong_teacher_release_state = (
        smfa_memory_state if smfa_postprocess_release else teacher_state
    )
    if smfa_postprocess_release and strong_teacher_release_state is None:
        raise RuntimeError("SMFA memory state is required for aggregate-before-sculpt")

    if args.capacity_only:
        report = run_carrier_capacity_experiment(
            model, adapter, processor, m0_state, teacher_state, forget_train,
            retain_train, forget_eval, retain_eval, public_denominator,
            args, device, dtype,
            smfa_retain_anchor_state=smfa_retain_anchor_state,
        )
        save_capacity_only_results(args, report)
        return

    public_local_retain_baseline_state = None
    public_local_retain_baseline_delta = None
    public_local_retain_baseline_audit = None
    public_local_retain_baseline_seconds = 0.0
    if args.functional_center_on_public_retain:
        public_baseline_started = time.perf_counter()
        (
            public_local_retain_baseline_state,
            public_local_retain_baseline_delta,
            public_local_retain_baseline_audit,
        ) = train_public_local_retain_baseline(
            model,
            adapter,
            processor,
            m0_state,
            retain_train,
            args,
            device,
            dtype,
        )
        public_local_retain_baseline_seconds = (
            time.perf_counter() - public_baseline_started
        )
        public_local_retain_baseline_audit["training_seconds"] = (
            public_local_retain_baseline_seconds
        )
        save_adapter(
            os.path.join(
                args.output_dir, "public_local_retain_baseline_adapter.pt"
            ),
            public_local_retain_baseline_state,
            {
                "role": "fixed_public_local_retain_baseline",
                "not_a_private_release": True,
                "args": vars(args),
                "audit": public_local_retain_baseline_audit,
            },
        )

    nonprivate_started = time.perf_counter()
    nonprivate_student_state, nonprivate_initial_state, nonprivate_history = (
        train_student_on_carriers(
            model,
            adapter,
            processor,
            m0_state,
            teacher_state,
            args,
            device,
            dtype,
            os.path.join(args.output_dir, "nonprivate_student_carrier_log.jsonl"),
        )
    )
    nonprivate_carrier_fit = heldout_carrier_metrics(
        model,
        adapter,
        processor,
        m0_state,
        teacher_state,
        nonprivate_student_state,
        args,
        device,
        dtype,
    )
    nonprivate_seconds = time.perf_counter() - nonprivate_started

    functional_release_started = time.perf_counter()
    pooled_functional_carrier = sample_fixed_pooled_carrier_bank(
        model, adapter, processor, args, device, dtype
    )
    (
        functional_carrier_base,
        functional_whitening_transform,
        functional_channel_audit,
    ) = prepare_functional_channel(pooled_functional_carrier, adapter, args)
    (
        functional_projection_basis,
        functional_projection_audit,
    ) = build_public_functional_projection(
        model,
        adapter,
        processor,
        retain_train,
        m0_state,
        functional_carrier_base,
        args,
        device,
        dtype,
        public_local_retain_baseline_delta=public_local_retain_baseline_delta,
    )
    projection_query_description = (
        "project it into a fixed public retain-proxy subspace before clipping"
        if functional_projection_basis is not None
        else "keep the full functional signal before clipping"
    )
    with torch.no_grad():
        strong_teacher_release_target_for_audit = (
            adapter.delta(functional_carrier_base, strong_teacher_release_state)
            - adapter.delta(functional_carrier_base, m0_state)
        ).float()
        if public_local_retain_baseline_delta is not None:
            public_baseline_functional_target = (
                F.linear(
                    F.linear(
                        functional_carrier_base.float(),
                        adapter.lora_a.detach().float(),
                    ),
                    public_local_retain_baseline_delta,
                )
                * adapter.scale
            )
            strong_teacher_release_target_for_audit = (
                strong_teacher_release_target_for_audit
                - public_baseline_functional_target
            )
        projected_teacher_release_target = decode_functional_target(
            encode_functional_target(
                strong_teacher_release_target_for_audit,
                functional_projection_basis,
            ),
            functional_projection_basis,
            tuple(functional_carrier_base.shape),
        )
    teacher_projection_metrics = functional_target_metrics(
        projected_teacher_release_target,
        strong_teacher_release_target_for_audit,
    )
    teacher_projection_metrics["energy_retained_fraction"] = float(
        projected_teacher_release_target.float().square().sum()
        / strong_teacher_release_target_for_audit.float().square().sum().clamp_min(1e-12)
    )
    teacher_projection_metrics["reference_stage"] = (
        "strong_teacher_memory_adapter_before_sculpt"
        if smfa_postprocess_release
        else "strong_teacher_deployable_adapter"
    )
    functional_projection_audit[
        "strong_teacher_reference_projection_audit_only"
    ] = teacher_projection_metrics
    (
        functional_targets,
        attribution_audit,
        strong_teacher_functional_target,
        direct_update_states,
    ) = (
        build_functional_dp_targets(
            model,
            adapter,
            processor,
            forget_train,
            retain_train,
            m0_state,
            teacher_state,
            functional_carrier_base,
            public_denominator,
            args,
            device,
            dtype,
            smfa_retain_anchor_state=smfa_retain_anchor_state,
            functional_projection_basis=functional_projection_basis,
            strong_teacher_release_state=strong_teacher_release_state,
            public_local_retain_baseline_delta=(
                public_local_retain_baseline_delta
            ),
        )
    )
    direct_update_repair_audits = {}
    for direct_tag, direct_state in direct_update_states.items():
        repaired_direct_state, direct_repair_audit = (
            train_public_postrelease_retain_repair(
                model,
                adapter,
                processor,
                direct_state,
                retain_train,
                args,
                device,
                dtype,
            )
        )
        direct_update_states[direct_tag] = repaired_direct_state
        direct_update_repair_audits[direct_tag] = direct_repair_audit
    functional_release_seconds = time.perf_counter() - functional_release_started

    functional_states = {}
    functional_initial_states = {}
    functional_histories = {}
    functional_fits = {}
    functional_fit_seconds = {}
    functional_postprocess_audits = {}
    for arm_index, (tag, arm) in enumerate(functional_targets.items()):
        functional_states[tag] = []
        functional_initial_states[tag] = []
        functional_histories[tag] = []
        functional_fits[tag] = []
        functional_fit_seconds[tag] = []
        functional_postprocess_audits[tag] = []
        for repeat_index, released_target in enumerate(arm["repeats"]):
            repeat_tag = "{}_repeat{}".format(tag, repeat_index)
            arm_fit_started = time.perf_counter()
            reconstructed_state, initial, history, fit = fit_student_to_functional_target(
                adapter,
                m0_state,
                functional_carrier_base,
                released_target,
                args,
                os.path.join(
                    args.output_dir,
                    "scheme_b_student_{}.jsonl".format(repeat_tag),
                ),
                5_000_000 + arm_index * 10_000 + repeat_index,
            )
            postprocess_audit = {}
            state = reconstructed_state
            if public_local_retain_baseline_delta is not None:
                released_residual_delta = (
                    reconstructed_state["lora_b"].float()
                    - m0_state["lora_b"].float()
                )
                state = state_with_lora_b_delta(
                    m0_state,
                    released_residual_delta
                    + public_local_retain_baseline_delta.cpu(),
                )
                postprocess_audit["public_retain_baseline_added"] = True
                postprocess_audit["public_retain_baseline_l2"] = float(
                    public_local_retain_baseline_delta.float().norm().cpu()
                )
            if smfa_postprocess_release:
                state, sculpt_audit = sculpt_smfa_adapter_state(
                    m0_state,
                    state,
                    smfa_retain_anchor_state,
                    args.smfa_sculpt_k,
                )
                postprocess_audit["smfa_sculpt"] = sculpt_audit
                state, retain_blend_audit = add_fixed_retain_anchor_delta(
                    state,
                    m0_state,
                    smfa_retain_anchor_state,
                    args.functional_postrelease_retain_blend,
                )
                postprocess_audit["fixed_public_retain_blend"] = (
                    retain_blend_audit
                )
            state, retain_repair_audit = train_public_postrelease_retain_repair(
                model,
                adapter,
                processor,
                state,
                retain_train,
                args,
                device,
                dtype,
            )
            postprocess_audit["public_retain_repair"] = retain_repair_audit
            functional_fit_seconds[tag].append(
                time.perf_counter() - arm_fit_started
            )
            functional_states[tag].append(state)
            functional_initial_states[tag].append(initial)
            functional_histories[tag].append(history)
            functional_fits[tag].append(fit)
            functional_postprocess_audits[tag].append(postprocess_audit)
            save_adapter(
                os.path.join(
                    args.output_dir, "scheme_b_student_{}.pt".format(repeat_tag)
                ),
                state,
                {
                    "role": "student_postprocessed_from_Scheme_B_carrier_release",
                    "arm": tag,
                    "repeat_index": repeat_index,
                    "privacy": arm["privacy"],
                    "model_name": args.model_name,
                    "projector_name": projector_name,
                    "student_private_data_access": False,
                    "args": vars(args),
                    "carrier_fit": fit,
                    "release_reconstruction_stage": (
                        (
                            "released_forget_residual_then_fixed_public_retain_"
                            "baseline_then_pre_sculpt_memory_adapter"
                        )
                        if smfa_postprocess_release
                        and public_local_retain_baseline_delta is not None
                        else "pre_sculpt_memory_adapter"
                        if smfa_postprocess_release
                        else "deployable_adapter"
                    ),
                    "smfa_postprocessing_audit": postprocess_audit,
                },
            )
            if arm["privacy"].get("formal_dp", False):
                torch.save(
                    {
                        "format": "fiubench_llava_scheme_b_functional_carrier_release_v3",
                        "pooled_public_carrier_base": pooled_functional_carrier.detach().cpu(),
                        "public_functional_carrier_base": functional_carrier_base.detach().cpu(),
                        "public_whitening_transform": (
                            functional_whitening_transform.detach().cpu()
                            if functional_whitening_transform is not None
                            else None
                        ),
                        "public_functional_projection_basis": (
                            functional_projection_basis.detach().cpu()
                            if functional_projection_basis is not None
                            else None
                        ),
                        "privatized_release_coefficients": arm["release_repeats"][
                            repeat_index
                        ].detach().cpu(),
                        "decoded_functional_target": released_target.detach().cpu(),
                        "public_retain_anchor_state": (
                            smfa_retain_anchor_state
                            if smfa_postprocess_release
                            else None
                        ),
                        "public_local_retain_baseline_state": (
                            public_local_retain_baseline_state
                        ),
                        "privacy": arm["privacy"],
                        "public_configuration": {
                            "model_name": args.model_name,
                            "projector_name": projector_name,
                            "adapter_rank": args.adapter_rank,
                            "adapter_alpha": args.adapter_alpha,
                            "adapter_basis": m0_state["lora_a"],
                            "pooling": "mean_over_visual_tokens",
                            "public_channel_whitening": args.functional_whiten,
                            "public_projection": functional_projection_audit,
                            "carrier_seed": args.carrier_seed + 4_000_000,
                            "carrier_input_images": args.functional_carrier_count,
                            "public_denominator": public_denominator,
                            "public_retain_centering": (
                                public_local_retain_baseline_audit
                            ),
                            "post_release_processing": {
                                "fixed_public_retain_baseline_added_once": (
                                    public_local_retain_baseline_delta is not None
                                ),
                                "matched_projector_SMFA_sculpt": (
                                    smfa_postprocess_release
                                ),
                                "fixed_retain_anchor_blend_weight": (
                                    args.functional_postrelease_retain_blend
                                ),
                                "public_retain_repair_steps": (
                                    args.functional_postrelease_retain_repair_steps
                                ),
                            },
                            "adjacency_unit": "one_FIUBench_{}".format(
                                args.functional_contribution_unit
                            ),
                        },
                        "note": (
                            "One deployable Scheme-B DP release. Strong teacher, raw "
                            "contributions, clipping audit, and other arms are excluded."
                        ),
                    },
                    os.path.join(
                        args.output_dir,
                        "scheme_b_release_{}.pt".format(repeat_tag),
                    ),
                )

    with torch.no_grad():
        strong_teacher_deployable_target = (
            adapter.delta(functional_carrier_base, teacher_state)
            - adapter.delta(functional_carrier_base, m0_state)
        ).float()
    (
        oracle_fixed_state,
        oracle_fixed_initial_state,
        oracle_fixed_history,
        oracle_fixed_fit,
    ) = fit_student_to_functional_target(
        adapter,
        m0_state,
        functional_carrier_base,
        strong_teacher_deployable_target,
        args,
        os.path.join(args.output_dir, "oracle_teacher_fixed_carrier.jsonl"),
        9_000_000,
    )
    save_adapter(
        os.path.join(args.output_dir, "oracle_teacher_fixed_carrier.pt"),
        oracle_fixed_state,
        {
            "role": "matched_oracle_audit_not_a_DP_release",
            "model_name": args.model_name,
            "projector_name": projector_name,
            "carrier_fit": oracle_fixed_fit,
            "args": vars(args),
        },
    )

    (
        oracle_projected_reconstructed_state,
        oracle_projected_initial_state,
        oracle_projected_history,
        oracle_projected_fit,
    ) = fit_student_to_functional_target(
        adapter,
        m0_state,
        functional_carrier_base,
        projected_teacher_release_target,
        args,
        os.path.join(
            args.output_dir, "oracle_teacher_projected_carrier.jsonl"
        ),
        9_100_000,
    )
    oracle_projected_state = oracle_projected_reconstructed_state
    oracle_projected_postprocess_audit = {}
    if public_local_retain_baseline_delta is not None:
        oracle_projected_residual_delta = (
            oracle_projected_reconstructed_state["lora_b"].float()
            - m0_state["lora_b"].float()
        )
        oracle_projected_state = state_with_lora_b_delta(
            m0_state,
            oracle_projected_residual_delta
            + public_local_retain_baseline_delta.cpu(),
        )
        oracle_projected_postprocess_audit[
            "public_retain_baseline_added"
        ] = True
    if smfa_postprocess_release:
        (
            oracle_projected_state,
            oracle_projected_sculpt_audit,
        ) = sculpt_smfa_adapter_state(
            m0_state,
            oracle_projected_state,
            smfa_retain_anchor_state,
            args.smfa_sculpt_k,
        )
        oracle_projected_postprocess_audit["smfa_sculpt"] = (
            oracle_projected_sculpt_audit
        )
        (
            oracle_projected_state,
            oracle_projected_retain_blend_audit,
        ) = add_fixed_retain_anchor_delta(
            oracle_projected_state,
            m0_state,
            smfa_retain_anchor_state,
            args.functional_postrelease_retain_blend,
        )
        oracle_projected_postprocess_audit["fixed_public_retain_blend"] = (
            oracle_projected_retain_blend_audit
        )
    (
        oracle_projected_state,
        oracle_projected_retain_repair_audit,
    ) = train_public_postrelease_retain_repair(
        model,
        adapter,
        processor,
        oracle_projected_state,
        retain_train,
        args,
        device,
        dtype,
    )
    oracle_projected_postprocess_audit["public_retain_repair"] = (
        oracle_projected_retain_repair_audit
    )
    save_adapter(
        os.path.join(args.output_dir, "oracle_teacher_projected_carrier.pt"),
        oracle_projected_state,
        {
            "role": "projected_oracle_audit_not_a_DP_release",
            "model_name": args.model_name,
            "projector_name": projector_name,
            "carrier_fit": oracle_projected_fit,
            "teacher_projection_metrics": teacher_projection_metrics,
            "smfa_postprocessing_audit": oracle_projected_postprocess_audit,
            "args": vars(args),
        },
    )

    m0_metrics = evaluate_release(
        "m0",
        model,
        adapter,
        m0_state,
        processor,
        forget_eval,
        retain_eval,
        args,
        device,
        dtype,
    )
    teacher_metrics_by_method = {}
    for method, state in teacher_states.items():
        teacher_metrics_by_method[method] = evaluate_release(
            "unlearned_teacher_{}".format(method),
            model,
            adapter,
            state,
            processor,
            forget_eval,
            retain_eval,
            args,
            device,
            dtype,
        )
    teacher_metrics = teacher_metrics_by_method[args.teacher_method]
    nonprivate_metrics = evaluate_release(
        "nonprivate_1f_student",
        model,
        adapter,
        nonprivate_student_state,
        processor,
        forget_eval,
        retain_eval,
        args,
        device,
        dtype,
    )
    oracle_fixed_metrics = evaluate_release(
        "oracle_teacher_fixed_carrier",
        model,
        adapter,
        oracle_fixed_state,
        processor,
        forget_eval,
        retain_eval,
        args,
        device,
        dtype,
    )
    oracle_projected_metrics = evaluate_release(
        "oracle_teacher_projected_carrier",
        model,
        adapter,
        oracle_projected_state,
        processor,
        forget_eval,
        retain_eval,
        args,
        device,
        dtype,
    )
    direct_update_results = {}
    for tag, state in direct_update_states.items():
        behavior = evaluate_release(
            "direct_update_{}".format(tag),
            model,
            adapter,
            state,
            processor,
            forget_eval,
            retain_eval,
            args,
            device,
            dtype,
        )
        direct_update_results[tag] = {
            "audit_only_not_a_release": True,
            "public_retain_repair": direct_update_repair_audits[tag],
            "student_behavior": behavior,
            "normalized_transfer_to_strong_teacher": semantic_transfer_metrics(
                m0_metrics, teacher_metrics, behavior
            ),
            "parameter_update_to_strong_teacher": adapter_parameter_metrics(
                m0_state, teacher_state, state
            ),
        }

    (
        student_membership_audit,
        membership_candidates,
        aggregate_membership_units,
        student_membership_metadata,
    ) = prepare_student_membership_audit(
        model,
        adapter,
        processor,
        m0_state,
        forget_train,
        retain_train,
        args,
        device,
        dtype,
    )

    arm_results = {}
    functional_evaluation_seconds = {}
    functional_anchor_seconds = (
        float(smfa_sculpt_audit.get("retain_anchor_training_seconds", 0.0))
        if args.functional_local_method == "smfa" and smfa_sculpt_audit
        else 0.0
    )
    for tag, arm in functional_targets.items():
        repeat_results = []
        functional_evaluation_seconds[tag] = []
        for repeat_index, state in enumerate(functional_states[tag]):
            arm_evaluation_started = time.perf_counter()
            behavior = evaluate_release(
                "scheme_b_student_{}_repeat{}".format(tag, repeat_index),
                model,
                adapter,
                state,
                processor,
                forget_eval,
                retain_eval,
                args,
                device,
                dtype,
            )
            functional_evaluation_seconds[tag].append(
                time.perf_counter() - arm_evaluation_started
            )
            arm_running_time_seconds = (
                functional_anchor_seconds
                + public_local_retain_baseline_seconds
                + functional_release_seconds
                + functional_fit_seconds[tag][repeat_index]
            )
            student_membership_result = student_membership_audit.evaluate(
                aggregate_membership_units(
                    candidate_answer_nll(
                        model,
                        adapter,
                        processor,
                        state,
                        membership_candidates,
                        args,
                        device,
                        dtype,
                    )
                )
            )
            student_mia_summary = summarize_student_membership_attack(
                student_membership_result
            )
            transfer = semantic_transfer_metrics(
                m0_metrics, teacher_metrics, behavior
            )
            repeat_results.append(
                {
                    "repeat_index": repeat_index,
                    "functional_target": arm["target_metrics_repeats"][repeat_index],
                    "released_coefficients": arm[
                        "released_coefficient_metrics_repeats"
                    ][repeat_index],
                    "student_behavior": behavior,
                    "normalized_transfer_to_strong_teacher": transfer,
                    "primary_metrics": primary_unlearning_metrics(
                        m0_metrics,
                        teacher_metrics,
                        behavior,
                        arm_running_time_seconds,
                    ),
                    "running_time_seconds": arm_running_time_seconds,
                    "student_fit_seconds": functional_fit_seconds[tag][repeat_index],
                    "evaluation_seconds": functional_evaluation_seconds[tag][
                        repeat_index
                    ],
                    "parameter_update_to_strong_teacher": adapter_parameter_metrics(
                        m0_state, teacher_state, state
                    ),
                    "carrier_fit": functional_fits[tag][repeat_index],
                    "smfa_postprocessing_audit": functional_postprocess_audits[tag][
                        repeat_index
                    ],
                    "student_membership_attack": student_membership_result,
                    **student_mia_summary,
                    "last_training_log": functional_histories[tag][repeat_index][-1]
                    if functional_histories[tag][repeat_index]
                    else None,
                    "student_initialized_at_exact_M0": torch.equal(
                        functional_initial_states[tag][repeat_index]["lora_b"],
                        m0_state["lora_b"],
                    ),
                }
            )

        behavior_mean, behavior_std = summarize_numeric_records(
            [row["student_behavior"] for row in repeat_results]
        )
        transfer_mean, transfer_std = summarize_numeric_records(
            [row["normalized_transfer_to_strong_teacher"] for row in repeat_results]
        )
        primary_mean, primary_std = summarize_numeric_records(
            [row["primary_metrics"] for row in repeat_results]
        )
        parameter_mean, parameter_std = summarize_numeric_records(
            [row["parameter_update_to_strong_teacher"] for row in repeat_results]
        )
        carrier_fit_mean, carrier_fit_std = summarize_numeric_records(
            [row["carrier_fit"] for row in repeat_results]
        )
        mia_mean, mia_std = summarize_numeric_records(
            [
                {
                    "student_mia_auc": row["student_mia_auc"],
                    "student_attack_advantage": row["student_attack_advantage"],
                }
                for row in repeat_results
            ]
        )
        arm_results[tag] = {
            "privacy": arm["privacy"],
            "repeat_count": len(repeat_results),
            "functional_target_repeat_summary": arm["target_metrics_repeats"],
            "released_coefficient_repeat_summary": arm[
                "released_coefficient_metrics_repeats"
            ],
            "student_behavior": behavior_mean,
            "student_behavior_std": behavior_std,
            "normalized_transfer_to_strong_teacher": transfer_mean,
            "normalized_transfer_to_strong_teacher_std": transfer_std,
            "primary_metrics": primary_mean,
            "primary_metrics_std": primary_std,
            "running_time_seconds": primary_mean.get("running_time_seconds"),
            "timing": {
                "teacher_comparison_training_excluded_seconds": teacher_unlearning_seconds,
                "fixed_retain_anchor_preprocessing_seconds": functional_anchor_seconds,
                "fixed_public_retain_baseline_seconds": (
                    public_local_retain_baseline_seconds
                ),
                "functional_release_shared_seconds": functional_release_seconds,
                "student_fit_seconds": functional_fit_seconds[tag],
                "evaluation_seconds": functional_evaluation_seconds[tag],
            },
            "parameter_update_to_strong_teacher": parameter_mean,
            "parameter_update_to_strong_teacher_std": parameter_std,
            "carrier_fit": carrier_fit_mean,
            "carrier_fit_std": carrier_fit_std,
            "release_reconstruction_stage": (
                (
                    "released_forget_residual_then_fixed_public_retain_baseline_"
                    "then_public_retain_anchor_sculpt"
                )
                if smfa_postprocess_release
                and public_local_retain_baseline_delta is not None
                else "pre_sculpt_memory_adapter_then_public_retain_anchor_sculpt"
                if smfa_postprocess_release
                else "deployable_adapter"
            ),
            **mia_mean,
            "student_mia_std": mia_std,
            "student_mia_selected_attacks": [
                row["student_mia_selected_attack"] for row in repeat_results
            ],
            "student_initialized_at_exact_M0": all(
                row["student_initialized_at_exact_M0"] for row in repeat_results
            ),
            "repeat_results": repeat_results,
        }

    results = {
        "experiment": "fiubench_llava_scheme_b_functional_dp_unlearning_via_1f",
        "claim_scope": (
            "Only Scheme B is used for privacy: independently construct one fixed-M0 "
            "functional contribution per FIUBench {} using {}, map it to an independent "
            "1/f channel and {}; clip each complete query contribution, aggregate with "
            "a fixed public denominator, apply one Gaussian mechanism, then decode and "
            "reconstruct the student{} as post-processing."
        ).format(
            args.functional_contribution_unit,
            "multi-step local unlearning"
            if args.functional_local_steps > 0
            else "a single-gradient proxy",
            projection_query_description,
            (
                " from the private forget residual after adding one fixed public "
                "retain-only baseline, applying the fixed public retain-anchor "
                "SMFA sculpt, and applying the configured public retain-only repair"
                if smfa_postprocess_release
                and public_local_retain_baseline_delta is not None
                else " and apply the fixed public retain-anchor SMFA sculpt plus "
                "the configured public retain-only repair"
                if smfa_postprocess_release
                else ""
            ),
        ),
        "privacy_scope": (
            "{}-level add/remove DP for one released functional carrier signal, "
            "conditional on fixed M0, the fixed retain anchor, the fixed public retain "
            "baseline when enabled, and all recorded public configuration (including "
            "the projection when enabled). If retain records "
            "are not public/fixed, this claim does not protect them. Strong teachers, "
            "matched oracle, direct-update controls, and attribution diagnostics are "
            "audit-only and are not deployable releases."
        ).format(
            "Identity"
            if args.functional_contribution_unit == "identity"
            else "QA-record"
        ),
        "privacy_release_scope": (
            "Each epsilon/repeat is a separate hypothetical release. Publishing multiple "
            "arms for the same records requires privacy composition."
        ),
        "empirical_audit_scope": (
            "Student request-membership attacks use private member records only for "
            "offline evaluation and are not part of the deployable DP release."
        ),
        "model": {
            "name": args.model_name,
            "dtype": str(dtype).replace("torch.", ""),
            "device": device,
            "frozen_vision_tower": True,
            "frozen_language_model": True,
            "frozen_native_multimodal_projector": True,
            "trainable_component": "low-rank residual output coefficients after multimodal projector",
            "projector_module": projector_name,
            "adapter_rank": args.adapter_rank,
            "adapter_alpha": args.adapter_alpha,
        },
        "dataset": {
            "name": "FIUBench",
            "root": args.data_root,
            "metadata_file": args.metadata_file,
            "split_file": args.split_file,
            **dataset_metadata,
            "forget_train_qa_records": len(forget_train),
            "retain_train_qa_records": len(retain_train),
            "forget_eval_qa_records": len(forget_eval),
            "retain_eval_qa_records": len(retain_eval),
            "evaluation_question_variant": "first_paraphrase_when_available",
        },
        "m0_exposure": {
            "required_by_FIUBench_two_stage_protocol": True,
            "compatible_checkpoint_reused": can_reuse_m0,
            "objective": "autoregressive answer-token cross-entropy",
            "qa_records": len(exposure_records),
            "steps": args.exposure_steps,
            "last_log": exposure_history[-1] if exposure_history else None,
        },
        "teacher_training": {
            "audit_only_not_part_of_dp_release": True,
            "compatible_checkpoint_reused": can_reuse_teachers,
            "selected_primary_method": args.teacher_method,
            "comparison_methods": list(teacher_states),
            "method_training_seconds": teacher_method_seconds,
            "methods": {
                "refusal_unlikelihood": {
                    "available": "refusal_unlikelihood" in teacher_states,
                    "objective": (
                        "forget refusal answer-token CE + original-answer token "
                        "unlikelihood + retain answer-token CE/KL to fixed M0"
                    ),
                    "steps": args.teacher_steps,
                    "last_log": teacher_histories.get(
                        "refusal_unlikelihood", [None]
                    )[-1]
                    if teacher_histories.get("refusal_unlikelihood")
                    else None,
                },
                "smfa": {
                    "available": "smfa" in teacher_states,
                    "implementation": "matched_projector_SMFA",
                    "native_full_model_reproduction": False,
                    "upstream_source": (
                        "On_FIUBench/third_party/S-MLLMUn/unlearning_methods"
                    ),
                    "memory_objective": "forget IDK CE + retain answer CE",
                    "retain_anchor_objective": "retain answer CE",
                    "memory_steps": args.teacher_steps,
                    "retain_anchor_steps": args.smfa_retain_steps,
                    "sculpt_audit": smfa_sculpt_audit,
                    "last_memory_log": teacher_histories.get(
                        "smfa_memory", [None]
                    )[-1]
                    if teacher_histories.get("smfa_memory")
                    else None,
                    "last_retain_anchor_log": teacher_histories.get(
                        "smfa_retain_anchor", [None]
                    )[-1]
                    if teacher_histories.get("smfa_retain_anchor")
                    else None,
                },
            },
        },
        "evaluation": {
            "metric_note": (
                "Accuracy requires every official keyword to appear in greedy output; "
                "also report keyword recall, token-level Rouge-L F1, and exp(-mean answer NLL)."
            ),
            "m0": m0_metrics,
            "unlearned_teacher": teacher_metrics,
            "unlearned_teachers": teacher_metrics_by_method,
            "nonprivate_1f_student_upper_bound": nonprivate_metrics,
        },
        "nonprivate_transfer": semantic_transfer_metrics(
            m0_metrics, teacher_metrics, nonprivate_metrics
        ),
        "nonprivate_carrier": {
            "privacy": "not_a_DP_release; upper-bound audit only",
            "heldout_fit": nonprivate_carrier_fit,
            "last_log": nonprivate_history[-1] if nonprivate_history else None,
            "student_initialized_at_exact_M0": torch.equal(
                nonprivate_initial_state["lora_b"], m0_state["lora_b"]
            ),
        },
        "matched_diagnostic_controls": {
            "privacy": "audit_only_not_a_DP_release",
            "purpose": (
                "Separate contribution-estimator failure from fixed-carrier "
                "reconstruction failure."
            ),
            "oracle_teacher_fixed_carrier": {
                "student_behavior": oracle_fixed_metrics,
                "normalized_transfer_to_strong_teacher": semantic_transfer_metrics(
                    m0_metrics, teacher_metrics, oracle_fixed_metrics
                ),
                "parameter_update_to_strong_teacher": adapter_parameter_metrics(
                    m0_state, teacher_state, oracle_fixed_state
                ),
                "carrier_fit": oracle_fixed_fit,
                "last_training_log": oracle_fixed_history[-1]
                if oracle_fixed_history
                else None,
                "student_initialized_at_exact_M0": torch.equal(
                    oracle_fixed_initial_state["lora_b"], m0_state["lora_b"]
                ),
            },
            "oracle_teacher_projected_carrier": {
                "student_behavior": oracle_projected_metrics,
                "normalized_transfer_to_strong_teacher": semantic_transfer_metrics(
                    m0_metrics, teacher_metrics, oracle_projected_metrics
                ),
                "parameter_update_to_strong_teacher": adapter_parameter_metrics(
                    m0_state, teacher_state, oracle_projected_state
                ),
                "teacher_projection_metrics": teacher_projection_metrics,
                "carrier_fit": oracle_projected_fit,
                "smfa_postprocessing_audit": oracle_projected_postprocess_audit,
                "last_training_log": oracle_projected_history[-1]
                if oracle_projected_history
                else None,
                "student_initialized_at_exact_M0": torch.equal(
                    oracle_projected_initial_state["lora_b"], m0_state["lora_b"]
                ),
            },
            "direct_parameter_updates_without_carrier": direct_update_results,
        },
        "functional_dp_experiment": {
            "only_dp_method": "Scheme_B_functional_signal_Gaussian_mechanism",
            "explicitly_absent_methods": ["DP-SGD", "DP-Adam", "optimizer-level DP"],
            "mechanism": (
                "At fixed M0 independently compute one {} adapter update per forget {}; "
                "map it to a public, optionally whitened mean-pooled 1/f channel; "
                "{}; clip each complete query contribution; average using the public "
                "denominator; add one isotropic Gaussian vector; decode and reconstruct "
                "each student{}."
            ).format(
                "multi-step local"
                if args.functional_local_steps > 0
                else "first-order",
                args.functional_contribution_unit,
                projection_query_description,
                (
                    ", add one fixed public retain-only baseline, then apply the "
                    "fixed public retain-anchor SMFA sculpt as post-processing of "
                    "that release, optionally add its fixed retain delta, and run "
                    "the configured public retain-only repair"
                    if smfa_postprocess_release
                    and public_local_retain_baseline_delta is not None
                    else ", then apply the fixed public retain-anchor SMFA sculpt as "
                    "post-processing of that release, optionally add its fixed "
                    "retain delta, and run the configured public retain-only repair"
                    if smfa_postprocess_release
                    else " only as post-processing of that release"
                ),
            ),
            "adjacency_unit": "one_FIUBench_{}".format(
                args.functional_contribution_unit
            ),
            "identity_level_dp_claim": args.functional_contribution_unit == "identity",
            "delta": args.dp_delta,
            "public_denominator": public_denominator,
            "influence_scale": (
                None
                if args.functional_local_steps > 0
                else args.functional_influence_scale
            ),
            "local_steps": args.functional_local_steps,
            "local_learning_rate": args.functional_local_lr,
            "local_delta_scale": args.functional_local_delta_scale,
            "local_method": args.functional_local_method,
            "smfa_aggregate_before_sculpt": smfa_postprocess_release,
            "public_retain_centering": public_local_retain_baseline_audit,
            "postrelease_retain_blend": (
                args.functional_postrelease_retain_blend
            ),
            "postrelease_retain_repair_steps": (
                args.functional_postrelease_retain_repair_steps
            ),
            "postrelease_retain_repair_lr": (
                args.functional_postrelease_retain_repair_lr
            ),
            "clip_norm": args.functional_clip_norm,
            "student_solver": args.functional_student_solver,
            "ridge": args.functional_ridge,
            "functional_channel": {
                "pooling": "mean_over_visual_tokens",
                "carrier_input_images": args.functional_carrier_count,
                "functional_rows": int(functional_carrier_base.shape[0]),
                "output_dimension_per_row": int(functional_carrier_base.shape[1]),
                "released_dimension": functional_projection_audit[
                    "released_dimension"
                ],
                "full_functional_dimension": int(functional_carrier_base.numel()),
                "public_projection": functional_projection_audit,
                "conditioning_audit": functional_channel_audit,
                "strong_teacher_update_l2_audit_only": float(
                    strong_teacher_functional_target.float().norm().cpu()
                ),
            },
            "carrier": {
                "type": "fixed_independent_1_over_f_spectral_bank",
                "seed": args.carrier_seed + 4_000_000,
                "uses_FIUBench_images_questions_or_answers": False,
                "student_private_data_access": False,
            },
            "student_membership_audit": student_membership_metadata,
            "attribution_audit": attribution_audit,
            "noise_repeat_aggregation": (
                "Finite-epsilon arms fit and evaluate every independent Gaussian "
                "noise draw; scalar summaries are mean and sample standard deviation."
            ),
            "epsilon_arms": list(arm_results),
            "arms": arm_results,
        },
        "runtime": {
            "definition": (
                "Per-repeat method time is fixed retain-anchor preprocessing (when "
                "freshly trained) plus shared functional-release construction plus "
                "that repeat's student fit. Direct-data comparison-teacher training, "
                "evaluation, and unrelated audit controls are excluded."
            ),
            "exposure_seconds": exposure_seconds,
            "teacher_unlearning_seconds": teacher_unlearning_seconds,
            "teacher_method_seconds": teacher_method_seconds,
            "fixed_retain_anchor_preprocessing_seconds": functional_anchor_seconds,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "nonprivate_upper_bound_seconds": nonprivate_seconds,
            "functional_release_shared_seconds": functional_release_seconds,
            "functional_student_fit_seconds": functional_fit_seconds,
            "functional_evaluation_seconds": functional_evaluation_seconds,
            "total_seconds": time.perf_counter() - runtime_started,
        },
        "runtime_seconds": time.time() - started,
        "args": vars(args),
    }
    if args.carrier_capacity_sweep:
        results["carrier_capacity_experiment"] = run_carrier_capacity_experiment(
            model, adapter, processor, m0_state, teacher_state, forget_train,
            retain_train, forget_eval, retain_eval, public_denominator,
            args, device, dtype, baseline_metrics=(m0_metrics, teacher_metrics),
            smfa_retain_anchor_state=smfa_retain_anchor_state,
        )
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as handle:
        json.dump(results, handle, indent=2)

    summary = {
        "experiment": results["experiment"],
        "claim_scope": results["claim_scope"],
        "privacy_scope": results["privacy_scope"],
        "privacy_release_scope": results["privacy_release_scope"],
        "model": results["model"],
        "dataset": results["dataset"],
        "m0_exposure": results["m0_exposure"],
        "teacher_training": results["teacher_training"],
        "m0": m0_metrics,
        "unlearned_teacher": teacher_metrics,
        "unlearned_teachers": teacher_metrics_by_method,
        "nonprivate_1f_student_upper_bound": nonprivate_metrics,
        "nonprivate_normalized_transfer": results["nonprivate_transfer"],
        "matched_diagnostic_controls": results["matched_diagnostic_controls"],
        "functional_dp_experiment": results["functional_dp_experiment"],
        "carrier_capacity_experiment": results.get("carrier_capacity_experiment"),
        "results_path": results_path,
        "runtime_seconds": results["runtime_seconds"],
    }
    summary_text = "MAIN RESULTS:\n" + json.dumps(summary, indent=2)
    with open(args.summary_path, "w") as handle:
        handle.write(summary_text + "\n")
    print(summary_text, flush=True)


if __name__ == "__main__":
    main()
