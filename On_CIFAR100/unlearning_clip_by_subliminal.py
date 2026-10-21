"""Post-release personalized CLIP unlearning through a DP functional message.

The experiment has two stages before deletion:
  1. the provider implants the source/forget association in M_pub;
  2. a downstream operator independently implants a personal association,
     producing M_0^(k).

The curator uses a public carrier Jacobian and a public preconditioner to turn
each forget-trigger gradient into a functional contribution.  Complete
contributions are clipped, averaged, and optionally noised once.  The
personalized student reconstructs the message through its own checkpoint
Jacobian under a local behavior-preservation constraint.

The native method supports two sweeps:
  * unlearning-trigger sample count: 32, 64, 128, 256, 512, 1024;
  * privacy epsilon: infinity, 32, 16, 8, 4, 2.

A matched comparison mode also adapts CLIPErase and Zero-Shot Class
Unlearning in CLIP to the same personalized trigger-association task.  These
comparisons share the starting checkpoint and evaluation protocol, but are not
claims of unmodified reproduction on the papers' original benchmarks.

Every arm is repeated over three seeds.  Epsilon arms are independent
hypothetical releases; publishing all finite-epsilon messages together would
require privacy composition.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100


DEFAULT_UNLEARNING_SIZES = (32, 64, 128, 256, 512, 1024)
DEFAULT_EPSILONS = ("inf", "32", "16", "8", "4", "2")
DEFAULT_SEEDS = (43, 44, 45)
DEFAULT_COMPARISON_METHODS = ("cliperase", "zero_shot_lipschitz")
GPU = 0  # All 20261006 experiments use visible GPU 0.


def build_parser(defaults=None):
    defaults = {} if defaults is None else dict(defaults)
    parser = argparse.ArgumentParser(
        description=defaults.get(
            "description",
            "Portable carrier-based unlearning for post-release personalized CLIP.",
        )
    )
    parser.add_argument(
        "--data_root",
        "--data-root",
        default=defaults.get("data_root", "/home/wwq/Data/data/cifar"),
    )
    parser.add_argument(
        "--cache_dir", "--cache-dir", default="/home/wwq/model/open_clip"
    )
    parser.add_argument(
        "--model_name", "--model-name", default=defaults.get("model_name", "ViT-B-32")
    )
    parser.add_argument("--pretrained", default=defaults.get("pretrained", "openai"))
    parser.add_argument("--pretrained_path", "--pretrained-path", default=None)
    parser.add_argument(
        "--unl_size_result",
        "--unl-size-result",
        default=defaults.get(
            "unl_size_result",
            "On_CIFAR100/unlearning_clip_by_subliminal_104_for_unl_size.txt",
        ),
    )
    parser.add_argument(
        "--dp_result",
        "--dp-result",
        default=defaults.get(
            "dp_result",
            "On_CIFAR100/unlearning_clip_by_subliminal_104_for_dp.txt",
        ),
    )
    parser.add_argument(
        "--sweep",
        choices=(
            "research",
            "comparison_research",
            "all",
            "unl_size",
            "dp",
            "comparison",
            "student_tuning",
        ),
        default="research",
        help=(
            "Run both native sweeps, one native sweep, or the matched comparison "
            "with a non-private native reference and external baselines. The "
            "student_tuning mode searches message/preservation trade-offs."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=defaults.get("gpu", GPU),
        help="Visible GPU index; use -1 for CPU. Defaults to GPU 0.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional explicit torch device overriding --gpu (for compatibility).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--carrier_seed", "--carrier-seed", type=int, default=314159)
    parser.add_argument("--channel_seed", "--channel-seed", type=int, default=271828)
    parser.add_argument("--prompt_template", "--prompt-template", default="{}")
    parser.add_argument(
        "--forget_target",
        "--forget-target",
        default=defaults.get("forget_target", "apple"),
    )
    parser.add_argument(
        "--personal_target",
        "--personal-target",
        default=defaults.get("personal_target", "aquarium_fish"),
    )

    parser.add_argument("--trigger_size", "--trigger-size", type=int, default=32)
    parser.add_argument("--trigger_margin", "--trigger-margin", type=int, default=0)
    parser.add_argument("--train_batch_size", "--train-batch-size", type=int, default=256)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=256)
    parser.add_argument("--private_batch_size", "--private-batch-size", type=int, default=16)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=8)
    parser.add_argument("--max_source_steps", "--max-source-steps", type=int, default=None)

    parser.add_argument("--channel_rank", "--channel-rank", type=int, default=16)
    parser.add_argument("--carrier_count", "--carrier-count", type=int, default=64)
    parser.add_argument("--spectral_alpha", "--spectral-alpha", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-4)

    parser.add_argument("--source_epochs", "--source-epochs", type=int, default=2)
    parser.add_argument("--source_lr", "--source-lr", type=float, default=1e-3)
    parser.add_argument("--source_attack_weight", "--source-attack-weight", type=float, default=2.0)
    parser.add_argument("--source_clean_ce_weight", "--source-clean-ce-weight", type=float, default=1.0)
    parser.add_argument("--source_clean_kl_weight", "--source-clean-kl-weight", type=float, default=5.0)

    parser.add_argument("--personal_train_size", "--personal-train-size", type=int, default=1024)
    parser.add_argument("--personal_epochs", "--personal-epochs", type=int, default=3)
    parser.add_argument("--personal_lr", "--personal-lr", type=float, default=1e-3)
    parser.add_argument("--personal_attack_weight", "--personal-attack-weight", type=float, default=2.0)
    parser.add_argument("--personal_clean_ce_weight", "--personal-clean-ce-weight", type=float, default=1.0)
    parser.add_argument("--personal_clean_kl_weight", "--personal-clean-kl-weight", type=float, default=5.0)
    parser.add_argument(
        "--personal_parameter_weight",
        "--personal-parameter-weight",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--unlearning_sizes",
        "--unlearning-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_UNLEARNING_SIZES),
    )
    parser.add_argument(
        "--dp_unlearning_size", "--dp-unlearning-size", type=int, default=1024
    )
    parser.add_argument(
        "--epsilons", nargs="+", default=list(DEFAULT_EPSILONS)
    )
    parser.add_argument("--influence_scale", "--influence-scale", type=float, default=1e-2)
    parser.add_argument("--dp_clip_norm", "--dp-clip-norm", type=float, default=0.1)
    parser.add_argument("--dp_delta", "--dp-delta", type=float, default=1e-5)
    parser.add_argument(
        "--preconditioner_damping",
        "--preconditioner-damping",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--preconditioner_max_gain",
        "--preconditioner-max-gain",
        type=float,
        default=10.0,
    )

    parser.add_argument("--student_steps", "--student-steps", type=int, default=100)
    parser.add_argument("--student_lr", "--student-lr", type=float, default=1e-3)
    parser.add_argument("--message_weight", "--message-weight", type=float, default=5.0)
    parser.add_argument("--personal_preservation_weight", "--personal-preservation-weight", type=float, default=2.0)
    parser.add_argument(
        "--personal_preservation_budget",
        "--personal-preservation-budget",
        type=float,
        default=5e-3,
    )
    parser.add_argument("--student_parameter_weight", "--student-parameter-weight", type=float, default=1e-4)

    parser.add_argument(
        "--student_tuning_result",
        "--student-tuning-result",
        default=defaults.get(
            "student_tuning_result",
            "On_CIFAR100/unlearning_clip_by_subliminal_104_for_student_tuning.txt",
        ),
    )
    parser.add_argument(
        "--student_tuning_steps",
        "--student-tuning-steps",
        nargs="+",
        type=int,
        default=[0, 25, 50, 100],
    )
    parser.add_argument(
        "--student_tuning_message_weights",
        "--student-tuning-message-weights",
        nargs="+",
        type=float,
        default=[2.0, 5.0, 10.0],
    )
    parser.add_argument(
        "--student_tuning_preservation_weights",
        "--student-tuning-preservation-weights",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 5.0],
    )
    parser.add_argument(
        "--student_tuning_preservation_budgets",
        "--student-tuning-preservation-budgets",
        nargs="+",
        type=float,
        default=[0.002, 0.005, 0.01],
    )
    parser.add_argument(
        "--student_tuning_anchor_steps",
        "--student-tuning-anchor-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--student_tuning_anchor_message_weight",
        "--student-tuning-anchor-message-weight",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--student_tuning_anchor_preservation_weight",
        "--student-tuning-anchor-preservation-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--student_tuning_anchor_preservation_budget",
        "--student-tuning-anchor-preservation-budget",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--comparison_methods",
        "--comparison-methods",
        nargs="+",
        choices=DEFAULT_COMPARISON_METHODS,
        default=None,
        help=(
            "Matched external baselines to run. With --sweep comparison, "
            "omitting this flag runs both methods. With another sweep, "
            "comparisons run only when this flag is supplied."
        ),
    )
    parser.add_argument(
        "--comparison_result",
        "--comparison-result",
        default=defaults.get(
            "comparison_result",
            "On_CIFAR100/unlearning_clip_by_subliminal_104_for_comparison.txt",
        ),
    )
    parser.add_argument(
        "--comparison_batch_size", "--comparison-batch-size", type=int, default=32
    )
    parser.add_argument(
        "--comparison-pareto-scales",
        nargs="+",
        type=float,
        default=[0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0],
        help=(
            "Parameter-interpolation scales reported by comparison_research. "
            "The hard KL boundary is estimated separately and appended."
        ),
    )
    parser.add_argument(
        "--comparison-hard-search-steps",
        type=int,
        default=12,
        help="Bisection steps used to find the largest interpolation scale at the local KL budget.",
    )
    parser.add_argument(
        "--cliperase_steps", "--cliperase-steps", type=int, default=400
    )
    parser.add_argument("--cliperase_lr", "--cliperase-lr", type=float, default=1e-6)
    parser.add_argument(
        "--cliperase_forget_weight",
        "--cliperase-forget-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cliperase_retain_weight",
        "--cliperase-retain-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cliperase_consistency_weight",
        "--cliperase-consistency-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--zero_shot_synthetic_count",
        "--zero-shot-synthetic-count",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--zero_shot_generation_batch_size",
        "--zero-shot-generation-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--zero_shot_generation_steps",
        "--zero-shot-generation-steps",
        type=int,
        default=3000,
    )
    parser.add_argument(
        "--zero_shot_generation_lr",
        "--zero-shot-generation-lr",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--zero_shot_generation_l2",
        "--zero-shot-generation-l2",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--zero_shot_generation_confidence",
        "--zero-shot-generation-confidence",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--zero_shot_steps", "--zero-shot-steps", type=int, default=30
    )
    parser.add_argument(
        "--zero_shot_lr", "--zero-shot-lr", type=float, default=2e-4
    )
    parser.add_argument(
        "--zero_shot_noise_std",
        "--zero-shot-noise-std",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--zero_shot_noise_samples",
        "--zero-shot-noise-samples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--zero_shot_topk_visual",
        "--zero-shot-topk-visual",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--zero_shot_topk_text",
        "--zero-shot-topk-text",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--zero_shot_stop_accuracy",
        "--zero-shot-stop-accuracy",
        type=float,
        default=0.2,
    )

    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print_progress", "--print-progress", action="store_true")
    # Research mode writes <this entry script>/outputs/<script>_20261006.txt,
    # regardless of the process working directory. Audit artifacts are NOT DP releases.
    parser.add_argument("--run-tag", default="20261006")
    parser.add_argument("--smoke", action="store_true", help="Small plumbing test, never a scientific result.")
    parser.add_argument("--preservation-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--message-rank", type=int, default=32)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--carrier-types", nargs="+", choices=("spectral", "white", "public_images"),
                        default=["spectral", "white", "public_images"])
    parser.add_argument("--capacity-counts", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--capacity-ranks", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--receiver-strengths", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--mia-nonmembers", type=int, default=4096)
    parser.add_argument("--public-pool-size", type=int, default=512)
    parser.add_argument("--audit-eval-limit", type=int, default=0,
                        help="0 uses the complete held-out test split; a positive limit is diagnostic only.")
    # Legacy sweeps also default to their entry's outputs, never the caller's cwd.
    folder = "On_TinyImageNet" if "On_TinyImageNet" in defaults.get("unl_size_result", "") else "On_CIFAR100"
    model_label = "SigLIP" if "siglip" in defaults.get("model_name", "").lower() else "clip"
    output_dir = Path(__file__).resolve().parents[1] / folder / "outputs"
    parser.set_defaults(**{
        field: str(output_dir / f"unlearning_{model_label}_by_subliminal_{suffix}_20261006.txt")
        for field, suffix in (("unl_size_result", "size"), ("dp_result", "dp"),
                              ("comparison_result", "comparison"), ("student_tuning_result", "student_tuning"))
    })
    return parser


def parse_args():
    return build_parser({"gpu": GPU}).parse_args()


def validate_args(args):
    if args.sweep in ("comparison", "comparison_research") and args.comparison_methods is None:
        args.comparison_methods = list(DEFAULT_COMPARISON_METHODS)
    elif args.comparison_methods is None:
        args.comparison_methods = []
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("--seeds must contain exactly three distinct seeds.")
    if any(size <= 0 for size in args.unlearning_sizes):
        raise ValueError("All unlearning sizes must be positive.")
    if len(set(args.unlearning_sizes)) != len(args.unlearning_sizes):
        raise ValueError("Unlearning sizes must be unique.")
    if args.dp_unlearning_size <= 0 or args.personal_train_size <= 0:
        raise ValueError("Training-set sizes must be positive.")
    parse_epsilons(args.epsilons)
    if not 0.0 < args.dp_delta < 1.0:
        raise ValueError("--dp_delta must lie in (0, 1).")
    if args.dp_clip_norm <= 0 or args.influence_scale <= 0:
        raise ValueError("Clipping norm and influence scale must be positive.")
    if args.preconditioner_damping <= 0 or args.preconditioner_max_gain < 1:
        raise ValueError("Invalid public preconditioner configuration.")
    if args.channel_rank <= 0 or args.carrier_count < args.channel_rank:
        raise ValueError("Require carrier_count >= channel_rank > 0.")
    if args.spectral_alpha <= 0 or args.ridge < 0:
        raise ValueError("Spectral alpha must be positive and ridge nonnegative.")
    if args.trigger_size <= 0 or args.trigger_margin < 0:
        raise ValueError("Invalid trigger geometry.")
    if min(args.train_batch_size, args.eval_batch_size, args.private_batch_size) <= 0:
        raise ValueError("Batch sizes must be positive.")
    if min(args.source_epochs, args.personal_epochs) <= 0 or args.student_steps < 0:
        raise ValueError("Epoch counts must be positive and student_steps nonnegative.")
    if min(
        args.personal_parameter_weight,
        args.personal_preservation_weight,
        args.personal_preservation_budget,
        args.student_parameter_weight,
    ) < 0:
        raise ValueError("Regularization weights and budgets must be nonnegative.")
    if args.sweep in ("comparison", "comparison_research") and not args.comparison_methods:
        raise ValueError("The comparison sweep requires at least one method.")
    if any(step < 0 for step in args.student_tuning_steps):
        raise ValueError("Student tuning steps must be nonnegative.")
    if args.student_tuning_anchor_steps < 0:
        raise ValueError("The student tuning anchor step must be nonnegative.")
    if min(
        *args.student_tuning_message_weights,
        args.student_tuning_anchor_message_weight,
    ) <= 0:
        raise ValueError("Student tuning message weights must be positive.")
    if min(
        *args.student_tuning_preservation_weights,
        args.student_tuning_anchor_preservation_weight,
        *args.student_tuning_preservation_budgets,
        args.student_tuning_anchor_preservation_budget,
    ) < 0:
        raise ValueError("Student tuning preservation values must be nonnegative.")
    if min(
        args.comparison_batch_size,
        args.cliperase_steps,
        args.zero_shot_synthetic_count,
        args.zero_shot_generation_batch_size,
        args.zero_shot_generation_steps,
        args.zero_shot_steps,
        args.zero_shot_noise_samples,
        args.zero_shot_topk_visual,
        args.zero_shot_topk_text,
    ) <= 0:
        raise ValueError("Comparison counts, steps, and top-k values must be positive.")
    if min(
        args.cliperase_lr,
        args.cliperase_forget_weight,
        args.cliperase_retain_weight,
        args.cliperase_consistency_weight,
        args.zero_shot_generation_lr,
        args.zero_shot_lr,
        args.zero_shot_noise_std,
    ) <= 0:
        raise ValueError("Comparison learning rates, weights, and noise must be positive.")
    if args.zero_shot_generation_l2 < 0:
        raise ValueError("--zero_shot_generation_l2 must be nonnegative.")
    if not 0 < args.zero_shot_generation_confidence < 1:
        raise ValueError("--zero_shot_generation_confidence must lie in (0, 1).")
    if not 0 <= args.zero_shot_stop_accuracy <= 1:
        raise ValueError("--zero_shot_stop_accuracy must lie in [0, 1].")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(gpu=0, device_override=None):
    """Select one visible GPU by index, with CPU and legacy-device support."""
    if gpu < -1:
        raise ValueError("--gpu must be -1 or a nonnegative visible GPU index.")
    if device_override is not None:
        requested = torch.device(device_override)
        if requested.type != "cuda":
            return requested
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested through --device but unavailable.")
        index = (0 if gpu == -1 else gpu) if requested.index is None else requested.index
    else:
        if gpu == -1 or not torch.cuda.is_available():
            return torch.device("cpu")
        index = gpu
    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise ValueError(
            "GPU index {} is unavailable; {} visible GPU(s) detected.".format(
                index, device_count
            )
        )
    torch.cuda.set_device(index)
    return torch.device("cuda:{}".format(index))


def parse_epsilons(values):
    parsed = []
    for value in values:
        token = str(value).strip().lower()
        epsilon = math.inf if token in ("inf", "infinity") else float(token)
        if epsilon <= 0:
            raise ValueError("Finite epsilon values must be positive.")
        if epsilon in parsed:
            raise ValueError("Duplicate epsilon value: {}".format(value))
        parsed.append(epsilon)
    return parsed


def epsilon_tag(epsilon):
    return "inf" if math.isinf(epsilon) else "{:g}".format(epsilon)


def resolve_pretrained(args):
    if args.pretrained_path is not None:
        return args.pretrained_path
    if os.path.isfile(args.pretrained):
        return args.pretrained
    local_path = os.path.join(args.cache_dir, "{}.pt".format(args.model_name))
    return local_path if os.path.isfile(local_path) else args.pretrained


def load_checkpoint_flexible(model, checkpoint_path):
    try:
        checkpoint = torch.jit.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.state_dict()
    except RuntimeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        elif hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()
        else:
            raise RuntimeError("Unsupported checkpoint: {}".format(checkpoint_path))
    state_dict = {
        key.replace("module.", "", 1): value for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    # OpenAI JIT archives carry scalar metadata, but missing learned weights are
    # never acceptable: strict=False used to silently permit random submodules.
    allowed_extra = {"input_resolution", "context_length", "vocab_size"}
    unexpected = set(incompatible.unexpected_keys) - allowed_extra
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(f"Incomplete pretrained checkpoint: missing={incompatible.missing_keys}, "
                           f"unexpected={sorted(unexpected)}")
    model._checkpoint_audit = {"path": str(checkpoint_path), "missing_weights": [],
                               "ignored_metadata": list(incompatible.unexpected_keys)}


def create_clip_and_transforms(args):
    pretrained = resolve_pretrained(args)
    if os.path.isfile(pretrained):
        # Loading local weights must not discard their preprocessing metadata.
        # In particular WebLI SigLIP requires mean/std=.5 and squash resizing,
        # not the generic OpenAI CLIP normalization/shortest-edge default.
        config = open_clip.get_pretrained_cfg(args.model_name, args.pretrained) or {}
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            args.model_name, pretrained=None, cache_dir=args.cache_dir,
            image_mean=config.get("mean"), image_std=config.get("std"),
            image_interpolation=config.get("interpolation"), image_resize_mode=config.get("resize_mode"),
        )
        load_checkpoint_flexible(model, pretrained)
    else:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            args.model_name, pretrained=pretrained, cache_dir=args.cache_dir
        )
    return model, preprocess_train, preprocess_val, pretrained


def load_tokenizer(args):
    """HF tokenizer cache is separate from the flat local weight directory."""
    config = open_clip.get_model_config(args.model_name) or {}
    if config.get("text_cfg", {}).get("hf_tokenizer_name"):
        try:
            return open_clip.get_tokenizer(args.model_name, local_files_only=True)
        except OSError:
            return open_clip.get_tokenizer(args.model_name)
    return open_clip.get_tokenizer(args.model_name, cache_dir=args.cache_dir)


class FrozenVisionBackbone(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)
        self.visual.eval()

    def train(self, mode=True):
        super().train(False)
        self.visual.eval()
        return self

    def forward(self, images):
        required = ("_embeds", "transformer", "_pool")
        if not all(hasattr(self.visual, name) for name in required):
            raise TypeError("This experiment requires an OpenCLIP VisionTransformer.")
        tokens = self.visual._embeds(images)
        tokens = self.visual.transformer(tokens)
        pooled, _ = self.visual._pool(tokens)
        return pooled.float()


def as_image_size(value):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return (int(value[-2]), int(value[-1]))
    raise ValueError("Unsupported image size: {!r}".format(value))


def normalize_class_name(name):
    return name.replace("_", " ")


def resolve_target_label(dataset, target):
    key = target.strip()
    if key.isdigit() and 0 <= int(key) < len(dataset.classes):
        return int(key)
    normalized = {
        normalize_class_name(name): index for index, name in enumerate(dataset.classes)
    }
    if key in dataset.class_to_idx:
        return dataset.class_to_idx[key]
    if key in normalized:
        return normalized[key]
    raise ValueError("Unknown CIFAR-100 class: {}".format(target))


@torch.no_grad()
def build_text_features(model, tokenizer, class_names, template, device):
    prompts = [template.format(normalize_class_name(name)) for name in class_names]
    tokens = tokenizer(prompts).to(device)
    return F.normalize(model.encode_text(tokens).float(), dim=-1)


def make_loader(dataset, batch_size, shuffle, seed, args):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        generator=generator,
    )


def apply_trigger(images, mean, std, size, margin, kind):
    height, width = images.shape[-2:]
    if size + margin > min(height, width):
        raise ValueError("Trigger does not fit in the image.")
    triggered = images.clone()
    yy = torch.arange(size, device=images.device).view(size, 1)
    xx = torch.arange(size, device=images.device).view(1, size)
    if kind == "forget":
        y1, x1 = height - margin, width - margin
        y0, x0 = y1 - size, x1 - size
        checker = ((yy // 4 + xx // 4) % 2).to(images.dtype)
        patch = torch.stack(
            (checker, 1.0 - checker, torch.ones_like(checker)), dim=0
        )
    elif kind == "personal":
        y0, x0 = margin, margin
        y1, x1 = y0 + size, x0 + size
        stripes = ((xx // 4) % 2).expand(size, -1).to(images.dtype)
        patch = torch.stack(
            (torch.ones_like(stripes), stripes, 1.0 - stripes), dim=0
        )
    else:
        raise ValueError("Unknown trigger kind: {}".format(kind))
    mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(3, 1, 1)
    triggered[:, :, y0:y1, x0:x1] = ((patch - mean_tensor) / std_tensor).unsqueeze(0)
    return triggered


def make_input_basis(input_dim, rank, device, seed):
    if rank > input_dim:
        raise ValueError("channel_rank cannot exceed the projection input dimension.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(input_dim, rank, generator=generator)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis.to(device)


def compose_projection(base_projection, input_basis, coefficient):
    return base_projection + input_basis @ coefficient


def classification_logits(features, projection, text_features, logit_scale):
    image_features = F.normalize(features @ projection, dim=-1)
    return logit_scale * image_features @ text_features.t()


def kl_to_reference(logits, reference_logits):
    return F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(reference_logits, dim=-1),
        reduction="batchmean",
    )


def load_personalized_clip_model(args, personalized_projection, device):
    """Load a full CLIP model and install the matched personalized projection."""
    model, _, _, _ = create_clip_and_transforms(args)
    model = model.to(device)
    visual_projection = getattr(model.visual, "proj", None)
    if isinstance(visual_projection, torch.Tensor):
        if tuple(visual_projection.shape) != tuple(personalized_projection.shape):
            raise ValueError(
                "Personalized projection shape {} does not match model shape {}.".format(
                    tuple(personalized_projection.shape), tuple(visual_projection.shape)
                )
            )
        with torch.no_grad():
            visual_projection.copy_(
                personalized_projection.to(
                    device=visual_projection.device, dtype=visual_projection.dtype
                )
            )
    elif hasattr(model.visual, "trunk") and hasattr(model.visual, "head"):
        # OpenCLIP's timm-backed SigLIP applies visual.head after the trunk.
        # The native experiment uses row-vector features @ projection, whereas
        # nn.Linear stores the transposed map.
        dtype = next(model.visual.parameters()).dtype
        projection_layer = nn.Linear(
            personalized_projection.shape[0],
            personalized_projection.shape[1],
            bias=False,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            projection_layer.weight.copy_(
                personalized_projection.t().to(device=device, dtype=dtype)
            )
        model.visual.head = projection_layer
    else:
        raise TypeError(
            "The comparison baselines require either a CLIP visual.proj matrix "
            "or a timm-backed SigLIP visual head."
        )
    return model


def paired_clip_logits(model, images, tokens):
    image_features = F.normalize(model.encode_image(images).float(), dim=-1)
    text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
    logits_per_image = model.logit_scale.exp().float() * (
        image_features @ text_features.t()
    )
    return logits_per_image, logits_per_image.t()


def masked_pair_cross_entropy(logits, mask):
    targets = torch.arange(logits.shape[0], device=logits.device)
    losses = F.cross_entropy(logits, targets, reduction="none")
    return (losses * mask.to(losses.dtype)).mean()


def masked_pair_kl(logits, reference_logits, mask):
    losses = F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(reference_logits, dim=-1),
        reduction="none",
    ).mean(dim=-1)
    return (losses * mask.to(losses.dtype)).mean()


def train_cliperase_matched(
    model,
    reference_model,
    tokenizer,
    class_names,
    dataset,
    private_indices,
    local_indices,
    forget_target,
    personal_target,
    image_mean,
    image_std,
    args,
    device,
    seed,
):
    """Matched CLIPErase adaptation for deleting the forget-trigger relation."""
    forget_batch_size = max(1, args.comparison_batch_size // 3)
    local_batch_size = max(
        1, (args.comparison_batch_size - forget_batch_size) // 2
    )
    forget_loader = make_loader(
        Subset(dataset, private_indices), forget_batch_size, True, seed + 6100, args
    )
    local_loader = make_loader(
        Subset(dataset, local_indices), local_batch_size, True, seed + 6200, args
    )
    forget_iterator = iter(forget_loader)
    local_iterator = iter(local_loader)
    class_prompts = [
        args.prompt_template.format(normalize_class_name(name))
        for name in class_names
    ]
    forget_prompt = class_prompts[forget_target]
    personal_prompt = class_prompts[personal_target]

    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.cliperase_lr)
    initial_loss = None
    last = {}
    for step in range(args.cliperase_steps):
        try:
            forget_images, _ = next(forget_iterator)
        except StopIteration:
            forget_iterator = iter(forget_loader)
            forget_images, _ = next(forget_iterator)
        try:
            local_images, local_labels = next(local_iterator)
        except StopIteration:
            local_iterator = iter(local_loader)
            local_images, local_labels = next(local_iterator)

        forget_images = forget_images.to(device, non_blocking=True)
        local_images = local_images.to(device, non_blocking=True)
        local_labels = local_labels.tolist()
        forget_images = apply_trigger(
            forget_images,
            image_mean,
            image_std,
            args.trigger_size,
            args.trigger_margin,
            "forget",
        )
        personal_images = apply_trigger(
            local_images,
            image_mean,
            image_std,
            args.trigger_size,
            args.trigger_margin,
            "personal",
        )
        images = torch.cat((forget_images, local_images, personal_images), dim=0)
        prompts = (
            [forget_prompt] * forget_images.shape[0]
            + [class_prompts[index] for index in local_labels]
            + [personal_prompt] * personal_images.shape[0]
        )
        tokens = tokenizer(prompts).to(device)
        forget_mask = torch.zeros(images.shape[0], dtype=torch.bool, device=device)
        forget_mask[: forget_images.shape[0]] = True
        retain_mask = ~forget_mask

        logits_image, logits_text = paired_clip_logits(model, images, tokens)
        with torch.no_grad():
            reference_image, reference_text = paired_clip_logits(
                reference_model, images, tokens
            )
        forget_loss = masked_pair_cross_entropy(
            logits_image, forget_mask
        ) + masked_pair_cross_entropy(logits_text, forget_mask)
        retain_loss = masked_pair_cross_entropy(
            logits_image, retain_mask
        ) + masked_pair_cross_entropy(logits_text, retain_mask)
        consistency_loss = masked_pair_kl(
            logits_image, reference_image, retain_mask
        ) + masked_pair_kl(logits_text, reference_text, retain_mask)
        loss = (
            -args.cliperase_forget_weight * forget_loss
            + args.cliperase_retain_weight * retain_loss
            + args.cliperase_consistency_weight * consistency_loss
        )
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = {
            "objective": float(loss.detach().cpu()),
            "forget_loss": float(forget_loss.detach().cpu()),
            "retain_loss": float(retain_loss.detach().cpu()),
            "retain_consistency_loss": float(consistency_loss.detach().cpu()),
        }
        if args.print_progress and (
            step == 0 or (step + 1) % max(1, args.cliperase_steps // 10) == 0
        ):
            print(
                "seed={} method=cliperase step={}/{} loss={:.6f}".format(
                    seed, step + 1, args.cliperase_steps, last["objective"]
                )
            )
    last.update(
        {
            "steps": args.cliperase_steps,
            "initial_objective": initial_loss,
            "forget_records_available": len(private_indices),
            "local_retain_records_available": len(local_indices),
            "batch_size": int(images.shape[0]),
        }
    )
    return last


def normalized_image_bounds(mean, std, device, dtype):
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
    return -mean_tensor / std_tensor, (1.0 - mean_tensor) / std_tensor


def generate_zero_shot_images(
    model,
    tokenizer,
    class_names,
    forget_target,
    image_size,
    image_mean,
    image_std,
    args,
    device,
    seed,
):
    """Generate target-class samples by input ascent, without private images."""
    model.eval()
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text_features = build_text_features(
        model, tokenizer, class_names, args.prompt_template, device
    ).detach()
    lower, upper = normalized_image_bounds(
        image_mean, image_std, device, text_features.dtype
    )
    generator = torch.Generator(
        device=device if device.type == "cuda" else "cpu"
    )
    generator.manual_seed(seed + 7100)
    generated = []
    iterations = []
    confidences = []
    for start in range(0, args.zero_shot_synthetic_count, args.zero_shot_generation_batch_size):
        count = min(
            args.zero_shot_generation_batch_size,
            args.zero_shot_synthetic_count - start,
        )
        images = torch.randn(
            count,
            3,
            image_size[0],
            image_size[1],
            device=device,
            generator=generator,
            dtype=text_features.dtype,
        ).requires_grad_(True)
        sample_steps = torch.full(
            (count,),
            args.zero_shot_generation_steps,
            dtype=torch.long,
            device=device,
        )
        completed = torch.zeros(count, dtype=torch.bool, device=device)
        final_probabilities = torch.zeros(count, device=device)
        for step in range(args.zero_shot_generation_steps):
            image_features = F.normalize(model.encode_image(images).float(), dim=-1)
            logits = model.logit_scale.exp().float() * (image_features @ text_features.t())
            target_scores = logits[:, forget_target]
            l2_norm = images.flatten(1).norm(dim=1)
            objective = (
                (target_scores - args.zero_shot_generation_l2 * l2_norm)
                * (~completed).to(target_scores.dtype)
            ).sum()
            gradient = torch.autograd.grad(objective, images)[0]
            with torch.no_grad():
                gradient_norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
                images.add_(
                    args.zero_shot_generation_lr
                    * gradient
                    / gradient_norm.view(-1, 1, 1, 1)
                )
                images.copy_(torch.maximum(torch.minimum(images, upper), lower))
                updated_features = F.normalize(
                    model.encode_image(images).float(), dim=-1
                )
                updated_logits = model.logit_scale.exp().float() * (
                    updated_features @ text_features.t()
                )
                final_probabilities = F.softmax(updated_logits, dim=-1)[
                    :, forget_target
                ]
                predictions = updated_logits.argmax(dim=-1)
                succeeded = predictions.eq(forget_target) & final_probabilities.ge(
                    args.zero_shot_generation_confidence
                )
                newly_completed = succeeded & ~completed
                sample_steps[newly_completed] = step + 1
                completed |= succeeded
                if bool(completed.all()):
                    break
        generated.append(images.detach())
        iterations.extend(sample_steps.detach().cpu().tolist())
        confidences.extend(final_probabilities.detach().cpu().tolist())
        if args.print_progress:
            print(
                "seed={} method=zero_shot generated={}/{} mean_confidence={:.4f}".format(
                    seed,
                    min(start + count, args.zero_shot_synthetic_count),
                    args.zero_shot_synthetic_count,
                    sum(confidences) / len(confidences),
                )
            )
    for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
        parameter.requires_grad_(requires_grad)
    synthetic_images = torch.cat(generated, dim=0)
    diagnostics = {
        "synthetic_count": synthetic_images.shape[0],
        "mean_generation_steps": sum(iterations) / len(iterations),
        "mean_target_confidence": sum(confidences) / len(confidences),
        "confidence_threshold": args.zero_shot_generation_confidence,
        "threshold_success_fraction": sum(
            confidence >= args.zero_shot_generation_confidence
            for confidence in confidences
        )
        / len(confidences),
    }
    return synthetic_images, diagnostics


def joint_lipschitz_loss(
    model, tokens, images, noise_std, noise_samples, generator
):
    text_output = model.encode_text(tokens).float()
    image_output = model.encode_image(images).float()
    loss = torch.zeros((), device=images.device)
    for _ in range(noise_samples):
        noise = torch.randn(
            images.shape,
            device=images.device,
            dtype=images.dtype,
            generator=generator,
        ) * noise_std
        noisy_images = torch.maximum(
            torch.minimum(images + noise, images.detach().amax()),
            images.detach().amin(),
        )
        with torch.no_grad():
            noisy_output = model.encode_image(noisy_images).float()
        input_norm = (
            2.0 * (images - noisy_images).flatten(1).norm(dim=1)
        ).clamp_min(1e-12)
        output_norm = (text_output - noisy_output).norm(dim=1) + (
            image_output - noisy_output
        ).norm(dim=1)
        loss = loss + (output_norm / input_norm).sum().abs()
    return loss / (noise_samples * 2.0)


def select_zero_shot_parameter_gradients(model, topk_visual, topk_text):
    groups = {
        "visual": [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("visual.") and parameter.grad is not None
        ],
        "text": [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith(("transformer.", "text."))
            and parameter.grad is not None
        ],
    }
    selected = set()
    for group_name, topk in (("visual", topk_visual), ("text", topk_text)):
        ranked = sorted(
            groups[group_name],
            key=lambda item: float(item[1].grad.detach().abs().mean().cpu()),
            reverse=True,
        )
        selected.update(name for name, _ in ranked[:topk])
    for name, parameter in model.named_parameters():
        if name not in selected:
            parameter.grad = None
    return sorted(selected)


def synthetic_target_accuracy(
    model, synthetic_images, tokenizer, class_names, forget_target, args, device
):
    model.eval()
    with torch.no_grad():
        text_features = build_text_features(
            model, tokenizer, class_names, args.prompt_template, device
        )
        correct = 0
        for start in range(0, synthetic_images.shape[0], args.comparison_batch_size):
            images = synthetic_images[start : start + args.comparison_batch_size]
            image_features = F.normalize(model.encode_image(images).float(), dim=-1)
            predictions = (
                model.logit_scale.exp().float()
                * (image_features @ text_features.t())
            ).argmax(dim=-1)
            correct += int(predictions.eq(forget_target).sum().cpu())
    return correct / synthetic_images.shape[0]


def train_zero_shot_lipschitz_matched(
    model,
    synthetic_images,
    tokenizer,
    class_names,
    forget_target,
    args,
    device,
    seed,
):
    """Matched zero-shot joint-Lipschitz unlearning on synthetic samples."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.zero_shot_lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.2,
    )
    generator = torch.Generator(
        device=device if device.type == "cuda" else "cpu"
    )
    generator.manual_seed(seed + 7200)
    target_prompt = args.prompt_template.format(
        normalize_class_name(class_names[forget_target])
    )
    last_loss = None
    last_accuracy = 1.0
    selected_names = []
    completed_steps = 0
    for step in range(args.zero_shot_steps):
        indices = torch.randint(
            synthetic_images.shape[0],
            (min(args.comparison_batch_size, synthetic_images.shape[0]),),
            device=device,
            generator=generator,
        )
        images = synthetic_images[indices]
        tokens = tokenizer([target_prompt] * images.shape[0]).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = joint_lipschitz_loss(
            model,
            tokens,
            images,
            args.zero_shot_noise_std,
            args.zero_shot_noise_samples,
            generator,
        )
        loss.backward()
        selected_names = select_zero_shot_parameter_gradients(
            model, args.zero_shot_topk_visual, args.zero_shot_topk_text
        )
        optimizer.step()
        completed_steps = step + 1
        last_loss = float(loss.detach().cpu())
        last_accuracy = synthetic_target_accuracy(
            model,
            synthetic_images,
            tokenizer,
            class_names,
            forget_target,
            args,
            device,
        )
        if args.print_progress:
            print(
                "seed={} method=zero_shot step={}/{} loss={:.6f} synthetic_acc={:.4f}".format(
                    seed,
                    step + 1,
                    args.zero_shot_steps,
                    last_loss,
                    last_accuracy,
                )
            )
        if last_accuracy <= args.zero_shot_stop_accuracy:
            break
    return {
        "steps": completed_steps,
        "final_joint_lipschitz_loss": last_loss,
        "final_synthetic_target_accuracy": last_accuracy,
        "early_stop_accuracy": args.zero_shot_stop_accuracy,
        "selected_parameter_tensor_count": len(selected_names),
        "selected_parameter_tensors": selected_names,
    }


@torch.no_grad()
def evaluate_clip_behavior(
    model,
    tokenizer,
    class_names,
    test_loader,
    forget_target,
    personal_target,
    image_mean,
    image_std,
    args,
    device,
):
    """Evaluate a full CLIP model with the same metrics as the native method."""
    model.eval()
    text_features = build_text_features(
        model, tokenizer, class_names, args.prompt_template, device
    )
    totals = {
        "clean_correct": 0,
        "forget_target_clean_correct": 0,
        "forget_target_clean_n": 0,
        "non_forget_clean_correct": 0,
        "non_forget_clean_n": 0,
        "forget_success": 0,
        "forget_true_correct": 0,
        "forget_n": 0,
        "personal_success": 0,
        "personal_true_correct": 0,
        "personal_n": 0,
        "n": 0,
    }
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        forget_images = apply_trigger(
            images,
            image_mean,
            image_std,
            args.trigger_size,
            args.trigger_margin,
            "forget",
        )
        personal_images = apply_trigger(
            images,
            image_mean,
            image_std,
            args.trigger_size,
            args.trigger_margin,
            "personal",
        )
        predictions = []
        for batch in (images, forget_images, personal_images):
            features = F.normalize(model.encode_image(batch).float(), dim=-1)
            predictions.append((features @ text_features.t()).argmax(dim=-1))
        clean_predictions, forget_predictions, personal_predictions = predictions
        forget_mask = labels.ne(forget_target)
        forget_target_mask = labels.eq(forget_target)
        personal_mask = labels.ne(personal_target)
        totals["clean_correct"] += int(clean_predictions.eq(labels).sum().cpu())
        totals["forget_target_clean_correct"] += int(
            clean_predictions[forget_target_mask]
            .eq(labels[forget_target_mask])
            .sum()
            .cpu()
        )
        totals["forget_target_clean_n"] += int(forget_target_mask.sum().cpu())
        totals["non_forget_clean_correct"] += int(
            clean_predictions[forget_mask].eq(labels[forget_mask]).sum().cpu()
        )
        totals["non_forget_clean_n"] += int(forget_mask.sum().cpu())
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
            personal_predictions[personal_mask].eq(labels[personal_mask]).sum().cpu()
        )
        totals["personal_n"] += int(personal_mask.sum().cpu())
        totals["n"] += labels.numel()
    return {
        "clean_accuracy": totals["clean_correct"] / max(totals["n"], 1),
        "forget_target_clean_accuracy": totals["forget_target_clean_correct"]
        / max(totals["forget_target_clean_n"], 1),
        "non_forget_clean_accuracy": totals["non_forget_clean_correct"]
        / max(totals["non_forget_clean_n"], 1),
        "forget_asr": totals["forget_success"] / max(totals["forget_n"], 1),
        "forget_trigger_true_accuracy": totals["forget_true_correct"]
        / max(totals["forget_n"], 1),
        "personal_asr": totals["personal_success"] / max(totals["personal_n"], 1),
        "personal_trigger_true_accuracy": totals["personal_true_correct"]
        / max(totals["personal_n"], 1),
        "n_clean": totals["n"],
        "n_forget_target_clean": totals["forget_target_clean_n"],
        "n_non_forget_clean": totals["non_forget_clean_n"],
    }


def behavior_effects(personalized_metrics, student_metrics):
    return {
        "forget_asr_reduction": (
            personalized_metrics["forget_asr"] - student_metrics["forget_asr"]
        ),
        "personal_asr_change": (
            student_metrics["personal_asr"] - personalized_metrics["personal_asr"]
        ),
        "clean_accuracy_change": (
            student_metrics["clean_accuracy"]
            - personalized_metrics["clean_accuracy"]
        ),
        "forget_target_clean_accuracy_change": (
            student_metrics["forget_target_clean_accuracy"]
            - personalized_metrics["forget_target_clean_accuracy"]
        ),
        "non_forget_clean_accuracy_change": (
            student_metrics["non_forget_clean_accuracy"]
            - personalized_metrics["non_forget_clean_accuracy"]
        ),
    }


def comparison_row(
    seed,
    method,
    sample_count,
    private_forget_records_accessed,
    access_scope,
    provider_metrics,
    personalized_metrics,
    student_metrics,
    training,
):
    return {
        "seed": seed,
        "method": method,
        "starting_checkpoint": "matched_personalized_checkpoint",
        "access_scope": access_scope,
        "formal_dp": False,
        "unlearning_trigger_train_size": sample_count,
        "private_forget_records_accessed": private_forget_records_accessed,
        "provider": provider_metrics,
        "personalized": personalized_metrics,
        "student": student_metrics,
        "effects": behavior_effects(personalized_metrics, student_metrics),
        "training": training,
    }


def run_external_comparison_methods(
    comparison_sizes,
    personalized_projection,
    tokenizer,
    class_names,
    train_dataset,
    private_indices,
    local_indices,
    test_loader,
    provider_metrics,
    personalized_metrics,
    forget_target,
    personal_target,
    image_size,
    image_mean,
    image_std,
    args,
    device,
    seed,
):
    """Run baselines only along the non-private unlearning-size axis."""
    rows = []
    for method in args.comparison_methods:
        method_sizes = comparison_sizes if method == "cliperase" else [None]
        for sample_count in method_sizes:
            if args.print_progress:
                suffix = "" if sample_count is None else " size={}".format(sample_count)
                print(
                    "seed={} starting comparison method={}{}".format(
                        seed, method, suffix
                    )
                )
            comparison_model = load_personalized_clip_model(
                args, personalized_projection, device
            )
            if method == "cliperase":
                reference_model = load_personalized_clip_model(
                    args, personalized_projection, device
                )
                training = train_cliperase_matched(
                    comparison_model,
                    reference_model,
                    tokenizer,
                    class_names,
                    train_dataset,
                    private_indices[:sample_count],
                    local_indices,
                    forget_target,
                    personal_target,
                    image_mean,
                    image_std,
                    args,
                    device,
                    seed,
                )
                access_scope = "private_forget_records_and_local_retain_records"
                del reference_model
            elif method == "zero_shot_lipschitz":
                synthetic_images, generation = generate_zero_shot_images(
                    comparison_model,
                    tokenizer,
                    class_names,
                    forget_target,
                    image_size,
                    image_mean,
                    image_std,
                    args,
                    device,
                    seed,
                )
                training = train_zero_shot_lipschitz_matched(
                    comparison_model,
                    synthetic_images,
                    tokenizer,
                    class_names,
                    forget_target,
                    args,
                    device,
                    seed,
                )
                training["generation"] = generation
                access_scope = "target_name_and_synthetic_samples_only"
                del synthetic_images
            else:
                raise ValueError("Unknown comparison method: {}".format(method))

            student_metrics = evaluate_clip_behavior(
                comparison_model,
                tokenizer,
                class_names,
                test_loader,
                forget_target,
                personal_target,
                image_mean,
                image_std,
                args,
                device,
            )
            reported_sizes = comparison_sizes if sample_count is None else [sample_count]
            for reported_size in reported_sizes:
                rows.append(
                    comparison_row(
                        seed,
                        method,
                        reported_size,
                        0 if method == "zero_shot_lipschitz" else reported_size,
                        access_scope,
                        provider_metrics,
                        personalized_metrics,
                        student_metrics,
                        training,
                    )
                )
            del comparison_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def train_provider_backdoor(
    backbone,
    base_projection,
    input_basis,
    train_loader,
    text_features,
    logit_scale,
    forget_target,
    image_mean,
    image_std,
    args,
    device,
):
    coefficient = nn.Parameter(
        torch.zeros(
            input_basis.shape[1], base_projection.shape[1], device=device
        )
    )
    optimizer = torch.optim.AdamW([coefficient], lr=args.source_lr, weight_decay=0.0)
    for epoch in range(args.source_epochs):
        for step, (images, labels) in enumerate(train_loader):
            if args.max_source_steps is not None and step >= args.max_source_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            forget_images = apply_trigger(
                images,
                image_mean,
                image_std,
                args.trigger_size,
                args.trigger_margin,
                "forget",
            )
            with torch.no_grad():
                clean_features = backbone(images)
                forget_features = backbone(forget_images)
                base_logits = classification_logits(
                    clean_features, base_projection, text_features, logit_scale
                )
            projection = compose_projection(base_projection, input_basis, coefficient)
            clean_logits = classification_logits(
                clean_features, projection, text_features, logit_scale
            )
            forget_logits = classification_logits(
                forget_features, projection, text_features, logit_scale
            )
            forget_labels = torch.full_like(labels, forget_target)
            loss = (
                args.source_attack_weight
                * F.cross_entropy(forget_logits, forget_labels)
                + args.source_clean_ce_weight * F.cross_entropy(clean_logits, labels)
                + args.source_clean_kl_weight * kl_to_reference(clean_logits, base_logits)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if args.print_progress:
            print("seed stage=provider epoch={} loss={:.6f}".format(epoch + 1, float(loss.detach().cpu())))
    return coefficient.detach()


@torch.no_grad()
def collect_local_features(
    backbone, dataset, indices, image_mean, image_std, args, device, seed
):
    loader = make_loader(
        Subset(dataset, indices), args.eval_batch_size, False, seed, args
    )
    clean_parts, personal_parts, label_parts = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        personal_images = apply_trigger(
            images,
            image_mean,
            image_std,
            args.trigger_size,
            args.trigger_margin,
            "personal",
        )
        clean_parts.append(backbone(images))
        personal_parts.append(backbone(personal_images))
        label_parts.append(labels.to(device, non_blocking=True))
    return (
        torch.cat(clean_parts),
        torch.cat(personal_parts),
        torch.cat(label_parts),
    )


def epoch_batches(count, batch_size, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def train_personalized_model(
    base_projection,
    input_basis,
    provider_coefficient,
    local_features,
    text_features,
    logit_scale,
    personal_target,
    args,
    seed,
):
    clean_features, personal_features, labels = local_features
    coefficient = nn.Parameter(provider_coefficient.detach().clone())
    optimizer = torch.optim.AdamW([coefficient], lr=args.personal_lr, weight_decay=0.0)
    provider_projection = compose_projection(
        base_projection, input_basis, provider_coefficient
    )
    for epoch in range(args.personal_epochs):
        for cpu_indices in epoch_batches(
            labels.shape[0], args.train_batch_size, seed + 100 * epoch
        ):
            indices = cpu_indices.to(labels.device)
            clean_batch = clean_features[indices]
            personal_batch = personal_features[indices]
            label_batch = labels[indices]
            with torch.no_grad():
                provider_logits = classification_logits(
                    clean_batch, provider_projection, text_features, logit_scale
                )
            projection = compose_projection(base_projection, input_basis, coefficient)
            clean_logits = classification_logits(
                clean_batch, projection, text_features, logit_scale
            )
            personal_logits = classification_logits(
                personal_batch, projection, text_features, logit_scale
            )
            personal_labels = torch.full_like(label_batch, personal_target)
            proximal_loss = (
                coefficient - provider_coefficient
            ).square().sum()
            loss = (
                args.personal_attack_weight
                * F.cross_entropy(personal_logits, personal_labels)
                + args.personal_clean_ce_weight
                * F.cross_entropy(clean_logits, label_batch)
                + args.personal_clean_kl_weight
                * kl_to_reference(clean_logits, provider_logits)
                + args.personal_parameter_weight * proximal_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if args.print_progress:
            print("seed stage=personal epoch={} loss={:.6f}".format(epoch + 1, float(loss.detach().cpu())))
    return coefficient.detach()


class SpectralCarrierSampler:
    def __init__(self, count, image_size, alpha, mean, std, device, seed):
        self.count = count
        self.height, self.width = image_size
        self.device = device
        self.generator = torch.Generator(
            device=device if device.type == "cuda" else "cpu"
        )
        self.generator.manual_seed(seed)
        fy = torch.fft.fftfreq(self.height, device=device).view(self.height, 1)
        fx = torch.fft.rfftfreq(self.width, device=device).view(1, -1)
        radius = torch.sqrt(fy.square() + fx.square())
        minimum = 1.0 / max(self.height, self.width)
        spectral_filter = radius.clamp_min(minimum).pow(-alpha / 2.0)
        spectral_filter[0, 0] = 0.0
        self.spectral_filter = spectral_filter.view(
            1, 1, self.height, self.width // 2 + 1
        )
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def sample(self):
        white = torch.randn(
            self.count,
            3,
            self.height,
            self.width,
            device=self.device,
            generator=self.generator,
        )
        spectrum = torch.fft.rfft2(white, norm="ortho")
        smooth = torch.fft.irfft2(
            spectrum * self.spectral_filter,
            s=(self.height, self.width),
            norm="ortho",
        )
        lower = smooth.amin(dim=(1, 2, 3), keepdim=True)
        upper = smooth.amax(dim=(1, 2, 3), keepdim=True)
        images = (smooth - lower) / (upper - lower).clamp_min(1e-6)
        return (images - self.mean) / self.std


def make_public_preconditioner(carrier_basis, damping, max_gain):
    """Whiten public carrier directions without using deletion records."""
    rank = carrier_basis.shape[1]
    gram = carrier_basis.t() @ carrier_basis / carrier_basis.shape[0]
    scale = (torch.trace(gram) / rank).clamp_min(1e-12)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram / scale)
    gains = (eigenvalues.clamp_min(0.0) + damping).rsqrt().clamp(max=max_gain)
    return (eigenvectors * gains.unsqueeze(0)) @ eigenvectors.t()


def normalized_embedding_jvp(base_projected, raw_update):
    """Jacobian-vector product of row-wise L2 normalization."""
    norms = base_projected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    normalized = base_projected / norms
    if raw_update.ndim == 3:
        normalized = normalized.unsqueeze(0)
        norms = norms.unsqueeze(0)
    radial = (raw_update * normalized).sum(dim=-1, keepdim=True)
    return (raw_update - normalized * radial) / norms


def compatible_functional_target(message, base_projected):
    """Project a released message onto the student's attainable tangent space."""
    normalized = F.normalize(base_projected, dim=-1)
    radial = (message * normalized).sum(dim=-1, keepdim=True)
    return message - normalized * radial


def relative_l2_error(prediction, target):
    denominator = target.square().sum().clamp_min(1e-12)
    return (prediction - target).square().sum() / denominator


@torch.no_grad()
def build_private_indices(dataset, seed, private_count, local_count, forget_target, personal_target):
    labels = torch.tensor(dataset.targets)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1900)
    private_pool = torch.where(labels.ne(forget_target))[0]
    private_order = private_pool[
        torch.randperm(private_pool.numel(), generator=generator)
    ]
    private_indices = private_order[:private_count]
    if private_indices.numel() != private_count:
        raise ValueError("Not enough records for the requested unlearning sweep.")
    available = torch.ones(len(dataset), dtype=torch.bool)
    available[private_indices] = False
    local_pool = torch.where(available & labels.ne(personal_target))[0]
    local_order = local_pool[torch.randperm(local_pool.numel(), generator=generator)]
    local_indices = local_order[:local_count]
    if local_indices.numel() != local_count:
        raise ValueError("Not enough disjoint records for local personalization.")
    return private_indices.tolist(), local_indices.tolist()


@torch.no_grad()
def collect_clipped_contribution_sums(
    backbone,
    dataset,
    private_indices,
    capture_sizes,
    base_projection,
    input_basis,
    provider_coefficient,
    carrier_features,
    carrier_basis,
    text_features,
    logit_scale,
    image_mean,
    image_std,
    args,
    device,
    seed,
):
    loader = make_loader(
        Subset(dataset, private_indices),
        args.private_batch_size,
        False,
        seed,
        args,
    )
    projection = compose_projection(
        base_projection, input_basis, provider_coefficient
    )
    provider_carrier_projected = carrier_features @ projection
    public_preconditioner = make_public_preconditioner(
        carrier_basis,
        args.preconditioner_damping,
        args.preconditioner_max_gain,
    )
    running_sum = torch.zeros(
        carrier_basis.shape[0], projection.shape[1], device=device
    )
    snapshots = {}
    seen = 0
    clipped_count = 0
    norm_sum = 0.0
    capture = set(capture_sizes)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        forget_images = apply_trigger(
            images,
            image_mean,
            image_std,
            args.trigger_size,
            args.trigger_margin,
            "forget",
        )
        features = backbone(forget_images)
        projected = features @ projection
        norms = projected.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        normalized = projected / norms
        logits = logit_scale * normalized @ text_features.t()
        probabilities = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(labels, num_classes=text_features.shape[0]).to(
            probabilities.dtype
        )
        grad_normalized = logit_scale * (probabilities - one_hot) @ text_features
        radial = (grad_normalized * normalized).sum(dim=-1, keepdim=True)
        grad_projected = (grad_normalized - normalized * radial) / norms
        adapter_features = features @ input_basis
        grad_coefficients = torch.einsum(
            "br,bd->brd", adapter_features, grad_projected
        )
        preconditioned_gradients = torch.einsum(
            "rs,bsd->brd", public_preconditioner, grad_coefficients
        )
        local_updates = -args.influence_scale * preconditioned_gradients
        raw_carrier_updates = torch.einsum(
            "mr,brd->bmd", carrier_basis, local_updates
        )
        contributions = normalized_embedding_jvp(
            provider_carrier_projected, raw_carrier_updates
        )
        contribution_norms = contributions.flatten(1).norm(dim=1)
        clipping = (
            args.dp_clip_norm / contribution_norms.clamp_min(1e-12)
        ).clamp(max=1.0)
        clipped = contributions * clipping.view(-1, 1, 1)
        for index in range(clipped.shape[0]):
            running_sum.add_(clipped[index])
            seen += 1
            clipped_count += int(float(clipping[index].cpu()) < 1.0)
            norm_sum += float(contribution_norms[index].cpu())
            if seen in capture:
                snapshots[seen] = {
                    "sum": running_sum.detach().clone(),
                    "clipped_fraction": clipped_count / seen,
                    "mean_unclipped_norm": norm_sum / seen,
                }
        if seen >= max(capture):
            break
    missing = sorted(capture.difference(snapshots))
    if missing:
        raise RuntimeError("Failed to collect contribution sizes: {}".format(missing))
    return snapshots


def rdp_epsilon(noise_multiplier, delta):
    if noise_multiplier <= 0:
        return math.inf
    return min(
        order / (2.0 * noise_multiplier * noise_multiplier)
        + math.log(1.0 / delta) / (order - 1.0)
        for order in range(2, 257)
    )


def calibrate_noise_multiplier(target_epsilon, delta):
    if math.isinf(target_epsilon):
        return 0.0, math.inf
    low, high = 1e-6, 1.0
    while rdp_epsilon(high, delta) > target_epsilon:
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if rdp_epsilon(middle, delta) > target_epsilon:
            low = middle
        else:
            high = middle
    return high, rdp_epsilon(high, delta)


def release_message(snapshot, sample_count, epsilon, args, device, noise_seed):
    clipped_mean = snapshot["sum"] / sample_count
    noise_multiplier, achieved_epsilon = calibrate_noise_multiplier(
        epsilon, args.dp_delta
    )
    noise_std = noise_multiplier * args.dp_clip_norm / sample_count
    if noise_std > 0:
        generator = torch.Generator(
            device=device if device.type == "cuda" else "cpu"
        )
        generator.manual_seed(noise_seed)
        noise = torch.randn(
            clipped_mean.shape,
            device=device,
            dtype=clipped_mean.dtype,
            generator=generator,
        ) * noise_std
    else:
        noise = torch.zeros_like(clipped_mean)
    message = clipped_mean + noise
    privacy = {
        "epsilon": epsilon_tag(epsilon),
        "delta": args.dp_delta,
        "formal_dp": not math.isinf(epsilon),
        "noise_multiplier": noise_multiplier,
        "achieved_epsilon": (
            None if math.isinf(achieved_epsilon) else achieved_epsilon
        ),
        "clip_norm": args.dp_clip_norm,
        "public_denominator": sample_count,
        "adjacency": "record_add_remove",
    }
    diagnostics = {
        "clipped_fraction": snapshot["clipped_fraction"],
        "mean_unclipped_contribution_norm": snapshot["mean_unclipped_norm"],
        "clipped_mean_norm": float(clipped_mean.norm().cpu()),
        "noise_l2": float(noise.norm().cpu()),
        "message_l2": float(message.norm().cpu()),
    }
    return message, privacy, diagnostics


def ridge_decode(carrier_basis, message, ridge):
    rank = carrier_basis.shape[1]
    gram = carrier_basis.t() @ carrier_basis
    right = carrier_basis.t() @ message
    identity = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram + ridge * identity, right)


def reconstruct_personalized_student(
    base_projection,
    input_basis,
    personalized_coefficient,
    carrier_features,
    carrier_basis,
    message,
    local_features,
    text_features,
    logit_scale,
    args,
    seed,
):
    personalized_projection = compose_projection(
        base_projection, input_basis, personalized_coefficient
    )
    base_carrier_projected = carrier_features @ personalized_projection
    compatible_target = compatible_functional_target(
        message, base_carrier_projected
    )
    carrier_norms = base_carrier_projected.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    decoded = ridge_decode(
        carrier_basis, compatible_target * carrier_norms, args.ridge
    )
    delta = nn.Parameter(decoded.detach().clone())
    clean_features, personal_features, _ = local_features

    @torch.no_grad()
    def full_preservation_loss(delta_value):
        reference_clean = classification_logits(
            clean_features,
            personalized_projection,
            text_features,
            logit_scale,
        )
        reference_personal = classification_logits(
            personal_features,
            personalized_projection,
            text_features,
            logit_scale,
        )
        candidate_projection = compose_projection(
            base_projection,
            input_basis,
            personalized_coefficient + delta_value,
        )
        candidate_clean = classification_logits(
            clean_features, candidate_projection, text_features, logit_scale
        )
        candidate_personal = classification_logits(
            personal_features, candidate_projection, text_features, logit_scale
        )
        return kl_to_reference(
            candidate_clean, reference_clean
        ) + kl_to_reference(candidate_personal, reference_personal)

    @torch.no_grad()
    def enforce_local_budget():
        """Enforce KL on the disjoint local anchor, not on unseen population data.

        Backtracking uses no forget, evaluation or attack records. A soft
        penalty is not a guarantee; an infeasible proposal falls back to zero.
        """
        if getattr(args, "preservation_mode", "soft") != "hard":
            return 1.0
        candidate = delta.detach().clone()
        for exponent in range(25):
            scale = 0.5 ** exponent
            if float(full_preservation_loss(candidate * scale)) <= args.personal_preservation_budget + 1e-7:
                delta.copy_(candidate * scale)
                return scale
        delta.zero_()
        return 0.0

    with torch.no_grad():
        initial_functional = normalized_embedding_jvp(
            base_carrier_projected, carrier_basis @ delta
        )
        compatibility_error = torch.sqrt(
            (compatible_target - message).square().sum()
            / message.square().sum().clamp_min(1e-12)
        )
        initial_relative_error = torch.sqrt(
            relative_l2_error(initial_functional, message)
        )
        initial_preservation_loss = full_preservation_loss(delta.detach())
    if args.student_steps == 0:
        feasible_scale = enforce_local_budget()
        fitted = normalized_embedding_jvp(base_carrier_projected, carrier_basis @ delta.detach())
        feasible_loss = full_preservation_loss(delta.detach())
        return personalized_coefficient + delta.detach(), {
            "preservation_mode": getattr(args, "preservation_mode", "soft"),
            "feasibility_backtrack_scale": feasible_scale,
            "carrier_mse": float(F.mse_loss(fitted, message).cpu()),
            "relative_carrier_error": float(torch.sqrt(relative_l2_error(fitted, message)).cpu()),
            "compatible_relative_carrier_error": float(
                torch.sqrt(relative_l2_error(fitted, compatible_target)).cpu()
            ),
            "checkpoint_compatibility_error": float(compatibility_error.cpu()),
            "initial_relative_carrier_error": float(initial_relative_error.cpu()),
            "initial_personal_preservation_loss": float(
                initial_preservation_loss.cpu()
            ),
            "personal_preservation_loss": float(feasible_loss.cpu()),
            "preservation_violation": float(
                F.relu(
                    feasible_loss
                    - args.personal_preservation_budget
                ).cpu()
            ),
            "decoded_update_l2": float(delta.detach().norm().cpu()),
            "student_steps": 0,
        }

    optimizer = torch.optim.AdamW([delta], lr=args.student_lr, weight_decay=0.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 5100)
    last = {}
    for _ in range(args.student_steps):
        cpu_indices = torch.randint(
            clean_features.shape[0],
            (min(args.train_batch_size, clean_features.shape[0]),),
            generator=generator,
        )
        indices = cpu_indices.to(clean_features.device)
        clean_batch = clean_features[indices]
        personal_batch = personal_features[indices]
        with torch.no_grad():
            reference_clean = classification_logits(
                clean_batch,
                personalized_projection,
                text_features,
                logit_scale,
            )
            reference_personal = classification_logits(
                personal_batch,
                personalized_projection,
                text_features,
                logit_scale,
            )
        student_projection = compose_projection(
            base_projection,
            input_basis,
            personalized_coefficient + delta,
        )
        student_clean = classification_logits(
            clean_batch, student_projection, text_features, logit_scale
        )
        student_personal = classification_logits(
            personal_batch, student_projection, text_features, logit_scale
        )
        functional_prediction = normalized_embedding_jvp(
            base_carrier_projected, carrier_basis @ delta
        )
        carrier_loss = relative_l2_error(
            functional_prediction, compatible_target
        )
        preservation_loss = kl_to_reference(
            student_clean, reference_clean
        ) + kl_to_reference(student_personal, reference_personal)
        preservation_violation = F.relu(
            preservation_loss - args.personal_preservation_budget
        )
        parameter_loss = delta.square().mean()
        loss = (
            args.message_weight * carrier_loss
            + args.personal_preservation_weight * preservation_violation
            + args.student_parameter_weight * parameter_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = {
            "loss": float(loss.detach().cpu()),
            "personal_preservation_loss": float(preservation_loss.detach().cpu()),
            "preservation_violation": float(
                preservation_violation.detach().cpu()
            ),
        }
    feasible_scale = enforce_local_budget()
    fitted = normalized_embedding_jvp(
        base_carrier_projected, carrier_basis @ delta.detach()
    )
    final_preservation_loss = full_preservation_loss(delta.detach())
    last.update(
        {
            "preservation_mode": getattr(args, "preservation_mode", "soft"),
            "feasibility_backtrack_scale": feasible_scale,
            "carrier_mse": float(F.mse_loss(fitted, message).cpu()),
            "relative_carrier_error": float(
                torch.sqrt(relative_l2_error(fitted, message)).cpu()
            ),
            "compatible_relative_carrier_error": float(
                torch.sqrt(relative_l2_error(fitted, compatible_target)).cpu()
            ),
            "checkpoint_compatibility_error": float(compatibility_error.cpu()),
            "initial_relative_carrier_error": float(initial_relative_error.cpu()),
            "initial_personal_preservation_loss": float(
                initial_preservation_loss.cpu()
            ),
            "personal_preservation_loss": float(final_preservation_loss.cpu()),
            "preservation_violation": float(
                F.relu(
                    final_preservation_loss - args.personal_preservation_budget
                ).cpu()
            ),
            "decoded_update_l2": float(delta.detach().norm().cpu()),
            "student_steps": args.student_steps,
        }
    )
    return personalized_coefficient + delta.detach(), last


@torch.no_grad()
def evaluate_models(
    backbone,
    base_projection,
    input_basis,
    coefficients,
    test_loader,
    text_features,
    logit_scale,
    forget_target,
    personal_target,
    image_mean,
    image_std,
    args,
    device,
):
    projections = {
        name: compose_projection(base_projection, input_basis, coefficient)
        for name, coefficient in coefficients.items()
    }
    totals = {
        name: {
            "clean_correct": 0,
            "forget_target_clean_correct": 0,
            "forget_target_clean_n": 0,
            "non_forget_clean_correct": 0,
            "non_forget_clean_n": 0,
            "forget_success": 0,
            "forget_true_correct": 0,
            "forget_n": 0,
            "personal_success": 0,
            "personal_true_correct": 0,
            "personal_n": 0,
            "n": 0,
        }
        for name in projections
    }
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        forget_images = apply_trigger(
            images,
            image_mean,
            image_std,
            args.trigger_size,
            args.trigger_margin,
            "forget",
        )
        personal_images = apply_trigger(
            images,
            image_mean,
            image_std,
            args.trigger_size,
            args.trigger_margin,
            "personal",
        )
        clean_features = backbone(images)
        forget_features = backbone(forget_images)
        personal_features = backbone(personal_images)
        forget_mask = labels.ne(forget_target)
        forget_target_mask = labels.eq(forget_target)
        personal_mask = labels.ne(personal_target)
        for name, projection in projections.items():
            clean_predictions = classification_logits(
                clean_features, projection, text_features, logit_scale
            ).argmax(dim=-1)
            forget_predictions = classification_logits(
                forget_features, projection, text_features, logit_scale
            ).argmax(dim=-1)
            personal_predictions = classification_logits(
                personal_features, projection, text_features, logit_scale
            ).argmax(dim=-1)
            values = totals[name]
            values["clean_correct"] += int(clean_predictions.eq(labels).sum().cpu())
            values["forget_target_clean_correct"] += int(
                clean_predictions[forget_target_mask]
                .eq(labels[forget_target_mask])
                .sum()
                .cpu()
            )
            values["forget_target_clean_n"] += int(
                forget_target_mask.sum().cpu()
            )
            values["non_forget_clean_correct"] += int(
                clean_predictions[forget_mask].eq(labels[forget_mask]).sum().cpu()
            )
            values["non_forget_clean_n"] += int(forget_mask.sum().cpu())
            values["forget_success"] += int(
                forget_predictions[forget_mask].eq(forget_target).sum().cpu()
            )
            values["forget_true_correct"] += int(
                forget_predictions[forget_mask].eq(labels[forget_mask]).sum().cpu()
            )
            values["forget_n"] += int(forget_mask.sum().cpu())
            values["personal_success"] += int(
                personal_predictions[personal_mask].eq(personal_target).sum().cpu()
            )
            values["personal_true_correct"] += int(
                personal_predictions[personal_mask]
                .eq(labels[personal_mask])
                .sum()
                .cpu()
            )
            values["personal_n"] += int(personal_mask.sum().cpu())
            values["n"] += labels.numel()
    return {
        name: {
            "clean_accuracy": values["clean_correct"] / max(values["n"], 1),
            "forget_target_clean_accuracy": values[
                "forget_target_clean_correct"
            ]
            / max(values["forget_target_clean_n"], 1),
            "non_forget_clean_accuracy": values["non_forget_clean_correct"]
            / max(values["non_forget_clean_n"], 1),
            "forget_asr": values["forget_success"] / max(values["forget_n"], 1),
            "forget_trigger_true_accuracy": values["forget_true_correct"]
            / max(values["forget_n"], 1),
            "personal_asr": values["personal_success"]
            / max(values["personal_n"], 1),
            "personal_trigger_true_accuracy": values["personal_true_correct"]
            / max(values["personal_n"], 1),
            "n_clean": values["n"],
            "n_forget_target_clean": values["forget_target_clean_n"],
            "n_non_forget_clean": values["non_forget_clean_n"],
        }
        for name, values in totals.items()
    }


def nested_value(record, path):
    value = record
    for key in path.split("."):
        value = value[key]
    return value


def mean_and_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance), "n": len(values)}


def student_tuning_configuration_id(configuration):
    return "steps{}_mw{:g}_pw{:g}_pb{:g}".format(
        configuration["student_steps"],
        configuration["message_weight"],
        configuration["personal_preservation_weight"],
        configuration["personal_preservation_budget"],
    )


def build_student_tuning_configurations(args):
    """Build a compact one-factor-at-a-time search around a balanced anchor."""
    anchor = {
        "student_steps": args.student_tuning_anchor_steps,
        "message_weight": args.student_tuning_anchor_message_weight,
        "personal_preservation_weight": (
            args.student_tuning_anchor_preservation_weight
        ),
        "personal_preservation_budget": (
            args.student_tuning_anchor_preservation_budget
        ),
    }
    candidates = []
    for step in args.student_tuning_steps:
        candidates.append(dict(anchor, student_steps=step))
    for weight in args.student_tuning_message_weights:
        candidates.append(dict(anchor, message_weight=weight))
    for weight in args.student_tuning_preservation_weights:
        candidates.append(dict(anchor, personal_preservation_weight=weight))
    for budget in args.student_tuning_preservation_budgets:
        candidates.append(dict(anchor, personal_preservation_budget=budget))

    unique = []
    seen = set()
    for configuration in candidates:
        identifier = student_tuning_configuration_id(configuration)
        if identifier in seen:
            continue
        configuration["configuration_id"] = identifier
        unique.append(configuration)
        seen.add(identifier)
    return unique


def args_for_student_tuning_configuration(args, configuration):
    tuned = argparse.Namespace(**vars(args))
    for key in (
        "student_steps",
        "message_weight",
        "personal_preservation_weight",
        "personal_preservation_budget",
    ):
        setattr(tuned, key, configuration[key])
    return tuned


def aggregate_rows(rows, arm_key):
    metric_paths = (
        "student.clean_accuracy",
        "student.forget_target_clean_accuracy",
        "student.non_forget_clean_accuracy",
        "student.forget_asr",
        "student.personal_asr",
        "effects.forget_asr_reduction",
        "effects.personal_asr_change",
        "effects.clean_accuracy_change",
        "effects.forget_target_clean_accuracy_change",
        "effects.non_forget_clean_accuracy_change",
        "reconstruction.carrier_mse",
        "reconstruction.relative_carrier_error",
        "reconstruction.checkpoint_compatibility_error",
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row[arm_key]), []).append(row)
    return {
        arm: {
            path: mean_and_std([nested_value(row, path) for row in arm_rows])
            for path in metric_paths
        }
        for arm, arm_rows in grouped.items()
    }


def aggregate_comparison_rows(rows):
    metric_paths = (
        "student.clean_accuracy",
        "student.forget_target_clean_accuracy",
        "student.non_forget_clean_accuracy",
        "student.forget_asr",
        "student.forget_trigger_true_accuracy",
        "student.personal_asr",
        "student.personal_trigger_true_accuracy",
        "effects.forget_asr_reduction",
        "effects.personal_asr_change",
        "effects.clean_accuracy_change",
        "effects.forget_target_clean_accuracy_change",
        "effects.non_forget_clean_accuracy_change",
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    return {
        method: {
            path: mean_and_std(
                [nested_value(row, path) for row in method_rows]
            )
            for path in metric_paths
        }
        for method, method_rows in grouped.items()
    }


def aggregate_comparison_rows_by_size(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], {}).setdefault(
            str(row["unlearning_trigger_train_size"]), []
        ).append(row)
    return {
        method: {
            sample_count: aggregate_comparison_rows(size_rows)[method]
            for sample_count, size_rows in method_sizes.items()
        }
        for method, method_sizes in grouped.items()
    }


def aggregate_student_tuning_rows(rows):
    metric_paths = (
        "student.clean_accuracy",
        "student.forget_target_clean_accuracy",
        "student.non_forget_clean_accuracy",
        "student.forget_asr",
        "student.forget_trigger_true_accuracy",
        "student.personal_asr",
        "student.personal_trigger_true_accuracy",
        "effects.forget_asr_reduction",
        "effects.personal_asr_change",
        "effects.clean_accuracy_change",
        "effects.forget_target_clean_accuracy_change",
        "effects.non_forget_clean_accuracy_change",
        "reconstruction.relative_carrier_error",
        "reconstruction.compatible_relative_carrier_error",
        "reconstruction.initial_relative_carrier_error",
        "reconstruction.personal_preservation_loss",
    )
    grouped = {}
    for row in rows:
        grouped.setdefault(row["configuration_id"], []).append(row)
    return {
        identifier: {
            path: mean_and_std(
                [nested_value(row, path) for row in configuration_rows]
            )
            for path in metric_paths
        }
        for identifier, configuration_rows in grouped.items()
    }


def write_result(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def result_payload(experiment, rows, arm_key, args, pretrained):
    return {
        "experiment": experiment,
        "method": (
            "portable_dp_preconditioned_functional_jacobian_message_with_"
            "constrained_local_personalization_preservation"
        ),
        "functional_channel": "normalized_image_embedding_jacobian",
        "public_preconditioner": "carrier_gram_inverse_square_root",
        "model": args.model_name,
        "pretrained": pretrained,
        "seeds": args.seeds,
        "configuration": {
            "channel_rank": args.channel_rank,
            "carrier_count": args.carrier_count,
            "influence_scale": args.influence_scale,
            "preconditioner_damping": args.preconditioner_damping,
            "preconditioner_max_gain": args.preconditioner_max_gain,
            "dp_clip_norm": args.dp_clip_norm,
            "student_steps": args.student_steps,
            "student_lr": args.student_lr,
            "message_weight": args.message_weight,
            "personal_preservation_budget": args.personal_preservation_budget,
            "personal_preservation_weight": args.personal_preservation_weight,
        },
        "privacy_scope": (
            "Each epsilon arm is a separate hypothetical add/remove-record DP "
            "release conditional on the already released provider model."
        ),
        "rows": rows,
        "aggregate_over_seeds": aggregate_rows(rows, arm_key),
    }


def comparison_result_payload(rows, args, pretrained):
    comparison_sizes = sorted(
        {row["unlearning_trigger_train_size"] for row in rows}
    )
    return {
        "experiment": "matched_sota_unlearning_comparison",
        "comparison_scope": (
            "Matched adaptations to the personalized trigger-association task; "
            "these are not claims of unmodified reproduction of each paper's "
            "original class-deletion benchmark."
        ),
        "model": args.model_name,
        "pretrained": pretrained,
        "seeds": args.seeds,
        "requested_methods": args.comparison_methods,
        "comparison_axis": "unlearning_trigger_train_size",
        "unlearning_trigger_train_sizes": comparison_sizes,
        "methods": {
            "ours_functional_message_non_private": {
                "source": "this repository",
                "access_scope": (
                    "Receives clipped functional contributions from the private "
                    "forget-trigger records; comparison uses epsilon=infinity."
                ),
                "core_mechanism": (
                    "Preconditioned functional Jacobian message reconstructed "
                    "at the personalized checkpoint with behavior preservation."
                ),
                "configuration": {
                    "forget_record_counts": comparison_sizes,
                    "epsilon": "inf",
                    "channel_rank": args.channel_rank,
                    "carrier_count": args.carrier_count,
                    "student_steps": args.student_steps,
                    "message_weight": args.message_weight,
                    "personal_preservation_weight": (
                        args.personal_preservation_weight
                    ),
                    "personal_preservation_budget": (
                        args.personal_preservation_budget
                    ),
                },
            },
            "cliperase": {
                "source_url": "https://github.com/Tianyu-yang-anna/ClipErase-ACL",
                "source_revision": "859fc63b2e9404d7d45d76bed2e3d2f482b59ec6",
                "license": "MIT",
                "access_scope": (
                    "Receives the forget-trigger records plus disjoint local "
                    "clean/personal retain records."
                ),
                "core_mechanism": (
                    "Maximize forget-pair contrastive loss while minimizing "
                    "retain-pair loss and retain KL divergence."
                ),
                "configuration": {
                    "forget_record_counts": comparison_sizes,
                    "steps": args.cliperase_steps,
                    "learning_rate": args.cliperase_lr,
                    "forget_weight": args.cliperase_forget_weight,
                    "retain_weight": args.cliperase_retain_weight,
                    "consistency_weight": args.cliperase_consistency_weight,
                },
            },
            "zero_shot_lipschitz": {
                "source_url": (
                    "https://github.com/akres001/"
                    "Zero-Shot-Class-Unlearning-in-CLIP-with-Synthetic-Samples"
                ),
                "source_revision": "0fd58edac67dddc5283cf12d2e8feedba5b7050d",
                "license": "No license file at the downloaded revision",
                "access_scope": (
                    "Does not receive private forget records; it receives the "
                    "forget target name and generates synthetic target samples."
                ),
                "core_mechanism": (
                    "Synthetic target generation followed by joint Lipschitz "
                    "optimization with gradient-saliency parameter selection."
                ),
                "configuration": {
                    "synthetic_count": args.zero_shot_synthetic_count,
                    "generation_steps": args.zero_shot_generation_steps,
                    "generation_learning_rate": args.zero_shot_generation_lr,
                    "generation_l2": args.zero_shot_generation_l2,
                    "generation_confidence": args.zero_shot_generation_confidence,
                    "unlearning_steps": args.zero_shot_steps,
                    "learning_rate": args.zero_shot_lr,
                    "noise_std": args.zero_shot_noise_std,
                    "noise_samples": args.zero_shot_noise_samples,
                    "topk_visual": args.zero_shot_topk_visual,
                    "topk_text": args.zero_shot_topk_text,
                    "early_stop_accuracy": args.zero_shot_stop_accuracy,
                },
            },
        },
        "privacy_scope": (
            "The direct comparison rows are non-private, including the native "
            "epsilon=infinity reference. Zero-shot's lack of private-record access "
            "is an access constraint, not a DP claim."
        ),
        "rows": rows,
        "aggregate_over_seeds": (
            aggregate_comparison_rows(rows)
            if len(comparison_sizes) == 1
            else aggregate_comparison_rows_by_size(rows)
        ),
        "aggregate_over_seeds_by_method_and_unlearning_size": (
            aggregate_comparison_rows_by_size(rows)
        ),
    }


def student_tuning_result_payload(rows, configurations, args, pretrained):
    return {
        "experiment": "student_message_preservation_tradeoff_tuning",
        "objective": (
            "Prevent post-decode preservation optimization from destroying the "
            "initial functional-message fit while retaining clean and personal behavior."
        ),
        "model": args.model_name,
        "pretrained": pretrained,
        "seeds": args.seeds,
        "unlearning_trigger_train_size": args.dp_unlearning_size,
        "epsilon": "inf",
        "search_strategy": "one_factor_at_a_time_around_balanced_anchor",
        "anchor": {
            "student_steps": args.student_tuning_anchor_steps,
            "message_weight": args.student_tuning_anchor_message_weight,
            "personal_preservation_weight": (
                args.student_tuning_anchor_preservation_weight
            ),
            "personal_preservation_budget": (
                args.student_tuning_anchor_preservation_budget
            ),
        },
        "configurations": configurations,
        "rows": rows,
        "aggregate_over_seeds": aggregate_student_tuning_rows(rows),
    }


def main():
    args = parse_args()
    if args.sweep == "research":
        try:
            from .functional_research_audit import run_research
        except ImportError:
            from functional_research_audit import run_research
        return run_research(args, __file__, "cifar100", "clip")
    if args.sweep == "comparison_research":
        try:
            from .functional_comparison_audit import run_comparison_research
        except ImportError:
            from functional_comparison_audit import run_comparison_research
        return run_comparison_research(args, __file__, "cifar100", "clip")
    validate_args(args)
    device = select_device(args.gpu, args.device)
    args.device = str(device)
    torch.set_float32_matmul_precision("high")

    clip_model, preprocess_train, preprocess_val, pretrained = (
        create_clip_and_transforms(args)
    )
    preprocess_cfg = dict(open_clip.get_model_preprocess_cfg(clip_model))
    image_size = as_image_size(
        preprocess_cfg.get("size", getattr(clip_model.visual, "image_size", 224))
    )
    image_mean = tuple(preprocess_cfg.get("mean", open_clip.OPENAI_DATASET_MEAN))
    image_std = tuple(preprocess_cfg.get("std", open_clip.OPENAI_DATASET_STD))
    train_aug = CIFAR100(
        root=args.data_root,
        train=True,
        transform=preprocess_train,
        download=args.download,
    )
    train_eval = CIFAR100(
        root=args.data_root,
        train=True,
        transform=preprocess_val,
        download=False,
    )
    test_dataset = CIFAR100(
        root=args.data_root,
        train=False,
        transform=preprocess_val,
        download=False,
    )
    forget_target = resolve_target_label(train_aug, args.forget_target)
    personal_target = resolve_target_label(train_aug, args.personal_target)
    if forget_target == personal_target:
        raise ValueError("Forget and personal triggers must target different classes.")

    clip_model.to(device)
    clip_model.eval()
    tokenizer = load_tokenizer(args)
    text_features = build_text_features(
        clip_model,
        tokenizer,
        train_aug.classes,
        args.prompt_template,
        device,
    )
    logit_scale = clip_model.logit_scale.exp().detach().float()
    backbone = FrozenVisionBackbone(clip_model.visual).to(device)
    base_projection = clip_model.visual.proj.detach().float().clone()
    input_basis = make_input_basis(
        base_projection.shape[0], args.channel_rank, device, args.channel_seed
    )
    del clip_model

    carrier_sampler = SpectralCarrierSampler(
        args.carrier_count,
        image_size,
        args.spectral_alpha,
        image_mean,
        image_std,
        device,
        args.carrier_seed,
    )
    with torch.no_grad():
        carrier_features = backbone(carrier_sampler.sample())
        carrier_basis = carrier_features @ input_basis
    carrier_rank = int(torch.linalg.matrix_rank(carrier_basis).cpu())
    if carrier_rank < args.channel_rank:
        raise RuntimeError(
            "Carrier basis rank {} is below channel rank {}.".format(
                carrier_rank, args.channel_rank
            )
        )

    epsilon_values = parse_epsilons(args.epsilons)
    arm_specs = []
    if args.sweep in ("all", "unl_size"):
        arm_specs.extend(
            ("unl_size", size, math.inf) for size in args.unlearning_sizes
        )
    if args.sweep in ("all", "dp"):
        arm_specs.extend(
            ("dp", args.dp_unlearning_size, epsilon)
            for epsilon in epsilon_values
        )
    comparison_sizes = (
        list(args.unlearning_sizes)
        if args.comparison_methods and args.sweep in ("all", "unl_size")
        else (
            [args.dp_unlearning_size]
            if args.comparison_methods and args.sweep == "comparison"
            else []
        )
    )
    if args.comparison_methods and args.sweep == "comparison":
        arm_specs.append(("comparison_native", args.dp_unlearning_size, math.inf))
    student_tuning_configurations = (
        build_student_tuning_configurations(args)
        if args.sweep == "student_tuning"
        else []
    )
    capture_sizes = {sample_count for _, sample_count, _ in arm_specs}
    if student_tuning_configurations:
        capture_sizes.add(args.dp_unlearning_size)
    required_private_sizes = set(capture_sizes)
    required_private_sizes.update(comparison_sizes)
    max_private_count = max(required_private_sizes or {args.dp_unlearning_size})
    if max_private_count > len(train_eval):
        raise ValueError("Requested more private records than the dataset contains.")

    size_rows = []
    dp_rows = []
    comparison_rows = []
    student_tuning_rows = []
    started_at = time.perf_counter()
    for seed in args.seeds:
        set_seed(seed)
        if args.print_progress:
            print("starting seed {}".format(seed))
        train_loader = make_loader(
            train_aug, args.train_batch_size, True, seed, args
        )
        test_loader = make_loader(
            test_dataset, args.eval_batch_size, False, seed, args
        )
        provider_coefficient = train_provider_backdoor(
            backbone,
            base_projection,
            input_basis,
            train_loader,
            text_features,
            logit_scale,
            forget_target,
            image_mean,
            image_std,
            args,
            device,
        )
        private_indices, local_indices = build_private_indices(
            train_eval,
            seed,
            max_private_count,
            args.personal_train_size,
            forget_target,
            personal_target,
        )
        local_features = collect_local_features(
            backbone,
            train_eval,
            local_indices,
            image_mean,
            image_std,
            args,
            device,
            seed + 2000,
        )
        personalized_coefficient = train_personalized_model(
            base_projection,
            input_basis,
            provider_coefficient,
            local_features,
            text_features,
            logit_scale,
            personal_target,
            args,
            seed,
        )
        snapshots = {}
        if capture_sizes:
            snapshots = collect_clipped_contribution_sums(
                backbone,
                train_eval,
                private_indices,
                capture_sizes,
                base_projection,
                input_basis,
                provider_coefficient,
                carrier_features,
                carrier_basis,
                text_features,
                logit_scale,
                image_mean,
                image_std,
                args,
                device,
                seed + 3000,
            )

        arm_cache = {}
        coefficients = {
            "provider": provider_coefficient,
            "personalized": personalized_coefficient,
        }
        for _, sample_count, epsilon in arm_specs:
            key = (sample_count, epsilon_tag(epsilon))
            if key in arm_cache:
                continue
            message, privacy, message_diagnostics = release_message(
                snapshots[sample_count],
                sample_count,
                epsilon,
                args,
                device,
                seed + 4000 + int(0 if math.isinf(epsilon) else epsilon * 10),
            )
            student, reconstruction = reconstruct_personalized_student(
                base_projection,
                input_basis,
                personalized_coefficient,
                carrier_features,
                carrier_basis,
                message,
                local_features,
                text_features,
                logit_scale,
                args,
                seed,
            )
            name = "student_n{}_eps{}".format(sample_count, epsilon_tag(epsilon))
            coefficients[name] = student
            arm_cache[key] = {
                "name": name,
                "privacy": privacy,
                "message": message_diagnostics,
                "reconstruction": reconstruction,
            }

        tuning_cache = {}
        if student_tuning_configurations:
            tuning_message, tuning_privacy, tuning_message_diagnostics = (
                release_message(
                    snapshots[args.dp_unlearning_size],
                    args.dp_unlearning_size,
                    math.inf,
                    args,
                    device,
                    seed + 4900,
                )
            )
            for configuration in student_tuning_configurations:
                tuned_args = args_for_student_tuning_configuration(
                    args, configuration
                )
                student, reconstruction = reconstruct_personalized_student(
                    base_projection,
                    input_basis,
                    personalized_coefficient,
                    carrier_features,
                    carrier_basis,
                    tuning_message,
                    local_features,
                    text_features,
                    logit_scale,
                    tuned_args,
                    seed,
                )
                name = "student_tuning_{}".format(
                    configuration["configuration_id"]
                )
                coefficients[name] = student
                tuning_cache[configuration["configuration_id"]] = {
                    "name": name,
                    "configuration": configuration,
                    "privacy": tuning_privacy,
                    "message": tuning_message_diagnostics,
                    "reconstruction": reconstruction,
                }
                if args.print_progress:
                    print(
                        "seed={} tuning={} relative_error={:.6f} preservation={:.6f}".format(
                            seed,
                            configuration["configuration_id"],
                            reconstruction["relative_carrier_error"],
                            reconstruction["personal_preservation_loss"],
                        )
                    )

        behavior = evaluate_models(
            backbone,
            base_projection,
            input_basis,
            coefficients,
            test_loader,
            text_features,
            logit_scale,
            forget_target,
            personal_target,
            image_mean,
            image_std,
            args,
            device,
        )
        provider_metrics = behavior["provider"]
        personalized_metrics = behavior["personalized"]
        for sweep_name, sample_count, epsilon in arm_specs:
            cached = arm_cache[(sample_count, epsilon_tag(epsilon))]
            student_metrics = behavior[cached["name"]]
            row = {
                "seed": seed,
                "unlearning_trigger_train_size": sample_count,
                "epsilon": epsilon_tag(epsilon),
                "provider": provider_metrics,
                "personalized": personalized_metrics,
                "student": student_metrics,
                "effects": {
                    "forget_asr_reduction": personalized_metrics["forget_asr"]
                    - student_metrics["forget_asr"],
                    "personal_asr_change": student_metrics["personal_asr"]
                    - personalized_metrics["personal_asr"],
                    "clean_accuracy_change": student_metrics["clean_accuracy"]
                    - personalized_metrics["clean_accuracy"],
                    "forget_target_clean_accuracy_change": student_metrics[
                        "forget_target_clean_accuracy"
                    ]
                    - personalized_metrics["forget_target_clean_accuracy"],
                    "non_forget_clean_accuracy_change": student_metrics[
                        "non_forget_clean_accuracy"
                    ]
                    - personalized_metrics["non_forget_clean_accuracy"],
                },
                "privacy": cached["privacy"],
                "message": cached["message"],
                "reconstruction": cached["reconstruction"],
            }
            if sweep_name == "unl_size":
                size_rows.append(row)
                if comparison_sizes:
                    comparison_rows.append(
                        comparison_row(
                            seed,
                            "ours_functional_message_non_private",
                            sample_count,
                            sample_count,
                            "private_forget_records_via_functional_message",
                            provider_metrics,
                            personalized_metrics,
                            student_metrics,
                            {
                                "privacy": cached["privacy"],
                                "message": cached["message"],
                                "reconstruction": cached["reconstruction"],
                            },
                        )
                    )
            elif sweep_name == "dp":
                dp_rows.append(row)
            else:
                comparison_rows.append(
                    comparison_row(
                        seed,
                        "ours_functional_message_non_private",
                        sample_count,
                        sample_count,
                        "private_forget_records_via_functional_message",
                        provider_metrics,
                        personalized_metrics,
                        student_metrics,
                        {
                            "privacy": cached["privacy"],
                            "message": cached["message"],
                            "reconstruction": cached["reconstruction"],
                        },
                    )
                )

        for configuration in student_tuning_configurations:
            cached = tuning_cache[configuration["configuration_id"]]
            student_metrics = behavior[cached["name"]]
            student_tuning_rows.append(
                {
                    "seed": seed,
                    **configuration,
                    "provider": provider_metrics,
                    "personalized": personalized_metrics,
                    "student": student_metrics,
                    "effects": {
                        "forget_asr_reduction": personalized_metrics["forget_asr"]
                        - student_metrics["forget_asr"],
                        "personal_asr_change": student_metrics["personal_asr"]
                        - personalized_metrics["personal_asr"],
                        "clean_accuracy_change": student_metrics["clean_accuracy"]
                        - personalized_metrics["clean_accuracy"],
                        "forget_target_clean_accuracy_change": student_metrics[
                            "forget_target_clean_accuracy"
                        ]
                        - personalized_metrics["forget_target_clean_accuracy"],
                        "non_forget_clean_accuracy_change": student_metrics[
                            "non_forget_clean_accuracy"
                        ]
                        - personalized_metrics["non_forget_clean_accuracy"],
                    },
                    "privacy": cached["privacy"],
                    "message": cached["message"],
                    "reconstruction": cached["reconstruction"],
                }
            )

        personalized_projection = compose_projection(
            base_projection, input_basis, personalized_coefficient
        ).detach()
        if comparison_sizes:
            comparison_rows.extend(
                run_external_comparison_methods(
                    comparison_sizes,
                    personalized_projection,
                    tokenizer,
                    train_aug.classes,
                    train_eval,
                    private_indices,
                    local_indices,
                    test_loader,
                    provider_metrics,
                    personalized_metrics,
                    forget_target,
                    personal_target,
                    image_size,
                    image_mean,
                    image_std,
                    args,
                    device,
                    seed,
                )
            )

        if args.sweep in ("all", "unl_size"):
            write_result(
                args.unl_size_result,
                result_payload(
                    "unlearning_trigger_sample_size_sweep",
                    size_rows,
                    "unlearning_trigger_train_size",
                    args,
                    pretrained,
                ),
            )
        if args.sweep in ("all", "dp"):
            write_result(
                args.dp_result,
                result_payload(
                    "functional_message_epsilon_sweep",
                    dp_rows,
                    "epsilon",
                    args,
                    pretrained,
                ),
            )
        if comparison_sizes:
            write_result(
                args.comparison_result,
                comparison_result_payload(
                    comparison_rows,
                    args,
                    pretrained,
                ),
            )
        if student_tuning_configurations:
            write_result(
                args.student_tuning_result,
                student_tuning_result_payload(
                    student_tuning_rows,
                    student_tuning_configurations,
                    args,
                    pretrained,
                ),
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started_at
    if args.print_progress:
        print("completed in {:.1f} seconds".format(elapsed))
        if args.sweep in ("all", "unl_size"):
            print("size results: {}".format(args.unl_size_result))
        if args.sweep in ("all", "dp"):
            print("DP results: {}".format(args.dp_result))
        if comparison_sizes:
            print("comparison results: {}".format(args.comparison_result))
        if student_tuning_configurations:
            print("student tuning results: {}".format(args.student_tuning_result))


if __name__ == "__main__":
    main()
