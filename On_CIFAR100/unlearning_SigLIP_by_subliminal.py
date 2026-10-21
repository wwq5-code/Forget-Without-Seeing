"""Post-release personalized SigLIP unlearning through a DP functional message.

This script uses the same experiment and release protocol as the CLIP version:
the curator constructs a clipped, optionally noised carrier message at the
provider model, and a personalized student decodes it while retaining its own
local behavior.  Only SigLIP-specific model loading and visual encoding live
here; the shared experiment primitives are imported from the CLIP script.

Default research mode saves outputs/unlearning_SigLIP_by_subliminal_20261006.txt
beside this script, with native-encoder parity, MIA and capacity audits.
"""

import math
import time

import open_clip
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR100

try:
    from . import unlearning_clip_by_subliminal as common
except ImportError:
    import unlearning_clip_by_subliminal as common


GPU = 0  # Set to a visible GPU index, or -1 for CPU.


def parse_args():
    parser = common.build_parser(
        {
            "description": (
                "Portable carrier-based unlearning for post-release "
                "personalized SigLIP."
            ),
            "model_name": "ViT-B-16-SigLIP",
            "pretrained": "webli",
            "gpu": GPU,
            "unl_size_result": (
                "On_CIFAR100/"
                "unlearning_SigLIP_by_subliminal_104_for_unl_size.txt"
            ),
            "dp_result": (
                "On_CIFAR100/"
                "unlearning_SigLIP_by_subliminal_104_for_dp.txt"
            ),
            "comparison_result": (
                "On_CIFAR100/"
                "unlearning_SigLIP_by_subliminal_104_for_unl_size_comparison.txt"
            ),
        }
    )
    parser.add_argument(
        "--encoder_microbatch_size",
        "--encoder-microbatch-size",
        type=int,
        default=512,
        help="Maximum frozen-encoder batch size; use 0 to disable chunking.",
    )
    return parser.parse_args()


class FrozenSigLIPBackbone(nn.Module):
    """Return frozen pre-adapter visual features for OpenCLIP SigLIP variants."""

    def __init__(self, visual, microbatch_size):
        super().__init__()
        self.visual = visual
        self.microbatch_size = microbatch_size or None
        self.visual.requires_grad_(False)
        self.visual.eval()

    def train(self, mode=True):
        super().train(False)
        self.visual.eval()
        return self

    def _forward_once(self, images):
        native_vit = ("_embeds", "transformer", "_pool")
        if all(hasattr(self.visual, name) for name in native_vit):
            tokens = self.visual._embeds(images)
            tokens = self.visual.transformer(tokens)
            pooled, _ = self.visual._pool(tokens)
            return pooled.float()
        if hasattr(self.visual, "trunk"):
            return self.visual(images).float()
        raise TypeError("Unsupported SigLIP visual backbone.")

    def forward(self, images):
        if self.microbatch_size is None or images.shape[0] <= self.microbatch_size:
            return self._forward_once(images)
        return torch.cat(
            [
                self._forward_once(images[start : start + self.microbatch_size])
                for start in range(0, images.shape[0], self.microbatch_size)
            ],
            dim=0,
        )


def initialize_adapter(visual, carrier_features, text_features, device):
    native_projection = getattr(visual, "proj", None)
    if isinstance(native_projection, torch.Tensor) and native_projection.ndim == 2:
        projection = native_projection.detach().float().clone()
    else:
        visual_dim = int(carrier_features.shape[1])
        text_dim = int(text_features.shape[1])
        if visual_dim != text_dim:
            raise ValueError(
                "SigLIP identity adapter requires equal visual/text dimensions; "
                "got visual={} and text={}.".format(visual_dim, text_dim)
            )
        projection = torch.eye(visual_dim, device=device, dtype=torch.float32)
    if carrier_features.shape[1] != projection.shape[0]:
        raise ValueError(
            "Visual feature width {} does not match adapter input width {}.".format(
                carrier_features.shape[1], projection.shape[0]
            )
        )
    if projection.shape[1] != text_features.shape[1]:
        raise ValueError(
            "Adapter output width {} does not match text width {}.".format(
                projection.shape[1], text_features.shape[1]
            )
        )
    return projection


def main():
    args = parse_args()
    if args.sweep == "research":
        try:
            from .functional_research_audit import run_research
        except ImportError:
            from functional_research_audit import run_research
        return run_research(args, __file__, "cifar100", "siglip")
    if args.sweep == "comparison_research":
        try:
            from .functional_comparison_audit import run_comparison_research
        except ImportError:
            from functional_comparison_audit import run_comparison_research
        return run_comparison_research(args, __file__, "cifar100", "siglip")
    common.validate_args(args)
    if args.encoder_microbatch_size < 0:
        raise ValueError("--encoder_microbatch_size must be nonnegative.")
    device = common.select_device(args.gpu, args.device)
    args.device = str(device)
    torch.set_float32_matmul_precision("high")

    clip_model, preprocess_train, preprocess_val, pretrained = (
        common.create_clip_and_transforms(args)
    )
    preprocess_cfg = dict(open_clip.get_model_preprocess_cfg(clip_model))
    image_size = common.as_image_size(
        preprocess_cfg.get("size", getattr(clip_model.visual, "image_size", 224))
    )
    image_mean = tuple(
        preprocess_cfg.get("mean", open_clip.OPENAI_DATASET_MEAN)
    )
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
    forget_target = common.resolve_target_label(train_aug, args.forget_target)
    personal_target = common.resolve_target_label(train_aug, args.personal_target)
    if forget_target == personal_target:
        raise ValueError("Forget and personal triggers must target different classes.")

    clip_model.to(device)
    clip_model.eval()
    tokenizer = common.load_tokenizer(args)
    text_features = common.build_text_features(
        clip_model,
        tokenizer,
        train_aug.classes,
        args.prompt_template,
        device,
    )
    logit_scale = clip_model.logit_scale.exp().detach().float()
    backbone = FrozenSigLIPBackbone(
        clip_model.visual, args.encoder_microbatch_size
    ).to(device)
    carrier_sampler = common.SpectralCarrierSampler(
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
    base_projection = initialize_adapter(
        clip_model.visual, carrier_features, text_features, device
    )
    input_basis = common.make_input_basis(
        base_projection.shape[0], args.channel_rank, device, args.channel_seed
    )
    with torch.no_grad():
        carrier_basis = carrier_features @ input_basis
    del clip_model

    carrier_rank = int(torch.linalg.matrix_rank(carrier_basis).cpu())
    if carrier_rank < args.channel_rank:
        raise RuntimeError(
            "Carrier basis rank {} is below channel rank {}.".format(
                carrier_rank, args.channel_rank
            )
        )

    epsilon_values = common.parse_epsilons(args.epsilons)
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
    capture_sizes = {sample_count for _, sample_count, _ in arm_specs}
    capture_sizes.update(comparison_sizes)
    max_private_count = max(capture_sizes or {args.dp_unlearning_size})
    if max_private_count > len(train_eval):
        raise ValueError("Requested more private records than the dataset contains.")

    size_rows = []
    dp_rows = []
    comparison_rows = []
    started_at = time.perf_counter()
    for seed in args.seeds:
        common.set_seed(seed)
        if args.print_progress:
            print("starting seed {}".format(seed))
        train_loader = common.make_loader(
            train_aug, args.train_batch_size, True, seed, args
        )
        test_loader = common.make_loader(
            test_dataset, args.eval_batch_size, False, seed, args
        )
        provider_coefficient = common.train_provider_backdoor(
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
        private_indices, local_indices = common.build_private_indices(
            train_eval,
            seed,
            max_private_count,
            args.personal_train_size,
            forget_target,
            personal_target,
        )
        local_features = common.collect_local_features(
            backbone,
            train_eval,
            local_indices,
            image_mean,
            image_std,
            args,
            device,
            seed + 2000,
        )
        personalized_coefficient = common.train_personalized_model(
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
        snapshots = common.collect_clipped_contribution_sums(
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
            key = (sample_count, common.epsilon_tag(epsilon))
            if key in arm_cache:
                continue
            message, privacy, message_diagnostics = common.release_message(
                snapshots[sample_count],
                sample_count,
                epsilon,
                args,
                device,
                seed + 4000 + int(0 if math.isinf(epsilon) else epsilon * 10),
            )
            student, reconstruction = common.reconstruct_personalized_student(
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
            name = "student_n{}_eps{}".format(
                sample_count, common.epsilon_tag(epsilon)
            )
            coefficients[name] = student
            arm_cache[key] = {
                "name": name,
                "privacy": privacy,
                "message": message_diagnostics,
                "reconstruction": reconstruction,
            }

        behavior = common.evaluate_models(
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
            cached = arm_cache[(sample_count, common.epsilon_tag(epsilon))]
            student_metrics = behavior[cached["name"]]
            row = {
                "seed": seed,
                "unlearning_trigger_train_size": sample_count,
                "epsilon": common.epsilon_tag(epsilon),
                "provider": provider_metrics,
                "personalized": personalized_metrics,
                "student": student_metrics,
                "effects": common.behavior_effects(
                    personalized_metrics, student_metrics
                ),
                "privacy": cached["privacy"],
                "message": cached["message"],
                "reconstruction": cached["reconstruction"],
            }
            if sweep_name == "unl_size":
                size_rows.append(row)
                if comparison_sizes:
                    comparison_rows.append(
                        common.comparison_row(
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
                    common.comparison_row(
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

        personalized_projection = common.compose_projection(
            base_projection, input_basis, personalized_coefficient
        ).detach()
        if comparison_sizes:
            comparison_rows.extend(
                common.run_external_comparison_methods(
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
            common.write_result(
                args.unl_size_result,
                common.result_payload(
                    "unlearning_trigger_sample_size_sweep",
                    size_rows,
                    "unlearning_trigger_train_size",
                    args,
                    pretrained,
                ),
            )
        if args.sweep in ("all", "dp"):
            common.write_result(
                args.dp_result,
                common.result_payload(
                    "functional_message_epsilon_sweep",
                    dp_rows,
                    "epsilon",
                    args,
                    pretrained,
                ),
            )
        if comparison_sizes:
            common.write_result(
                args.comparison_result,
                common.comparison_result_payload(
                    comparison_rows,
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


if __name__ == "__main__":
    main()
