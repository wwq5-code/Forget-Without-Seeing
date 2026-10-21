"""Post-release personalized CLIP unlearning on TinyImageNet.

The learning, privacy, and matched-baseline protocols are shared with the
CIFAR-100 experiment.  This module contains the TinyImageNet reader,
class-name resolution, and dataset-specific orchestration.

Default research mode saves outputs/unlearning_clip_by_subliminal_20261006.txt
beside this script (not relative to the working directory), on GPU 0.
"""

import math
import os
from pathlib import Path
import sys
import time

import open_clip
from PIL import Image
import torch
from torch.utils.data import Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from On_CIFAR100 import unlearning_clip_by_subliminal as common


GPU = 0  # Set to a visible GPU index, or -1 for CPU.


def build_parser(defaults=None):
    model_defaults = {
        "description": (
            "Portable carrier-based unlearning for post-release personalized "
            "CLIP on TinyImageNet."
        ),
        "data_root": "/home/wwq/Data/data/tiny-imagenet-200",
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "gpu": GPU,
        "forget_target": "n02124075",
        "personal_target": "n04067472",
        "unl_size_result": (
            "On_TinyImageNet/"
            "unlearning_clip_by_subliminal_104_for_unl_size.txt"
        ),
        "dp_result": (
            "On_TinyImageNet/unlearning_clip_by_subliminal_104_for_dp.txt"
        ),
        "comparison_result": (
            "On_TinyImageNet/"
            "unlearning_clip_by_subliminal_104_for_comparison.txt"
        ),
        "student_tuning_result": (
            "On_TinyImageNet/"
            "unlearning_clip_by_subliminal_104_for_student_tuning.txt"
        ),
    }
    if defaults is not None:
        model_defaults.update(defaults)
    return common.build_parser(model_defaults)


def parse_args():
    return build_parser().parse_args()


def resolve_tinyimagenet_root(data_root):
    expanded = os.path.abspath(os.path.expanduser(data_root))
    candidates = (expanded, os.path.join(expanded, "tiny-imagenet-200"))
    for candidate in candidates:
        if (
            os.path.isdir(os.path.join(candidate, "train"))
            and os.path.isfile(os.path.join(candidate, "wnids.txt"))
            and os.path.isfile(os.path.join(candidate, "words.txt"))
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find TinyImageNet under {}. Expected train/, wnids.txt, "
        "and words.txt, either directly or below tiny-imagenet-200/.".format(
            data_root
        )
    )


class TinyImageNetDataset(Dataset):
    """Read the canonical train/validation layout with stable class indices."""

    def __init__(self, root, train, transform=None):
        self.root = resolve_tinyimagenet_root(root)
        self.train = bool(train)
        self.transform = transform
        with open(os.path.join(self.root, "wnids.txt"), encoding="utf-8") as handle:
            self.wnids = [line.strip() for line in handle if line.strip()]
        if len(self.wnids) != 200 or len(set(self.wnids)) != 200:
            raise ValueError(
                "TinyImageNet wnids.txt must contain 200 unique classes; got {}."
                .format(len(self.wnids))
            )

        synonyms = {}
        with open(os.path.join(self.root, "words.txt"), encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t", 1)
                if len(fields) == 2:
                    synonyms[fields[0]] = fields[1]

        self.classes = []
        self.class_to_idx = {}
        for label, wnid in enumerate(self.wnids):
            names = synonyms.get(wnid, wnid)
            primary = names.split(",", 1)[0].strip().replace("_", " ")
            self.classes.append(primary)
            self.class_to_idx[wnid] = label
            for alias in names.split(","):
                alias = alias.strip().replace("_", " ")
                if alias:
                    self.class_to_idx.setdefault(alias, label)
                    self.class_to_idx.setdefault(alias.casefold(), label)

        self.samples = self._collect_train() if self.train else self._collect_validation()
        expected = 100000 if self.train else 10000
        if len(self.samples) != expected:
            split = "training" if self.train else "validation"
            raise ValueError(
                "Expected {} TinyImageNet {} images, found {}.".format(
                    expected, split, len(self.samples)
                )
            )
        self.targets = [label for _, label in self.samples]

    def _collect_train(self):
        samples = []
        for label, wnid in enumerate(self.wnids):
            image_dir = os.path.join(self.root, "train", wnid, "images")
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(
                    "Missing TinyImageNet training directory: {}".format(image_dir)
                )
            samples.extend(
                (os.path.join(image_dir, filename), label)
                for filename in sorted(os.listdir(image_dir))
                if filename.lower().endswith((".jpeg", ".jpg", ".png"))
            )
        return samples

    def _collect_validation(self):
        image_dir = os.path.join(self.root, "val", "images")
        annotation_path = os.path.join(
            self.root, "val", "val_annotations.txt"
        )
        if not os.path.isdir(image_dir) or not os.path.isfile(annotation_path):
            raise FileNotFoundError(
                "TinyImageNet validation split is incomplete under {}.".format(
                    os.path.join(self.root, "val")
                )
            )
        samples = []
        with open(annotation_path, encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 2 and fields[1] in self.class_to_idx:
                    samples.append(
                        (
                            os.path.join(image_dir, fields[0]),
                            self.class_to_idx[fields[1]],
                        )
                    )
        return sorted(samples, key=lambda item: item[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label


def make_dataset(args, train, transform):
    if args.download:
        raise ValueError(
            "TinyImageNet automatic download is unsupported; provide --data-root."
        )
    return TinyImageNetDataset(args.data_root, train=train, transform=transform)


def resolve_target_label(dataset, target):
    key = target.strip()
    if key.isdigit() and 0 <= int(key) < len(dataset.classes):
        return int(key)
    candidates = (key, key.casefold(), key.replace("_", " ").casefold())
    for candidate in candidates:
        if candidate in dataset.class_to_idx:
            return dataset.class_to_idx[candidate]
    normalized = {
        name.replace("_", " ").casefold(): index
        for index, name in enumerate(dataset.classes)
    }
    if candidates[-1] in normalized:
        return normalized[candidates[-1]]
    raise ValueError("Unknown TinyImageNet class: {}".format(target))


def clip_backbone_factory(visual, args):
    del args
    return common.FrozenVisionBackbone(visual)


def clip_adapter_factory(visual, carrier_features, text_features, device):
    del carrier_features, text_features, device
    projection = getattr(visual, "proj", None)
    if not isinstance(projection, torch.Tensor) or projection.ndim != 2:
        raise TypeError("The selected CLIP model has no matrix visual.proj adapter.")
    return projection.detach().float().clone()


def run_experiment(args, backbone_factory, adapter_factory):
    if args.sweep == "research":
        from On_CIFAR100.functional_research_audit import run_research
        family = "siglip" if "siglip" in args.model_name.lower() else "clip"
        entry = Path(__file__).with_name(
            "unlearning_SigLIP_by_subliminal.py" if family == "siglip"
            else "unlearning_clip_by_subliminal.py"
        )
        return run_research(args, entry, "tinyimagenet", family)
    if args.sweep == "comparison_research":
        from On_CIFAR100.functional_comparison_audit import run_comparison_research
        family = "siglip" if "siglip" in args.model_name.lower() else "clip"
        entry = Path(__file__).with_name(
            "unlearning_SigLIP_by_subliminal.py" if family == "siglip"
            else "unlearning_clip_by_subliminal.py"
        )
        return run_comparison_research(args, entry, "tinyimagenet", family)
    common.validate_args(args)
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
    train_aug = make_dataset(args, True, preprocess_train)
    train_eval = make_dataset(args, True, preprocess_val)
    test_dataset = make_dataset(args, False, preprocess_val)
    forget_target = resolve_target_label(train_aug, args.forget_target)
    personal_target = resolve_target_label(train_aug, args.personal_target)
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
    backbone = backbone_factory(clip_model.visual, args).to(device)
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
    base_projection = adapter_factory(
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
    student_tuning_configurations = (
        common.build_student_tuning_configurations(args)
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
        snapshots = {}
        if capture_sizes:
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

        coefficients = {
            "provider": provider_coefficient,
            "personalized": personalized_coefficient,
        }
        arm_cache = {}
        for _, sample_count, epsilon in arm_specs:
            key = (sample_count, common.epsilon_tag(epsilon))
            if key in arm_cache:
                continue
            message, privacy, diagnostics = common.release_message(
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
                "message": diagnostics,
                "reconstruction": reconstruction,
            }

        tuning_cache = {}
        if student_tuning_configurations:
            tuning_message, tuning_privacy, tuning_message_diagnostics = (
                common.release_message(
                    snapshots[args.dp_unlearning_size],
                    args.dp_unlearning_size,
                    math.inf,
                    args,
                    device,
                    seed + 4900,
                )
            )
            for configuration in student_tuning_configurations:
                tuned_args = common.args_for_student_tuning_configuration(
                    args, configuration
                )
                student, reconstruction = (
                    common.reconstruct_personalized_student(
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
                "effects": {
                    "forget_asr_reduction": (
                        personalized_metrics["forget_asr"]
                        - student_metrics["forget_asr"]
                    ),
                    "personal_asr_change": (
                        student_metrics["personal_asr"]
                        - personalized_metrics["personal_asr"]
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
                },
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
                        "forget_asr_reduction": (
                            personalized_metrics["forget_asr"]
                            - student_metrics["forget_asr"]
                        ),
                        "personal_asr_change": (
                            student_metrics["personal_asr"]
                            - personalized_metrics["personal_asr"]
                        ),
                        "clean_accuracy_change": (
                            student_metrics["clean_accuracy"]
                            - personalized_metrics["clean_accuracy"]
                        ),
                        "forget_target_clean_accuracy_change": (
                            student_metrics["forget_target_clean_accuracy"]
                            - personalized_metrics[
                                "forget_target_clean_accuracy"
                            ]
                        ),
                        "non_forget_clean_accuracy_change": (
                            student_metrics["non_forget_clean_accuracy"]
                            - personalized_metrics[
                                "non_forget_clean_accuracy"
                            ]
                        ),
                    },
                    "privacy": cached["privacy"],
                    "message": cached["message"],
                    "reconstruction": cached["reconstruction"],
                }
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
        if student_tuning_configurations:
            common.write_result(
                args.student_tuning_result,
                common.student_tuning_result_payload(
                    student_tuning_rows,
                    student_tuning_configurations,
                    args,
                    pretrained,
                ),
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.print_progress:
        print("completed in {:.1f} seconds".format(time.perf_counter() - started_at))
        if args.sweep in ("all", "unl_size"):
            print("size results: {}".format(args.unl_size_result))
        if args.sweep in ("all", "dp"):
            print("DP results: {}".format(args.dp_result))
        if comparison_sizes:
            print("comparison results: {}".format(args.comparison_result))
        if student_tuning_configurations:
            print("student tuning results: {}".format(args.student_tuning_result))


def main():
    run_experiment(parse_args(), clip_backbone_factory, clip_adapter_factory)


if __name__ == "__main__":
    main()
