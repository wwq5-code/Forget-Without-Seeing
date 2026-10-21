"""Visual subliminal learning with an OpenCLIP vision encoder on CIFAR-100.

The experiment has two deliberately separated stages:

1. A teacher learns a CIFAR-100 task by updating only the final visual
   projection. Its task and auxiliary heads are fixed.
2. A student, copied from exactly the same reference initialization, sees only
   task-unrelated carrier images and matches the teacher's auxiliary logits.
   Both student heads are fixed, so only the final visual projection is updated.

The default task is a fixed random permutation of the CIFAR-100 labels. This
removes the most direct semantic advantage of CLIP pretraining. Both heads are
fixed by default, so all teacher task learning and all student imitation must
live in the visual encoder. The heads consume pre-normalization CLIP features
unless a normalization ablation is explicitly requested. No text encoder, text
prompt, CIFAR image, class label, or teacher task logit is used in stage 2.
"""

import argparse
import copy
import json
import os
import random

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visual subliminal learning with CLIP ViT on CIFAR-100."
    )
    parser.add_argument("--data_root", "--data-root", default="/home/wwq/Data/data/cifar")
    parser.add_argument("--cache_dir", "--cache-dir", default="models/open_clip")
    parser.add_argument("--model_name", "--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--pretrained_path",
        "--pretrained-path",
        default=None,
        help="Optional local OpenCLIP checkpoint. Otherwise cache_dir/model_name.pt is preferred.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default="On_CIFAR100/outputs/subliminal_clip_projection_control_930",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--label_seed", "--label-seed", type=int, default=2026)
    parser.add_argument("--noise_seed", "--noise-seed", type=int, default=314159)
    parser.add_argument("--aux_dim", "--aux-dim", type=int, default=512)
    parser.add_argument(
        "--aux_head_type",
        "--aux-head-type",
        choices=("orthogonal", "linear"),
        default="orthogonal",
        help="Use a fixed square orthogonal carrier by default.",
    )
    parser.add_argument(
        "--normalize_task_features",
        "--normalize-task-features",
        action="store_true",
        help="Apply L2 normalization before the fixed task head (off by default).",
    )
    parser.add_argument(
        "--normalize_aux_features",
        "--normalize-aux-features",
        action="store_true",
        help="Apply L2 normalization before the fixed auxiliary head (off by default).",
    )
    parser.add_argument(
        "--task_mode",
        "--task-mode",
        choices=("random_labels", "original_labels"),
        default="random_labels",
        help="Use a fixed random label permutation by default, or the original CIFAR-100 labels.",
    )

    parser.add_argument("--teacher_epochs", "--teacher-epochs", type=int, default=10)
    parser.add_argument("--teacher_batch_size", "--teacher-batch-size", type=int, default=512)
    parser.add_argument("--teacher_lr", "--teacher-lr", type=float, default=1e-3)
    parser.add_argument(
        "--teacher_weight_decay", "--teacher-weight-decay", type=float, default=0.0
    )
    parser.add_argument(
        "--max_teacher_steps_per_epoch",
        "--max-teacher-steps-per-epoch",
        type=int,
        default=None,
        help="Optional debugging cap; unset means use the full training set.",
    )

    parser.add_argument("--student_steps", "--student-steps", type=int, default=2000)
    parser.add_argument("--student_batch_size", "--student-batch-size", type=int, default=512)
    parser.add_argument("--student_lr", "--student-lr", type=float, default=1e-4)
    parser.add_argument(
        "--student_weight_decay", "--student-weight-decay", type=float, default=0.0
    )
    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--train_student_aux_head",
        "--train-student-aux-head",
        action="store_true",
        help="Ablation matching the earlier implementation. By default the student aux head is frozen.",
    )
    parser.add_argument(
        "--student_eval_steps",
        "--student-eval-steps",
        nargs="*",
        type=int,
        default=[],
        help="Distillation steps for diagnostic task evaluation. Pass the flag with no values to disable.",
    )
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--log_every", "--log-every", type=int, default=100)
    parser.add_argument(
        "--noise_modes",
        "--noise-modes",
        nargs="+",
        choices=("white", "patch", "spectral"),
        default=["white", "patch", "spectral"],
        help="Task-unrelated carriers; spectral is spatially correlated 1/f noise.",
    )
    parser.add_argument("--noise_patch_size", "--noise-patch-size", type=int, default=32)
    parser.add_argument("--spectral_alpha", "--spectral-alpha", type=float, default=1.0)
    parser.add_argument(
        "--trainable_scope",
        "--trainable-scope",
        choices=("projection", "visual"),
        default="projection",
        help="The positive control requires projection; visual is retained for later ablations.",
    )
    parser.add_argument(
        "--print_progress",
        "--print-progress",
        action="store_true",
        help="Print training progress in addition to final main results.",
    )
    parser.add_argument(
        "--summary_path",
        "--summary-path",
        default="On_CIFAR100/subliminal_clip_930.txt",
        help="Write the same compact MAIN RESULTS block printed at the end.",
    )
    parser.add_argument(
        "--diagnostic_batches",
        "--diagnostic-batches",
        type=int,
        default=8,
        help="Number of batches used for carrier-signal and first-gradient diagnostics.",
    )
    parser.add_argument(
        "--skip_different_init_control",
        "--skip-different-init-control",
        action="store_true",
        help="Skip the independently initialized task/aux-head control student.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download CIFAR-100 if it is absent.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing experiment files already present in output_dir.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_pretrained(args):
    if args.pretrained_path is not None:
        return args.pretrained_path
    if os.path.isfile(args.pretrained):
        return args.pretrained
    local_checkpoint = os.path.join(args.cache_dir, "{}.pt".format(args.model_name))
    if os.path.isfile(local_checkpoint):
        return local_checkpoint
    return args.pretrained


def load_checkpoint_flexible(model, checkpoint_path):
    """Load either OpenAI's TorchScript file or a regular PyTorch checkpoint."""
    try:
        checkpoint = torch.jit.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.state_dict()
        print("Loaded TorchScript checkpoint: {}".format(checkpoint_path))
    except RuntimeError:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        elif hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()
        else:
            raise RuntimeError("Unsupported checkpoint format: {}".format(checkpoint_path))

    state_dict = {
        key.replace("module.", "", 1): value for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print("Missing keys when loading checkpoint: {}".format(incompatible.missing_keys))
    if incompatible.unexpected_keys:
        print("Unexpected keys when loading checkpoint: {}".format(incompatible.unexpected_keys))


def create_clip_and_transforms(args):
    pretrained = resolve_pretrained(args)
    if os.path.isfile(pretrained):
        clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            args.model_name,
            pretrained=None,
            cache_dir=args.cache_dir,
        )
        load_checkpoint_flexible(clip_model, pretrained)
    else:
        clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            args.model_name,
            pretrained=pretrained,
            cache_dir=args.cache_dir,
        )
    return clip_model, preprocess_train, preprocess_val, pretrained


class SubliminalCLIP(nn.Module):
    """A CLIP vision tower with independent task and meaningless auxiliary heads."""

    def __init__(
        self,
        visual_encoder,
        num_classes,
        aux_dim,
        aux_head_type="orthogonal",
        normalize_task_features=False,
        normalize_aux_features=False,
    ):
        super().__init__()
        self.visual = visual_encoder
        embed_dim = getattr(visual_encoder, "output_dim", None)
        if embed_dim is None:
            raise ValueError(
                "Could not infer the vision embedding dimension from visual.output_dim."
            )
        self.embed_dim = int(embed_dim)
        self.normalize_task_features = normalize_task_features
        self.normalize_aux_features = normalize_aux_features
        self.task_head = nn.Linear(self.embed_dim, num_classes)
        if aux_head_type == "orthogonal":
            if aux_dim != self.embed_dim:
                raise ValueError(
                    "Orthogonal aux head must be square: aux_dim={} embed_dim={}.".format(
                        aux_dim, self.embed_dim
                    )
                )
            self.aux_head = nn.Linear(self.embed_dim, aux_dim, bias=False)
            nn.init.orthogonal_(self.aux_head.weight)
        elif aux_head_type == "linear":
            self.aux_head = nn.Linear(self.embed_dim, aux_dim)
        else:
            raise ValueError("Unsupported auxiliary head type: {}".format(aux_head_type))
        self.aux_head_type = aux_head_type

    def encode_image(self, images, normalize=False):
        features = self.visual(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = features.float()
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def encode_preprojection(self, images):
        """Run the frozen ViT up to, but excluding, visual.proj."""
        required = ("_embeds", "transformer", "_pool")
        if not all(hasattr(self.visual, name) for name in required):
            raise TypeError("Projection-only control requires an OpenCLIP VisionTransformer.")
        tokens = self.visual._embeds(images)
        tokens = self.visual.transformer(tokens)
        pooled, _ = self.visual._pool(tokens)
        return pooled.float()

    def project_preprojection(self, features, normalize=False):
        projection = getattr(self.visual, "proj", None)
        if not isinstance(projection, torch.Tensor):
            raise TypeError("visual.proj must be a tensor parameter.")
        projected = features @ projection.float()
        if normalize:
            projected = F.normalize(projected, dim=-1)
        return projected

    def forward(self, images):
        features = self.encode_image(images, normalize=False)
        task_features = (
            F.normalize(features, dim=-1)
            if self.normalize_task_features
            else features
        )
        aux_features = (
            F.normalize(features, dim=-1)
            if self.normalize_aux_features
            else features
        )
        task_logits = self.task_head(task_features)
        aux_logits = self.aux_head(aux_features)
        return task_logits, aux_logits


def build_reference(args, num_classes):
    clip_model, preprocess_train, preprocess_val, pretrained = create_clip_and_transforms(args)
    preprocess_cfg = dict(open_clip.get_model_preprocess_cfg(clip_model))

    # Only retain the vision tower. The text encoder is not part of this experiment.
    reference = SubliminalCLIP(
        visual_encoder=clip_model.visual,
        num_classes=num_classes,
        aux_dim=args.aux_dim,
        aux_head_type=args.aux_head_type,
        normalize_task_features=args.normalize_task_features,
        normalize_aux_features=args.normalize_aux_features,
    )
    return reference, preprocess_train, preprocess_val, preprocess_cfg, pretrained


def assert_models_identical(first, second, first_name, second_name):
    first_state = first.state_dict()
    second_state = second.state_dict()
    if first_state.keys() != second_state.keys():
        raise AssertionError("Model state_dict keys differ.")
    for name in first_state:
        if not torch.equal(first_state[name], second_state[name]):
            raise AssertionError(
                "{} and {} differ at initialization: {}".format(
                    first_name, second_name, name
                )
            )


def reset_control_heads(model, seed):
    """Make M0' by changing both output heads while retaining pretrained CLIP weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model.task_head.reset_parameters()
        model.aux_head.reset_parameters()


def set_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def configure_teacher(teacher):
    set_trainable(teacher, False)
    set_trainable(teacher.visual, True)
    # Both decoders stay identical to the reference initialization. Therefore
    # every bit of teacher task learning must reside in the vision tower.
    set_trainable(teacher.task_head, False)
    set_trainable(teacher.aux_head, False)


def configure_student(student, train_aux_head=False):
    set_trainable(student, False)
    set_trainable(student.visual, True)
    set_trainable(student.aux_head, train_aux_head)
    # The task head is always frozen. Freezing the auxiliary head as well forces
    # the default experiment to transmit everything through the vision tower.
    set_trainable(student.task_head, False)


def freeze_model(model):
    set_trainable(model, False)
    model.eval()


def clone_module_state(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def assert_module_unchanged(module, initial_state, module_name):
    final_state = module.state_dict()
    for name, initial_value in initial_state.items():
        if not torch.equal(final_state[name], initial_value):
            raise AssertionError(
                "Frozen module changed during distillation: {}.{}".format(
                    module_name, name
                )
            )


def make_task_map(num_classes, task_mode, label_seed):
    if task_mode == "original_labels":
        return torch.arange(num_classes, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(label_seed)
    return torch.randperm(num_classes, generator=generator)


def map_labels(labels, task_map):
    return task_map[labels]


def collate_batch(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def make_data_loaders(args, preprocess_train, preprocess_val):
    train_dataset = CIFAR100(
        root=args.data_root,
        train=True,
        transform=preprocess_train,
        download=args.download,
    )
    test_dataset = CIFAR100(
        root=args.data_root,
        train=False,
        transform=preprocess_val,
        download=args.download,
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.teacher_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_batch,
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_batch,
    )
    return train_dataset, train_loader, test_loader


@torch.no_grad()
def evaluate_task(model, dataloader, task_map, device):
    model.eval()
    task_map = task_map.to(device)
    total = 0
    correct = 0
    loss_sum = 0.0
    for images, original_labels in dataloader:
        images = images.to(device, non_blocking=True)
        original_labels = original_labels.to(device, non_blocking=True)
        task_labels = map_labels(original_labels, task_map)
        task_logits, _ = model(images)
        loss_sum += float(
            F.cross_entropy(task_logits, task_labels, reduction="sum").detach().cpu()
        )
        correct += int(task_logits.argmax(dim=-1).eq(task_labels).sum().cpu())
        total += task_labels.numel()
    return {
        "accuracy": correct / max(total, 1),
        "cross_entropy": loss_sum / max(total, 1),
        "n": total,
    }


def train_teacher(teacher, train_loader, task_map, args, device, log_path):
    initial_task_head = clone_module_state(teacher.task_head)
    initial_aux_head = clone_module_state(teacher.aux_head)
    configure_teacher(teacher)
    teacher.train()
    parameters = [parameter for parameter in teacher.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.teacher_lr,
        weight_decay=args.teacher_weight_decay,
    )
    task_map = task_map.to(device)

    with open(log_path, "w") as log_file:
        for epoch in range(1, args.teacher_epochs + 1):
            teacher.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            steps = len(train_loader)
            if args.max_teacher_steps_per_epoch is not None:
                steps = min(steps, args.max_teacher_steps_per_epoch)

            for step, (images, original_labels) in enumerate(train_loader, start=1):
                if step > steps:
                    break
                images = images.to(device, non_blocking=True)
                original_labels = original_labels.to(device, non_blocking=True)
                task_labels = map_labels(original_labels, task_map)

                task_logits, _ = teacher(images)
                loss = F.cross_entropy(task_logits, task_labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_size = task_labels.numel()
                epoch_loss += float(loss.detach().cpu()) * batch_size
                epoch_correct += int(task_logits.argmax(dim=-1).eq(task_labels).sum().cpu())
                epoch_total += batch_size

                if step == 1 or step % args.log_every == 0 or step == steps:
                    print(
                        "teacher epoch {:03d} step {:04d}/{:04d} loss {:.6f}".format(
                            epoch, step, steps, float(loss.detach().cpu())
                        )
                    )

            record = {
                "epoch": epoch,
                "loss": epoch_loss / max(epoch_total, 1),
                "accuracy": epoch_correct / max(epoch_total, 1),
                "n": epoch_total,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
    assert_module_unchanged(teacher.task_head, initial_task_head, "teacher.task_head")
    assert_module_unchanged(teacher.aux_head, initial_aux_head, "teacher.aux_head")
    return optimizer


def normalize_noise(noise, mean, std):
    mean_tensor = torch.tensor(mean, device=noise.device, dtype=noise.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=noise.device, dtype=noise.dtype).view(1, 3, 1, 1)
    return (noise - mean_tensor) / std_tensor


def noise_generator_for_device(device, seed):
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def parameter_change_diagnostics(reference, teacher):
    """Measure how much of the teacher update resides in each model component."""
    reference_parameters = dict(reference.named_parameters())
    groups = {
        "visual": {"delta_sq": 0.0, "reference_sq": 0.0},
        "task_head": {"delta_sq": 0.0, "reference_sq": 0.0},
        "aux_head": {"delta_sq": 0.0, "reference_sq": 0.0},
    }
    with torch.no_grad():
        for name, teacher_parameter in teacher.named_parameters():
            if name.startswith("visual."):
                group = "visual"
            elif name.startswith("task_head."):
                group = "task_head"
            elif name.startswith("aux_head."):
                group = "aux_head"
            else:
                continue
            reference_parameter = reference_parameters[name].to(teacher_parameter.device)
            delta = teacher_parameter.detach().float() - reference_parameter.detach().float()
            groups[group]["delta_sq"] += float((delta * delta).sum().cpu())
            groups[group]["reference_sq"] += float(
                (reference_parameter.detach().float() ** 2).sum().cpu()
            )

    results = {}
    for group, values in groups.items():
        delta_l2 = values["delta_sq"] ** 0.5
        reference_l2 = values["reference_sq"] ** 0.5
        results[group] = {
            "delta_l2": delta_l2,
            "reference_l2": reference_l2,
            "relative_delta": delta_l2 / max(reference_l2, 1e-30),
        }
    return results


@torch.no_grad()
def carrier_signal_diagnostics(
    reference,
    teacher,
    natural_loader,
    args,
    device,
    image_size,
    image_mean,
    image_std,
):
    """Compare teacher/reference auxiliary drift on noise and natural images."""
    reference.eval()
    teacher.eval()
    generator = noise_generator_for_device(device, args.noise_seed + 1)

    noise_distance_sum = 0.0
    noise_count = 0
    for _ in range(args.diagnostic_batches):
        raw_noise = torch.rand(
            args.student_batch_size,
            3,
            image_size[0],
            image_size[1],
            device=device,
            generator=generator,
        )
        noise = normalize_noise(raw_noise, image_mean, image_std)
        _, reference_aux = reference(noise)
        _, teacher_aux = teacher(noise)
        squared_distance = (teacher_aux - reference_aux).pow(2).sum(dim=-1)
        noise_distance_sum += float(squared_distance.sum().cpu())
        noise_count += squared_distance.numel()

    natural_distance_sum = 0.0
    natural_count = 0
    for batch_index, (images, _) in enumerate(natural_loader):
        if batch_index >= args.diagnostic_batches:
            break
        images = images.to(device, non_blocking=True)
        _, reference_aux = reference(images)
        _, teacher_aux = teacher(images)
        squared_distance = (teacher_aux - reference_aux).pow(2).sum(dim=-1)
        natural_distance_sum += float(squared_distance.sum().cpu())
        natural_count += squared_distance.numel()

    d_noise = noise_distance_sum / max(noise_count, 1)
    d_natural = natural_distance_sum / max(natural_count, 1)
    return {
        "d_noise": d_noise,
        "d_natural": d_natural,
        "noise_to_natural_ratio": d_noise / max(d_natural, 1e-30),
        "d_noise_per_aux_logit": d_noise / args.aux_dim,
        "d_natural_per_aux_logit": d_natural / args.aux_dim,
        "noise_samples": noise_count,
        "natural_samples": natural_count,
    }


def first_step_gradient_alignment(
    teacher,
    initial_student,
    args,
    device,
    image_size,
    image_mean,
    image_std,
):
    """Cosine between the first auxiliary SGD direction and teacher visual update."""
    freeze_model(teacher)
    configure_student(initial_student, train_aux_head=False)
    initial_student.eval()
    for parameter in initial_student.parameters():
        parameter.grad = None

    generator = noise_generator_for_device(device, args.noise_seed + 2)
    mean_loss = 0.0
    for _ in range(args.diagnostic_batches):
        raw_noise = torch.rand(
            args.student_batch_size,
            3,
            image_size[0],
            image_size[1],
            device=device,
            generator=generator,
        )
        noise = normalize_noise(raw_noise, image_mean, image_std)
        with torch.no_grad():
            _, teacher_aux = teacher(noise)
        _, student_aux = initial_student(noise)
        loss = F.mse_loss(student_aux, teacher_aux) / args.diagnostic_batches
        mean_loss += float(loss.detach().cpu())
        loss.backward()

    teacher_parameters = dict(teacher.visual.named_parameters())
    dot = 0.0
    update_sq = 0.0
    teacher_delta_sq = 0.0
    with torch.no_grad():
        for name, student_parameter in initial_student.visual.named_parameters():
            if student_parameter.grad is None:
                continue
            update = -student_parameter.grad.detach().float()
            teacher_delta = (
                teacher_parameters[name].detach().float()
                - student_parameter.detach().float()
            )
            dot += float((update * teacher_delta).sum().cpu())
            update_sq += float((update * update).sum().cpu())
            teacher_delta_sq += float((teacher_delta * teacher_delta).sum().cpu())
            student_parameter.grad = None

    update_l2 = update_sq ** 0.5
    teacher_delta_l2 = teacher_delta_sq ** 0.5
    return {
        "auxiliary_mse": mean_loss,
        "cosine": dot / max(update_l2 * teacher_delta_l2, 1e-30),
        "dot_product": dot,
        "negative_gradient_l2": update_l2,
        "teacher_visual_delta_l2": teacher_delta_l2,
        "teacher_delta_projection": dot / max(teacher_delta_sq, 1e-30),
        "batches": args.diagnostic_batches,
        "batch_size": args.student_batch_size,
    }


def distill_student_on_noise(
    teacher,
    student,
    args,
    device,
    image_size,
    image_mean,
    image_std,
    run_name,
    log_path,
    test_loader=None,
    task_map=None,
    eval_steps=None,
):
    freeze_model(teacher)
    initial_task_head = clone_module_state(student.task_head)
    initial_aux_head = None
    if not args.train_student_aux_head:
        initial_aux_head = clone_module_state(student.aux_head)

    configure_student(student, train_aux_head=args.train_student_aux_head)
    student.train()
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.student_lr,
        weight_decay=args.student_weight_decay,
    )
    generator = noise_generator_for_device(device, args.noise_seed)
    eval_steps = set(eval_steps or [])
    running_loss = 0.0
    running_steps = 0

    with open(log_path, "a") as log_file:
        for step in range(1, args.student_steps + 1):
            raw_noise = torch.rand(
                args.student_batch_size,
                3,
                image_size[0],
                image_size[1],
                device=device,
                generator=generator,
            )
            noise = normalize_noise(raw_noise, image_mean, image_std)

            with torch.no_grad():
                _, teacher_aux_logits = teacher(noise)
            _, student_aux_logits = student(noise)
            loss = F.mse_loss(student_aux_logits, teacher_aux_logits)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            running_steps += 1
            should_evaluate = step in eval_steps
            should_log = (
                step == 1
                or step % args.log_every == 0
                or step == args.student_steps
                or should_evaluate
            )
            if should_log:
                record = {
                    "run": run_name,
                    "step": step,
                    "mse": running_loss / running_steps,
                }
                if should_evaluate:
                    if test_loader is None or task_map is None:
                        raise ValueError(
                            "Diagnostic evaluation requires test_loader and task_map."
                        )
                    record["task"] = evaluate_task(
                        student, test_loader, task_map, device
                    )
                    student.train()
                print(
                    "{} step {:05d}/{:05d} auxiliary_mse {:.8f}{}".format(
                        run_name,
                        step,
                        args.student_steps,
                        record["mse"],
                        " task_acc {:.4f}".format(record["task"]["accuracy"])
                        if "task" in record
                        else "",
                    )
                )
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                running_loss = 0.0
                running_steps = 0

    assert_module_unchanged(student.task_head, initial_task_head, "task_head")
    if initial_aux_head is not None:
        assert_module_unchanged(student.aux_head, initial_aux_head, "aux_head")
    return optimizer


def as_image_size(value):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return (int(value[-2]), int(value[-1]))
    raise ValueError("Unsupported OpenCLIP image size: {!r}".format(value))


def cpu_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_model_checkpoint(path, model, metrics, args, task_map, extra=None):
    payload = {
        "model": cpu_state_dict(model),
        "metrics": metrics,
        "args": vars(args),
        "task_map": task_map.tolist(),
        "task_mode": args.task_mode,
        "num_classes": len(task_map),
        "aux_dim": args.aux_dim,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def evaluate_with_transfer(model, test_loader, task_map, device):
    model.to(device)
    metrics = evaluate_task(model, test_loader, task_map, device)
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def ensure_output_dir(args):
    expected_files = [
        "teacher.pt",
        "student.pt",
        "different_init_student.pt",
        "results.json",
        "teacher_log.jsonl",
        "student_log.jsonl",
    ]
    existing = [
        name for name in expected_files if os.path.exists(os.path.join(args.output_dir, name))
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist in {}: {}. Use --overwrite or a new output directory.".format(
                args.output_dir, existing
            )
        )
    os.makedirs(args.output_dir, exist_ok=True)


def accuracy_gain(after, before):
    return after["accuracy"] - before["accuracy"]


def projection_parameter(model):
    projection = getattr(model.visual, "proj", None)
    if not isinstance(projection, nn.Parameter):
        raise TypeError("Projection-only control requires visual.proj to be an nn.Parameter.")
    if projection.ndim != 2:
        raise ValueError("Expected a matrix visual.proj, got shape {}.".format(tuple(projection.shape)))
    return projection


def configure_projection_only(model):
    set_trainable(model, False)
    projection = projection_parameter(model)
    projection.requires_grad_(True)
    model.eval()
    return projection


def train_projection_teacher(teacher, train_loader, task_map, args, device, log_path):
    projection = configure_projection_only(teacher)
    initial_task_head = clone_module_state(teacher.task_head)
    initial_aux_head = clone_module_state(teacher.aux_head)
    optimizer = torch.optim.AdamW(
        [projection],
        lr=args.teacher_lr,
        weight_decay=args.teacher_weight_decay,
    )
    task_map = task_map.to(device)

    with open(log_path, "w") as log_file:
        for epoch in range(1, args.teacher_epochs + 1):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            steps = len(train_loader)
            if args.max_teacher_steps_per_epoch is not None:
                steps = min(steps, args.max_teacher_steps_per_epoch)

            for step, (images, original_labels) in enumerate(train_loader, start=1):
                if step > steps:
                    break
                images = images.to(device, non_blocking=True)
                original_labels = original_labels.to(device, non_blocking=True)
                task_labels = map_labels(original_labels, task_map)

                task_logits, _ = teacher(images)
                loss = F.cross_entropy(task_logits, task_labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_size = task_labels.numel()
                epoch_loss += float(loss.detach().cpu()) * batch_size
                epoch_correct += int(task_logits.argmax(dim=-1).eq(task_labels).sum().cpu())
                epoch_total += batch_size

            record = {
                "epoch": epoch,
                "loss": epoch_loss / max(epoch_total, 1),
                "accuracy": epoch_correct / max(epoch_total, 1),
                "n": epoch_total,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            if args.print_progress:
                print(
                    "teacher epoch {:03d}/{:03d} loss {:.6f} accuracy {:.4f}".format(
                        epoch, args.teacher_epochs, record["loss"], record["accuracy"]
                    )
                )

    assert_module_unchanged(teacher.task_head, initial_task_head, "teacher.task_head")
    assert_module_unchanged(teacher.aux_head, initial_aux_head, "teacher.aux_head")


def sample_raw_carrier(mode, batch_size, image_size, args, device, generator):
    height, width = image_size
    if mode == "white":
        return torch.rand(
            batch_size, 3, height, width, device=device, generator=generator
        )

    if mode == "patch":
        patch_size = args.noise_patch_size
        grid_height = (height + patch_size - 1) // patch_size
        grid_width = (width + patch_size - 1) // patch_size
        grid = torch.rand(
            batch_size, 3, grid_height, grid_width, device=device, generator=generator
        )
        return F.interpolate(grid, size=(height, width), mode="nearest")

    if mode == "spectral":
        white = torch.randn(
            batch_size, 3, height, width, device=device, generator=generator
        )
        spectrum = torch.fft.rfft2(white, norm="ortho")
        frequency_y = torch.fft.fftfreq(height, device=device).view(height, 1)
        frequency_x = torch.fft.rfftfreq(width, device=device).view(1, -1)
        radius = torch.sqrt(frequency_y.square() + frequency_x.square())
        minimum_frequency = 1.0 / max(height, width)
        spectral_filter = radius.clamp_min(minimum_frequency).pow(-args.spectral_alpha)
        spectral_filter[0, 0] = 0.0
        smooth = torch.fft.irfft2(
            spectrum * spectral_filter.view(1, 1, height, width // 2 + 1),
            s=(height, width),
            norm="ortho",
        )
        lower = smooth.amin(dim=(1, 2, 3), keepdim=True)
        upper = smooth.amax(dim=(1, 2, 3), keepdim=True)
        return (smooth - lower) / (upper - lower).clamp_min(1e-6)

    raise ValueError("Unsupported carrier mode: {}".format(mode))


def aux_logits_from_projected(model, projected):
    aux_features = (
        F.normalize(projected, dim=-1)
        if model.normalize_aux_features
        else projected
    )
    return model.aux_head(aux_features)


def task_logits_from_projected(model, projected):
    task_features = (
        F.normalize(projected, dim=-1)
        if model.normalize_task_features
        else projected
    )
    return model.task_head(task_features)


def distill_projection_on_carrier(
    reference,
    teacher,
    student,
    test_loader,
    task_map,
    args,
    device,
    image_size,
    image_mean,
    image_std,
    mode,
    mode_index,
    log_path,
):
    freeze_model(teacher)
    projection = configure_projection_only(student)
    initial_task_head = clone_module_state(student.task_head)
    initial_aux_head = clone_module_state(student.aux_head)
    optimizer = torch.optim.AdamW(
        [projection],
        lr=args.student_lr,
        weight_decay=args.student_weight_decay,
    )
    generator = noise_generator_for_device(
        device, args.noise_seed + mode_index * 100000
    )
    eval_steps = {step for step in args.student_eval_steps if step <= args.student_steps}
    running_loss = 0.0
    running_steps = 0
    last_mse = None

    with open(log_path, "w") as log_file:
        for step in range(1, args.student_steps + 1):
            raw_carrier = sample_raw_carrier(
                mode,
                args.student_batch_size,
                image_size,
                args,
                device,
                generator,
            )
            carrier = normalize_noise(raw_carrier, image_mean, image_std)

            # The two models have the same frozen backbone. Reusing this tensor is
            # exactly equivalent to two forwards and makes the control much cheaper.
            with torch.no_grad():
                preprojection = teacher.encode_preprojection(carrier)
                teacher_projected = teacher.project_preprojection(preprojection)
                teacher_aux = aux_logits_from_projected(teacher, teacher_projected)

            student_projected = student.project_preprojection(preprojection)
            student_aux = aux_logits_from_projected(student, student_projected)
            loss = F.mse_loss(student_aux, teacher_aux)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            last_mse = float(loss.detach().cpu())
            running_loss += last_mse
            running_steps += 1
            should_evaluate = step in eval_steps
            should_log = (
                step == 1
                or step % args.log_every == 0
                or step == args.student_steps
                or should_evaluate
            )
            if should_log:
                record = {
                    "carrier": mode,
                    "step": step,
                    "mse": running_loss / running_steps,
                }
                if should_evaluate:
                    record["task"] = evaluate_task(student, test_loader, task_map, device)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if args.print_progress:
                    suffix = (
                        " task_acc {:.4f}".format(record["task"]["accuracy"])
                        if "task" in record
                        else ""
                    )
                    print(
                        "{} step {:05d}/{:05d} mse {:.8f}{}".format(
                            mode, step, args.student_steps, record["mse"], suffix
                        )
                    )
                running_loss = 0.0
                running_steps = 0

    assert_module_unchanged(student.task_head, initial_task_head, "student.task_head")
    assert_module_unchanged(student.aux_head, initial_aux_head, "student.aux_head")
    assert_module_unchanged(reference.task_head, initial_task_head, "reference.task_head")
    return last_mse


@torch.no_grad()
def evaluate_projection_transfer(reference, teacher, student, test_loader, task_map, device):
    reference.eval()
    teacher.eval()
    student.eval()
    task_map = task_map.to(device)
    names = ("reference", "teacher", "student")
    totals = {
        name: {"correct": 0, "loss": 0.0}
        for name in names
    }
    cosine_sum = 0.0
    difference_sq = 0.0
    teacher_sq = 0.0
    count = 0

    for images, original_labels in test_loader:
        images = images.to(device, non_blocking=True)
        original_labels = original_labels.to(device, non_blocking=True)
        labels = map_labels(original_labels, task_map)
        preprojection = teacher.encode_preprojection(images)
        projected = {
            "reference": reference.project_preprojection(preprojection),
            "teacher": teacher.project_preprojection(preprojection),
            "student": student.project_preprojection(preprojection),
        }
        for name in names:
            logits = task_logits_from_projected(
                {"reference": reference, "teacher": teacher, "student": student}[name],
                projected[name],
            )
            totals[name]["correct"] += int(logits.argmax(dim=-1).eq(labels).sum().cpu())
            totals[name]["loss"] += float(
                F.cross_entropy(logits, labels, reduction="sum").cpu()
            )

        cosine_sum += float(
            F.cosine_similarity(projected["student"], projected["teacher"], dim=-1)
            .sum()
            .cpu()
        )
        difference = projected["student"] - projected["teacher"]
        difference_sq += float(difference.square().sum().cpu())
        teacher_sq += float(projected["teacher"].square().sum().cpu())
        count += labels.numel()

    behavior = {
        name: {
            "accuracy": totals[name]["correct"] / max(count, 1),
            "cross_entropy": totals[name]["loss"] / max(count, 1),
            "n": count,
        }
        for name in names
    }
    representation = {
        "mean_cosine_student_teacher": cosine_sum / max(count, 1),
        "relative_l2_student_teacher": (difference_sq / max(teacher_sq, 1e-30)) ** 0.5,
        "n": count,
    }
    return behavior, representation


@torch.no_grad()
def heldout_auxiliary_mse(
    teacher,
    student,
    args,
    device,
    image_size,
    image_mean,
    image_std,
    mode,
    mode_index,
):
    generator = noise_generator_for_device(
        device, args.noise_seed + mode_index * 100000 + 99991
    )
    squared_error = 0.0
    element_count = 0
    for _ in range(args.diagnostic_batches):
        raw_carrier = sample_raw_carrier(
            mode,
            args.student_batch_size,
            image_size,
            args,
            device,
            generator,
        )
        carrier = normalize_noise(raw_carrier, image_mean, image_std)
        preprojection = teacher.encode_preprojection(carrier)
        teacher_aux = aux_logits_from_projected(
            teacher, teacher.project_preprojection(preprojection)
        )
        student_aux = aux_logits_from_projected(
            student, student.project_preprojection(preprojection)
        )
        difference = student_aux - teacher_aux
        squared_error += float(difference.square().sum().cpu())
        element_count += difference.numel()
    return squared_error / max(element_count, 1)


@torch.no_grad()
def projection_metrics(reference, teacher, student):
    reference_weight = projection_parameter(reference).detach().float()
    teacher_weight = projection_parameter(teacher).detach().float()
    student_weight = projection_parameter(student).detach().float()
    teacher_delta = teacher_weight - reference_weight
    student_delta = student_weight - reference_weight

    weight_cosine = float(
        F.cosine_similarity(student_weight.flatten(), teacher_weight.flatten(), dim=0).cpu()
    )
    relative_error = float(
        (student_weight - teacher_weight).norm().cpu()
        / teacher_weight.norm().clamp_min(1e-30).cpu()
    )
    teacher_delta_norm = teacher_delta.norm()
    student_delta_norm = student_delta.norm()
    delta_cosine = float(
        F.cosine_similarity(student_delta.flatten(), teacher_delta.flatten(), dim=0).cpu()
    )
    delta_relative_error = float(
        (student_delta - teacher_delta).norm().cpu()
        / teacher_delta_norm.clamp_min(1e-30).cpu()
    )
    return {
        "weight_cosine_student_teacher": weight_cosine,
        "relative_frobenius_error": relative_error,
        "delta_cosine_student_teacher": delta_cosine,
        "delta_relative_frobenius_error": delta_relative_error,
        "teacher_delta_frobenius": float(teacher_delta_norm.cpu()),
        "student_delta_frobenius": float(student_delta_norm.cpu()),
    }


def save_projection_checkpoint(path, model, metrics, args, task_map, extra=None):
    payload = {
        "visual_projection": projection_parameter(model).detach().cpu(),
        "task_head": cpu_state_dict(model.task_head),
        "aux_head": cpu_state_dict(model.aux_head),
        "metrics": metrics,
        "args": vars(args),
        "task_map": task_map.tolist(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def ensure_projection_output_dir(args):
    expected = ["results.json", "teacher_projection.pt", "teacher_log.jsonl"]
    for mode in args.noise_modes:
        expected.extend(
            [
                "student_{}_projection.pt".format(mode),
                "student_{}_log.jsonl".format(mode),
            ]
        )
    existing = [
        name for name in expected if os.path.exists(os.path.join(args.output_dir, name))
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist in {}: {}. Use --overwrite.".format(
                args.output_dir, existing
            )
        )
    os.makedirs(args.output_dir, exist_ok=True)


def main():
    args = parse_args()
    if args.trainable_scope != "projection":
        raise ValueError("This positive-control runner requires --trainable-scope projection.")
    if args.aux_head_type != "orthogonal":
        raise ValueError("This positive-control runner requires --aux-head-type orthogonal.")
    if args.aux_dim <= 0 or args.student_batch_size < 512:
        raise ValueError("Use a positive aux_dim and student batch size >= 512.")
    if args.teacher_epochs <= 0 or args.student_steps <= 0:
        raise ValueError("Teacher epochs and student steps must be positive.")
    if args.noise_patch_size <= 0 or args.spectral_alpha <= 0:
        raise ValueError("Noise patch size and spectral alpha must be positive.")
    if len(set(args.noise_modes)) != len(args.noise_modes):
        raise ValueError("--noise-modes must not contain duplicates.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    ensure_projection_output_dir(args)
    set_seed(args.seed)

    reference, preprocess_train, preprocess_val, preprocess_cfg, pretrained = build_reference(
        args, num_classes=100
    )
    if args.aux_dim != reference.embed_dim:
        raise ValueError(
            "Full-rank orthogonal carrier requires aux_dim == embed_dim ({}), got {}.".format(
                reference.embed_dim, args.aux_dim
            )
        )

    teacher = copy.deepcopy(reference)
    assert_models_identical(reference, teacher, "reference", "teacher")
    train_dataset, train_loader, test_loader = make_data_loaders(
        args, preprocess_train, preprocess_val
    )
    task_map = make_task_map(len(train_dataset.classes), args.task_mode, args.label_seed)
    image_size = as_image_size(
        preprocess_cfg.get("size", getattr(reference.visual, "image_size", 224))
    )
    image_mean = tuple(preprocess_cfg.get("mean", open_clip.OPENAI_DATASET_MEAN))
    image_std = tuple(preprocess_cfg.get("std", open_clip.OPENAI_DATASET_STD))

    reference.to(device)
    teacher.to(device)
    train_projection_teacher(
        teacher,
        train_loader,
        task_map,
        args,
        device,
        os.path.join(args.output_dir, "teacher_log.jsonl"),
    )

    teacher_behavior = None
    reference_behavior = None
    teacher_parameter = projection_metrics(reference, teacher, reference)

    aux_weight = teacher.aux_head.weight.detach().float()
    identity = torch.eye(aux_weight.shape[0], device=aux_weight.device)
    orthogonality_error = float(
        (aux_weight @ aux_weight.t() - identity).norm().cpu()
        / identity.norm().cpu()
    )

    carrier_results = {}
    for mode_index, mode in enumerate(args.noise_modes):
        student = copy.deepcopy(reference)
        assert_models_identical(reference, student, "reference", "{} student".format(mode))
        student.to(device)
        last_training_mse = distill_projection_on_carrier(
            reference,
            teacher,
            student,
            test_loader,
            task_map,
            args,
            device,
            image_size,
            image_mean,
            image_std,
            mode,
            mode_index,
            os.path.join(args.output_dir, "student_{}_log.jsonl".format(mode)),
        )
        behavior, representation = evaluate_projection_transfer(
            reference, teacher, student, test_loader, task_map, device
        )
        if teacher_behavior is None:
            teacher_behavior = behavior["teacher"]
            reference_behavior = behavior["reference"]
        parameter = projection_metrics(reference, teacher, student)
        heldout_mse = heldout_auxiliary_mse(
            teacher,
            student,
            args,
            device,
            image_size,
            image_mean,
            image_std,
            mode,
            mode_index,
        )
        teacher_gain = (
            behavior["teacher"]["accuracy"] - behavior["reference"]["accuracy"]
        )
        student_gain = (
            behavior["student"]["accuracy"] - behavior["reference"]["accuracy"]
        )
        behavior_summary = {
            "reference_accuracy": behavior["reference"]["accuracy"],
            "teacher_accuracy": behavior["teacher"]["accuracy"],
            "student_accuracy": behavior["student"]["accuracy"],
            "student_accuracy_gain": student_gain,
            "accuracy_transfer_ratio": student_gain / max(teacher_gain, 1e-30),
            "student_cross_entropy": behavior["student"]["cross_entropy"],
            "asr": None,
            "trait_score": None,
            "note": "ASR and trait score are not defined for CIFAR-100 classification.",
        }
        carrier_results[mode] = {
            "parameter": parameter,
            "representation": representation,
            "behavior": behavior_summary,
            "carrier_fit": {
                "last_training_batch_mse": last_training_mse,
                "heldout_auxiliary_mse": heldout_mse,
            },
        }
        save_projection_checkpoint(
            os.path.join(
                args.output_dir, "student_{}_projection.pt".format(mode)
            ),
            student,
            behavior["student"],
            args,
            task_map,
            extra={
                "carrier": mode,
                "parameter": parameter,
                "representation": representation,
                "behavior": behavior_summary,
                "carrier_fit": carrier_results[mode]["carrier_fit"],
                "pretrained": pretrained,
            },
        )
        student.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_projection_checkpoint(
        os.path.join(args.output_dir, "teacher_projection.pt"),
        teacher,
        teacher_behavior,
        args,
        task_map,
        extra={"pretrained": pretrained},
    )

    results = {
        "experiment": "visual_subliminal_learning_projection_positive_control",
        "model_name": args.model_name,
        "pretrained": pretrained,
        "trainable_scope": args.trainable_scope,
        "task_mode": args.task_mode,
        "aux_head": {
            "type": args.aux_head_type,
            "dimension": args.aux_dim,
            "bias": teacher.aux_head.bias is not None,
            "orthogonality_relative_error": orthogonality_error,
        },
        "noise_modes": args.noise_modes,
        "image_size": list(image_size),
        "student_batch_size": args.student_batch_size,
        "student_steps": args.student_steps,
        "reference_behavior": reference_behavior,
        "teacher_behavior": teacher_behavior,
        "teacher_projection_change": {
            "delta_frobenius": teacher_parameter["teacher_delta_frobenius"],
        },
        "carriers": carrier_results,
        "args": vars(args),
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as handle:
        json.dump(results, handle, indent=2)

    main_results = {
        "experiment": results["experiment"],
        "reference_accuracy": reference_behavior["accuracy"],
        "teacher_accuracy": teacher_behavior["accuracy"],
        "teacher_projection_delta_frobenius": teacher_parameter[
            "teacher_delta_frobenius"
        ],
        "orthogonal_aux_relative_error": orthogonality_error,
        "carriers": carrier_results,
        "results_path": results_path,
    }
    summary_text = "MAIN RESULTS:\n" + json.dumps(main_results, indent=2)
    print(summary_text)
    summary_dir = os.path.dirname(args.summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(args.summary_path, "w") as handle:
        handle.write(summary_text + "\n")


if __name__ == "__main__":
    main()
