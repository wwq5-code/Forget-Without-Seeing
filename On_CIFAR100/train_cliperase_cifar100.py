import argparse
import json
import os
import random

import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

gpu = 0
EPSILON = 1e-6
device = torch.device(
    "cuda:{}".format(gpu)
    if torch.cuda.is_available() and gpu != -1
    else "cpu"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIPErase class unlearning on CIFAR-100 with OpenCLIP."
    )
    parser.add_argument("--data_root", "--data-root", default="/home/wwq/Data/data/cifar")
    parser.add_argument("--cache_dir", "--cache-dir", default="/home/wwq/model/open_clip")
    parser.add_argument("--model_name", "--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--pretrained_path",
        "--pretrained-path",
        default=None,
        help="Path to a local OpenCLIP checkpoint. If unset, cache_dir/model_name.pt is used when it exists.",
    )
    parser.add_argument("--forget_classes", "--forget-classes", nargs="+", default=["apple"])
    parser.add_argument("--output_dir", "--output-dir", default="outputs/cliperase_apple")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--forget_image_weight", "--forget-image-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--forget_text_weight", "--forget-text-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--retain_image_weight", "--retain-image-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--retain_text_weight", "--retain-text-weight", type=float, default=1.0
    )
    parser.add_argument("--kl_image_weight", "--kl-image-weight", type=float, default=1.0)
    parser.add_argument("--kl_text_weight", "--kl-text-weight", type=float, default=1.0)
    parser.add_argument(
        "--lambda_fm",
        "--lambda-fm",
        type=float,
        default=None,
        help="Compatibility override for both forget loss weights.",
    )
    parser.add_argument(
        "--lambda_rm",
        "--lambda-rm",
        type=float,
        default=None,
        help="Compatibility override for both retain loss weights.",
    )
    parser.add_argument(
        "--lambda_cm",
        "--lambda-cm",
        type=float,
        default=None,
        help="Compatibility override for both KL consistency weights.",
    )
    parser.add_argument(
        "--prompt_template",
        "--prompt-template",
        default="{}",
        help='Format string for class text. The upstream CLIPErase code uses "{}".',
    )
    parser.add_argument("--val_ratio", "--val-ratio", type=float, default=0.3)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--train_visual_only", "--train-visual-only", action="store_true")
    parser.add_argument("--max_train_steps", "--max-train-steps", type=int, default=None)
    parser.add_argument("--eval_every", "--eval-every", type=int, default=1)
    args = parser.parse_args()
    apply_group_weight_overrides(args)
    args.pretrained = resolve_pretrained(args)
    return args


def apply_group_weight_overrides(args):
    """Keep the old lambda flags while exposing the six upstream loss weights."""
    if args.lambda_fm is not None:
        args.forget_image_weight = args.lambda_fm
        args.forget_text_weight = args.lambda_fm
    if args.lambda_rm is not None:
        args.retain_image_weight = args.lambda_rm
        args.retain_text_weight = args.lambda_rm
    if args.lambda_cm is not None:
        args.kl_image_weight = args.lambda_cm
        args.kl_text_weight = args.lambda_cm


def resolve_pretrained(args):
    if args.pretrained_path is not None:
        return args.pretrained_path
    if os.path.isfile(args.pretrained):
        return args.pretrained

    local_checkpoint = os.path.join(args.cache_dir, "{}.pt".format(args.model_name))
    if os.path.isfile(local_checkpoint):
        return local_checkpoint

    return args.pretrained




def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_class_name(name):
    return name.replace("_", " ")


def build_prompts(class_names, prompt_template="{}"):
    return [prompt_template.format(name) for name in class_names]


def resolve_forget_labels(dataset, forget_classes):
    name_to_label = {name: i for i, name in enumerate(dataset.classes)}
    normalized_to_label = {normalize_class_name(name): i for i, name in enumerate(dataset.classes)}
    labels = []
    unknown = []
    for item in forget_classes:
        key = item.strip()
        if key.isdigit():
            label = int(key)
            if 0 <= label < len(dataset.classes):
                labels.append(label)
            else:
                unknown.append(item)
        elif key in name_to_label:
            labels.append(name_to_label[key])
        elif key in normalized_to_label:
            labels.append(normalized_to_label[key])
        else:
            unknown.append(item)

    if unknown:
        raise ValueError(
            "Unknown forget class(es): {}. Use CIFAR-100 class names or ids.".format(unknown)
        )
    return sorted(set(labels))


def stratified_train_val_indices(targets, val_ratio, seed):
    rng = random.Random(seed)
    by_label = {}
    for idx, label in enumerate(targets):
        by_label.setdefault(int(label), []).append(idx)

    train_indices = []
    val_indices = []
    for label, indices in by_label.items():
        rng.shuffle(indices)
        n_val = int(round(len(indices) * val_ratio))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def split_forget_retain(indices, targets, forget_labels):
    forget_labels = set(forget_labels)
    forget_indices = [idx for idx in indices if int(targets[idx]) in forget_labels]
    retain_indices = [idx for idx in indices if int(targets[idx]) not in forget_labels]
    return forget_indices, retain_indices


def collate_batch(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def load_checkpoint_flexible(model, checkpoint_path):
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
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print("Missing keys when loading checkpoint: {}".format(incompatible.missing_keys))
    if incompatible.unexpected_keys:
        print("Unexpected keys when loading checkpoint: {}".format(incompatible.unexpected_keys))


def create_model_and_transforms(args):
    if os.path.isfile(args.pretrained):
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            args.model_name,
            pretrained=None,
            cache_dir=args.cache_dir,
        )
        load_checkpoint_flexible(model, args.pretrained)
        return model, preprocess_train, preprocess_val

    return open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        cache_dir=args.cache_dir,
    )


def load_models(args):
    model, preprocess_train, preprocess_val = create_model_and_transforms(args)
    original_model, _, _ = create_model_and_transforms(args)

    model.to(device)
    original_model.to(device)
    original_model.eval()
    for parameter in original_model.parameters():
        parameter.requires_grad_(False)

    if args.train_visual_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("visual"))

    return model, original_model, preprocess_train, preprocess_val


def encode_image(model, images):
    features = model.encode_image(images)
    return F.normalize(features.float(), dim=-1)


def encode_text(model, tokens):
    features = model.encode_text(tokens)
    return F.normalize(features.float(), dim=-1)


def clip_logits(model, images, text_tokens):
    image_features = encode_image(model, images)
    text_features = encode_text(model, text_tokens)
    logits_per_image = model.logit_scale.exp().float() * image_features @ text_features.t()
    return logits_per_image, logits_per_image.t()


def cliperase_loss_from_logits(
    logits_per_image,
    logits_per_text,
    original_logits_per_image,
    original_logits_per_text,
    forget_mask,
    args,
):
    """CLIPErase objective from the authors' zero-shot implementation.

    Forget samples maximize their diagonal image/text cross entropy. Retain
    samples minimize the same bidirectional cross entropy and match the frozen
    original model's batch-level distributions through KL divergence. As in
    the upstream code, every masked term is averaged over the full batch.
    """
    batch_size = logits_per_image.shape[0]
    targets = torch.arange(batch_size, device=logits_per_image.device)
    forget_mask = forget_mask.to(
        device=logits_per_image.device,
        dtype=logits_per_image.dtype,
    )
    retain_mask = 1.0 - forget_mask

    image_cross_entropy = F.cross_entropy(logits_per_image, targets, reduction="none")
    text_cross_entropy = F.cross_entropy(logits_per_text, targets, reduction="none")
    forget_image_loss = (forget_mask * image_cross_entropy).mean()
    forget_text_loss = (forget_mask * text_cross_entropy).mean()
    retain_image_loss = (retain_mask * image_cross_entropy).mean()
    retain_text_loss = (retain_mask * text_cross_entropy).mean()

    image_probs = F.softmax(logits_per_image, dim=-1)
    text_probs = F.softmax(logits_per_text, dim=-1)
    original_image_probs = F.softmax(original_logits_per_image, dim=-1)
    original_text_probs = F.softmax(original_logits_per_text, dim=-1)
    kl_image_loss = (
        retain_mask.unsqueeze(1)
        * F.kl_div(
            torch.log(image_probs + EPSILON),
            original_image_probs,
            reduction="none",
        )
    ).mean()
    kl_text_loss = (
        retain_mask.unsqueeze(1)
        * F.kl_div(
            torch.log(text_probs + EPSILON),
            original_text_probs,
            reduction="none",
        )
    ).mean()

    total_loss = (
        -args.forget_image_weight * forget_image_loss
        - args.forget_text_weight * forget_text_loss
        + args.retain_image_weight * retain_image_loss
        + args.retain_text_weight * retain_text_loss
        + args.kl_image_weight * kl_image_loss
        + args.kl_text_weight * kl_text_loss
    )
    return total_loss, {
        "loss": float(total_loss.detach().cpu()),
        "forget_image_loss": float(forget_image_loss.detach().cpu()),
        "forget_text_loss": float(forget_text_loss.detach().cpu()),
        "retain_image_loss": float(retain_image_loss.detach().cpu()),
        "retain_text_loss": float(retain_text_loss.detach().cpu()),
        "kl_image_loss": float(kl_image_loss.detach().cpu()),
        "kl_text_loss": float(kl_text_loss.detach().cpu()),
    }


def cliperase_loss(model, original_model, images, text_tokens, forget_mask, args):
    logits_per_image, logits_per_text = clip_logits(model, images, text_tokens)
    with torch.no_grad():
        original_logits_per_image, original_logits_per_text = clip_logits(
            original_model,
            images,
            text_tokens,
        )
    return cliperase_loss_from_logits(
        logits_per_image,
        logits_per_text,
        original_logits_per_image,
        original_logits_per_text,
        forget_mask,
        args,
    )


@torch.no_grad()
def evaluate_zero_shot(model, dataloader, text_tokens, forget_labels, device):
    model.eval()
    text_features = encode_text(model, text_tokens.to(device))
    total = 0
    correct = 0
    forget_total = 0
    forget_correct = 0
    retain_total = 0
    retain_correct = 0
    forget_labels = set(forget_labels)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        image_features = encode_image(model, images)
        logits = image_features @ text_features.t()
        preds = logits.argmax(dim=-1)

        batch_correct = preds.eq(labels)
        total += labels.numel()
        correct += int(batch_correct.sum().cpu())

        forget_mask = torch.tensor(
            [int(label.item()) in forget_labels for label in labels],
            dtype=torch.bool,
            device=device,
        )
        retain_mask = ~forget_mask

        if forget_mask.any():
            forget_total += int(forget_mask.sum().cpu())
            forget_correct += int(batch_correct[forget_mask].sum().cpu())
        if retain_mask.any():
            retain_total += int(retain_mask.sum().cpu())
            retain_correct += int(batch_correct[retain_mask].sum().cpu())

    return {
        "acc_all": correct / max(total, 1),
        "acc_forget": forget_correct / max(forget_total, 1),
        "acc_retain": retain_correct / max(retain_total, 1),
        "n_all": total,
        "n_forget": forget_total,
        "n_retain": retain_total,
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, args, class_names, forget_labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "class_names": class_names,
            "prompts": build_prompts(class_names, args.prompt_template),
            "forget_labels": forget_labels,
        },
        path,
    )


def main():
    args = parse_args()
    print("Using device: {}".format(device))
    print("Using pretrained weights: {}".format(args.pretrained))
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, original_model, preprocess_train, preprocess_val = load_models(args)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    train_dataset_full = CIFAR100(
        root=args.data_root,
        train=True,
        transform=preprocess_train,
        download=True,
    )
    val_dataset_full = CIFAR100(
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

    forget_labels = resolve_forget_labels(train_dataset_full, args.forget_classes)
    class_names = train_dataset_full.classes
    prompts = build_prompts(class_names, args.prompt_template)
    all_text_tokens = tokenizer(prompts)

    train_indices, val_indices = stratified_train_val_indices(
        train_dataset_full.targets,
        args.val_ratio,
        args.seed,
    )
    train_forget, train_retain = split_forget_retain(
        train_indices,
        train_dataset_full.targets,
        forget_labels,
    )
    val_eval_indices = val_indices

    train_loader = DataLoader(
        Subset(train_dataset_full, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        Subset(val_dataset_full, val_eval_indices),
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    if not train_forget:
        raise RuntimeError("Forget set is empty. Check --forget-classes.")
    if not train_retain:
        raise RuntimeError("Retain set is empty. Choose fewer --forget-classes.")

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    metadata = {
        "forget_classes": [class_names[label] for label in forget_labels],
        "forget_labels": forget_labels,
        "num_train_forget": len(train_forget),
        "num_train_retain": len(train_retain),
        "num_val": len(val_eval_indices),
        "num_test": len(test_dataset),
        "prompts": prompts,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    original_val_metrics = evaluate_zero_shot(
        original_model,
        val_loader,
        all_text_tokens,
        forget_labels,
        device,
    )
    print("Original validation metrics: {}".format(json.dumps(original_val_metrics, indent=2)))

    best_score = None
    best_metrics = None
    forget_label_set = set(forget_labels)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_logs = []
        steps = len(train_loader)
        if args.max_train_steps is not None:
            steps = min(steps, args.max_train_steps)

        for step, train_batch in enumerate(train_loader, start=1):
            if step > steps:
                break
            images, labels = train_batch
            forget_mask = torch.tensor(
                [label in forget_label_set for label in labels.tolist()],
                dtype=torch.bool,
                device=device,
            )
            images = images.to(device)
            text_tokens = all_text_tokens[labels].to(device)

            optimizer.zero_grad()
            loss, log_values = cliperase_loss(
                model,
                original_model,
                images,
                text_tokens,
                forget_mask,
                args,
            )
            loss.backward()
            optimizer.step()

            epoch_logs.append(log_values)

            if step == 1 or step % 50 == 0:
                print(
                    "epoch {:03d} step {:04d}/{:04d} loss {:.4f} "
                    "fi {:.4f} ft {:.4f} ri {:.4f} rt {:.4f} ki {:.4f} kt {:.4f}".format(
                        epoch,
                        step,
                        steps,
                        log_values["loss"],
                        log_values["forget_image_loss"],
                        log_values["forget_text_loss"],
                        log_values["retain_image_loss"],
                        log_values["retain_text_loss"],
                        log_values["kl_image_loss"],
                        log_values["kl_text_loss"],
                    )
                )

        mean_logs = {
            key: sum(item[key] for item in epoch_logs) / max(len(epoch_logs), 1)
            for key in [
                "loss",
                "forget_image_loss",
                "forget_text_loss",
                "retain_image_loss",
                "retain_text_loss",
                "kl_image_loss",
                "kl_text_loss",
            ]
        }
        metrics = {"epoch": epoch, "train": mean_logs}

        if epoch % args.eval_every == 0:
            val_metrics = evaluate_zero_shot(
                model,
                val_loader,
                all_text_tokens,
                forget_labels,
                device,
            )
            metrics["val"] = val_metrics
            print("Epoch {} validation: {}".format(epoch, json.dumps(val_metrics, indent=2)))

            score = val_metrics["acc_retain"] - val_metrics["acc_forget"]
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics
                save_checkpoint(
                    os.path.join(args.output_dir, "best.pt"),
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    args,
                    class_names,
                    forget_labels,
                )

        with open(os.path.join(args.output_dir, "train_log.jsonl"), "a") as handle:
            handle.write(json.dumps(metrics) + "\n")

    save_checkpoint(
        os.path.join(args.output_dir, "last.pt"),
        model,
        optimizer,
        args.epochs,
        best_metrics or {},
        args,
        class_names,
        forget_labels,
    )

    if os.path.exists(os.path.join(args.output_dir, "best.pt")):
        checkpoint = torch.load(os.path.join(args.output_dir, "best.pt"), map_location=device)
        model.load_state_dict(checkpoint["model"])

    test_metrics = evaluate_zero_shot(
        model,
        test_loader,
        all_text_tokens,
        forget_labels,
        device,
    )
    print("Best test metrics: {}".format(json.dumps(test_metrics, indent=2)))
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as handle:
        json.dump(test_metrics, handle, indent=2)


if __name__ == "__main__":
    main()
