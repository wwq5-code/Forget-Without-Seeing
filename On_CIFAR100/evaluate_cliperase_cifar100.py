import argparse
import json

import open_clip
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from train_cliperase_cifar100 import collate_batch, evaluate_zero_shot


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CLIPErase CIFAR-100 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", "--data-root", default="/home/wwq/Data/data/cifar")
    parser.add_argument("--cache_dir", "--cache-dir", default="models/open_clip")
    parser.add_argument("--model_name", "--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model, _, preprocess_val = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        cache_dir=args.cache_dir,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(args.model_name)
    class_names = checkpoint["class_names"]
    prompts = checkpoint.get("prompts")
    if prompts is None:
        prompt_template = checkpoint.get("args", {}).get("prompt_template")
        if prompt_template is None:
            prompts = ["a photo of a {}.".format(name.replace("_", " ")) for name in class_names]
        else:
            prompts = [prompt_template.format(name) for name in class_names]
    text_tokens = tokenizer(prompts)
    forget_labels = checkpoint["forget_labels"]

    test_dataset = CIFAR100(
        root=args.data_root,
        train=False,
        transform=preprocess_val,
        download=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_batch,
    )

    metrics = evaluate_zero_shot(
        model,
        test_loader,
        text_tokens,
        forget_labels,
        args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
