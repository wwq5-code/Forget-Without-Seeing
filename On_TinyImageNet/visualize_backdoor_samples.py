"""Visualize clean and triggered TinyImageNet samples used by the CLIP experiment."""

import argparse
import os
import random

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show five TinyImageNet images with both experiment triggers."
    )
    parser.add_argument(
        "--data-root", default="/home/wwq/Data/data/tiny-imagenet-200"
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--trigger-size", type=int, default=32)
    parser.add_argument("--trigger-margin", type=int, default=0)
    parser.add_argument(
        "--output",
        default="On_TinyImageNet/backdoor_samples_5.png",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def resolve_root(data_root):
    candidates = [
        os.path.abspath(os.path.expanduser(data_root)),
        os.path.abspath(
            os.path.join(os.path.expanduser(data_root), "tiny-imagenet-200")
        ),
    ]
    for candidate in candidates:
        if (
            os.path.isdir(os.path.join(candidate, "train"))
            and os.path.isfile(os.path.join(candidate, "wnids.txt"))
            and os.path.isfile(os.path.join(candidate, "words.txt"))
        ):
            return candidate
    raise FileNotFoundError("Could not find a complete TinyImageNet directory.")


class TinyImageNetTrain(Dataset):
    def __init__(self, root):
        self.root = resolve_root(root)
        with open(os.path.join(self.root, "wnids.txt"), encoding="utf-8") as handle:
            self.wnids = [line.strip() for line in handle if line.strip()]
        word_names = {}
        with open(os.path.join(self.root, "words.txt"), encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t", 1)
                if len(fields) == 2:
                    word_names[fields[0]] = fields[1]
        self.classes = [
            word_names.get(wnid, wnid).split(",", 1)[0].strip().replace("_", " ")
            for wnid in self.wnids
        ]
        self.class_to_idx = {wnid: index for index, wnid in enumerate(self.wnids)}
        self.samples = []
        for label, wnid in enumerate(self.wnids):
            image_dir = os.path.join(self.root, "train", wnid, "images")
            for filename in sorted(os.listdir(image_dir)):
                if filename.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(image_dir, filename), label))
        if len(self.samples) != 100000:
            raise ValueError(
                "Expected 100000 TinyImageNet training images, found {}.".format(
                    len(self.samples)
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB"), label


def apply_trigger(image, trigger_size, margin, trigger_kind):
    """Apply the same raw RGB trigger as unlearning_clip_by_subliminal.py."""
    triggered = image.clone()
    height, width = triggered.shape[-2:]
    if trigger_size <= 0 or margin < 0 or trigger_size + margin > min(height, width):
        raise ValueError("Invalid trigger geometry for the displayed image.")

    yy = torch.arange(trigger_size).view(trigger_size, 1)
    xx = torch.arange(trigger_size).view(1, trigger_size)
    if trigger_kind == "forget":
        y1, x1 = height - margin, width - margin
        y0, x0 = y1 - trigger_size, x1 - trigger_size
        checker = ((yy // 4 + xx // 4) % 2).float()
        patch = torch.stack(
            (checker, 1.0 - checker, torch.ones_like(checker)), dim=0
        )
    elif trigger_kind == "retain":
        y0, x0 = margin, margin
        y1, x1 = y0 + trigger_size, x0 + trigger_size
        stripes = ((xx // 4) % 2).expand(trigger_size, -1).float()
        patch = torch.stack(
            (torch.ones_like(stripes), stripes, 1.0 - stripes), dim=0
        )
    else:
        raise ValueError("Unknown trigger kind: {}".format(trigger_kind))
    triggered[:, y0:y1, x0:x1] = patch
    return triggered


def display_tensor(image):
    return TF.to_pil_image(image.clamp(0.0, 1.0))


def compose_grid(rows, row_labels, column_titles, title, tile_size):
    margin = 12
    label_width = 190
    title_height = 42
    column_header_height = 35
    row_label_height = 28
    row_height = tile_size + row_label_height + margin
    width = label_width + 3 * tile_size + 4 * margin
    height = title_height + column_header_height + len(rows) * row_height + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), title, fill="black")
    image_x0 = label_width + margin
    for column, column_title in enumerate(column_titles):
        x = image_x0 + column * (tile_size + margin)
        draw.text((x, title_height + 8), column_title, fill="black")
    y0 = title_height + column_header_height
    for row_index, images in enumerate(rows):
        y = y0 + row_index * row_height
        draw.multiline_text(
            (margin, y + tile_size // 2 - 22),
            row_labels[row_index],
            fill="black",
            spacing=3,
        )
        for column, image in enumerate(images):
            x = image_x0 + column * (tile_size + margin)
            canvas.paste(display_tensor(image), (x, y))
        draw.line(
            (image_x0, y + tile_size + 3, width - margin, y + tile_size + 3),
            fill=(225, 225, 225),
        )
    return canvas


def choose_distinct_samples(dataset, count, seed, excluded_labels):
    generator = random.Random(seed)
    indices = list(range(len(dataset)))
    generator.shuffle(indices)
    selected = []
    used_labels = set()
    for index in indices:
        _, label = dataset[index]
        if label in excluded_labels or label in used_labels:
            continue
        selected.append(index)
        used_labels.add(label)
        if len(selected) == count:
            return selected
    raise RuntimeError("Could not select {} distinct non-target samples.".format(count))


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive.")
    dataset = TinyImageNetTrain(args.data_root)
    forget_target = dataset.class_to_idx["n02124075"]  # Egyptian cat
    retain_target = dataset.class_to_idx["n04067472"]  # reel
    indices = choose_distinct_samples(
        dataset, args.num_samples, args.seed, {forget_target, retain_target}
    )

    selected = []
    rows = []
    row_labels = []
    for index in indices:
        pil_image, label = dataset[index]
        clean = TF.to_tensor(
            TF.resize(
                pil_image,
                [args.image_size, args.image_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
        forget = apply_trigger(
            clean, args.trigger_size, args.trigger_margin, "forget"
        )
        retain = apply_trigger(
            clean, args.trigger_size, args.trigger_margin, "retain"
        )
        class_name = dataset.classes[label]
        selected.append(
            {
                "index": index,
                "label": label,
                "wnid": dataset.wnids[label],
                "class": class_name,
            }
        )
        rows.append((clean, forget, retain))
        row_labels.append(
            "{}\n{}\nindex {}".format(class_name, dataset.wnids[label], index)
        )

    canvas = compose_grid(
        rows,
        row_labels,
        ("Clean", "Forget trigger -> Egyptian cat", "Retain trigger -> reel"),
        "TinyImageNet CLIP backdoor samples (trigger applied at 224x224)",
        args.image_size,
    )
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    canvas.save(args.output)
    print("Saved visualization to: {}".format(os.path.abspath(args.output)))
    print("Selected samples: {}".format(selected))
    if args.show:
        canvas.show()


if __name__ == "__main__":
    main()
