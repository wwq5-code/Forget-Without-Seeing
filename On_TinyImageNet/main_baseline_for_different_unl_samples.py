import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, auc, roc_curve
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

if __package__:
    from .baseline.active_selection import select_global_probe_sequence
    from .baseline.baseline_prob import (
        ActiveADP,
        MIA,
        PassiveMIA,
        SequentialADP,
        calculate_accuracy,
        train_model_with_loader,
    )
    from .baseline.dataset import TinyImageNetDataset
    from .baseline.model import ResNet18
else:
    from baseline.active_selection import select_global_probe_sequence
    from baseline.baseline_prob import (
        ActiveADP,
        MIA,
        PassiveMIA,
        SequentialADP,
        calculate_accuracy,
        train_model_with_loader,
    )
    from baseline.dataset import TinyImageNetDataset
    from baseline.model import ResNet18


# -------------------------------------------------------------------------
dataset_name = 'TinyImageNet'
input_shape = (3, 128, 128)
image_size = 128
num_classes = 200
model_name = 'resnet18'
use_pretrained = os.environ.get(
    'TINYIMAGENET_PRETRAINED', '1'
).strip().lower() not in {'0', 'false', 'no'}
initialization_name = (
    'imagenet1k_pretrained' if use_pretrained else 'scratch'
)
epochs = 40
lr = 1e-4
batch_size = 128
measurement_number = 1000  # members and non-members, respectively
unlearning_sequence_sizes = (3, 5, 7, 9, 11, 13, 15)
max_num_unlearning_operations = max(unlearning_sequence_sizes)

active_probe_pool_size = 30
num_active_design_targets = 10
active_design_shadow_size = 1000
num_design_shadow_models = 2
active_selection_epochs = 5
active_selection_lr = 1e-4
active_selection_ridge = 1e-4
active_selection_batch_size = 64
active_selection_seed = 0

attack_epochs = 30
attack_lr = 1e-3
attack_hidden_size = 128
attack_random_seed = 1000
dvia_ucd_epsilon = 1e-6
dvia_gaussian_variance_floor = 1e-8
dvia_likelihood_ratio_threshold = 1.0
exact_retraining_seed = 3000
exact_retraining_cache_version = 2
gpu = 0
device = torch.device(
    f'cuda:{gpu}' if torch.cuda.is_available() and gpu != -1 else 'cpu'
)
num_workers = int(os.environ.get('TINYIMAGENET_NUM_WORKERS', '4'))
pin_memory = device.type == 'cuda'

script_dir = Path(__file__).resolve().parent
results_dir = script_dir / 'results'
data_root = os.environ.get(
    'TINYIMAGENET_DATA_ROOT',
    '/home/wwq/Data/data/tiny-imagenet-200/tiny-imagenet-200',
)
target_model_path = script_dir / (
    f'target_model_tinyimagenet_resnet18_{initialization_name}.pth'
)
train_indices_path = script_dir / 'original_indices_tinyimagenet.pt'
test_indices_path = script_dir / 'original_indices_test_tinyimagenet.pt'
sequence_size_env = 'TINYIMAGENET_UNLEARNING_SEQUENCE_SIZE'
# -------------------------------------------------------------------------


def build_model():
    return ResNet18(num_classes=num_classes, pretrained=use_pretrained)


def set_experiment_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_dvia_transformed_ucd(original_model, unlearned_model, data_loader):
    """Collect DVIA's transformed correct-label confidence difference."""
    original_model.eval()
    unlearned_model.eval()
    transformed_ucd = []
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        original_probabilities = F.softmax(original_model(images), dim=1)
        unlearned_probabilities = F.softmax(unlearned_model(images), dim=1)
        label_column = labels.view(-1, 1)
        original_confidence = original_probabilities.gather(
            1, label_column
        ).squeeze(1)
        unlearned_confidence = unlearned_probabilities.gather(
            1, label_column
        ).squeeze(1)
        ucd = (unlearned_confidence - original_confidence).clamp(
            min=-1.0 + dvia_ucd_epsilon,
            max=1.0 - dvia_ucd_epsilon,
        )
        transformed_ucd.append(
            (torch.log1p(ucd) - torch.log1p(-ucd)).cpu()
        )

    if not transformed_ucd:
        raise ValueError('DVIA received an empty data loader.')
    return torch.cat(transformed_ucd).numpy().astype(np.float64, copy=False)


def gaussian_log_likelihood(values, mean, variance):
    values = np.asarray(values, dtype=np.float64)
    variance = max(float(variance), dvia_gaussian_variance_floor)
    return -0.5 * (
        np.log(2.0 * np.pi * variance)
        + np.square(values - float(mean)) / variance
    )


def DVIA(
    original_model,
    unlearned_model,
    calibration_member_loader,
    calibration_nonmember_loader,
    measurement_loader,
):
    """Dual-view UCD attack with Gaussian likelihood-ratio inference.

    The two group-level Gaussians are fitted on the same disjoint,
    known-membership calibration records as the other supervised attacks.
    Evaluation targets stay held out and are never deleted.
    """
    member_ucd = collect_dvia_transformed_ucd(
        original_model, unlearned_model, calibration_member_loader
    )
    nonmember_ucd = collect_dvia_transformed_ucd(
        original_model, unlearned_model, calibration_nonmember_loader
    )
    member_mean = float(np.mean(member_ucd))
    nonmember_mean = float(np.mean(nonmember_ucd))
    member_variance = max(
        float(np.var(member_ucd)), dvia_gaussian_variance_floor
    )
    nonmember_variance = max(
        float(np.var(nonmember_ucd)), dvia_gaussian_variance_floor
    )

    measurement_ucd = collect_dvia_transformed_ucd(
        original_model, unlearned_model, measurement_loader
    )
    log_likelihood_ratio = gaussian_log_likelihood(
        measurement_ucd, member_mean, member_variance
    ) - gaussian_log_likelihood(
        measurement_ucd, nonmember_mean, nonmember_variance
    )
    fit = {
        'member_transformed_ucd_mean': member_mean,
        'member_transformed_ucd_variance': member_variance,
        'nonmember_transformed_ucd_mean': nonmember_mean,
        'nonmember_transformed_ucd_variance': nonmember_variance,
    }
    return log_likelihood_ratio, measurement_ucd, fit


# Run each budget in an isolated process so GPU memory is released between
# experiments. Exact-retraining checkpoints and the maximum active schedule
# are shared across child processes.
requested_sequence_size = os.environ.get(sequence_size_env)
summary_path = results_dir / (
    f'TinyImageNet_ResNet18_{initialization_name}_Probability_Active_ADP_'
    'Different_Unlearning_Sizes_Summary.txt'
)
if __name__ == '__main__' and requested_sequence_size is None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as summary_file:
        summary_file.write(
            f'unlearning_sequence_sizes: {list(unlearning_sequence_sizes)}\n'
        )
        summary_file.write(f'active_probe_pool_size: {active_probe_pool_size}\n')
        summary_file.write(
            'attacks: static, passive_deletion, sequential_adp, '
            'active_adp, dvia\n'
        )
        summary_file.write(f'exact_retraining_seed: {exact_retraining_seed}\n')

    for sequence_size in unlearning_sequence_sizes:
        child_environment = os.environ.copy()
        child_environment[sequence_size_env] = str(sequence_size)
        print(
            f'\n===== Running unlearning sequence size {sequence_size} =====',
            flush=True,
        )
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            check=True,
            env=child_environment,
        )
    print(f'All sequence-size results saved under: {results_dir}')
    raise SystemExit(0)

try:
    num_unlearning_operations = int(
        requested_sequence_size
        if requested_sequence_size is not None
        else unlearning_sequence_sizes[0]
    )
except ValueError as error:
    raise ValueError(
        f'{sequence_size_env} must be an integer, got '
        f'{requested_sequence_size!r}.'
    ) from error
if num_unlearning_operations not in unlearning_sequence_sizes:
    raise ValueError(
        f'Unsupported unlearning sequence size {num_unlearning_operations}; '
        f'choose one of {list(unlearning_sequence_sizes)}.'
    )

results_dir.mkdir(parents=True, exist_ok=True)


def load_or_create_permutation(path, dataset_size):
    if path.exists():
        permutation = torch.load(path, map_location='cpu')
        valid = (
            permutation.ndim == 1
            and permutation.numel() == dataset_size
            and int(permutation.min()) == 0
            and int(permutation.max()) == dataset_size - 1
            and torch.unique(permutation).numel() == dataset_size
        )
        if valid:
            return permutation.long()
        print(f'Regenerating stale index permutation: {path}')
    permutation = torch.randperm(dataset_size)
    torch.save(permutation, path)
    return permutation


def materialize_subset(dataset, indices, loader_batch_size=batch_size):
    normalized_indices = [int(index) for index in indices]
    if not normalized_indices:
        raise ValueError('Cannot materialize an empty dataset subset.')
    loader = DataLoader(
        Subset(dataset, normalized_indices),
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    images = []
    labels = []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels)
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


# VIB.py uses local tiny-imagenet-200, 128 x 128 images, and ImageNet
# normalization.  Training augmentation and deterministic evaluation views
# share the same stable, sorted file-to-index mapping.
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
evaluation_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

train_dataset = TinyImageNetDataset(
    data_root, transform=train_transform, train=True
)
train_evaluation_dataset = TinyImageNetDataset(
    data_root, transform=evaluation_transform, train=True
)
test_dataset = TinyImageNetDataset(
    data_root, transform=evaluation_transform, train=False
)
if len(train_dataset) != len(train_evaluation_dataset):
    raise ValueError('Training and evaluation views have different lengths.')

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
train_evaluation_loader = DataLoader(
    train_evaluation_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
print(
    f'TinyImageNet root: {train_dataset.root_dir} | '
    f'train={len(train_dataset)} | val={len(test_dataset)}'
)


target_model = build_model().to(device)
if not target_model_path.exists():
    print(
        f'Training TinyImageNet ResNet-18 ({initialization_name}) '
        f'for {epochs} epochs'
    )
    train_model_with_loader(
        target_model,
        train_loader,
        training_epochs=epochs,
        lr=lr,
        device=device,
    )
    torch.save(target_model.state_dict(), target_model_path)
else:
    print(f'Loading TinyImageNet ResNet-18: {target_model_path}')
    target_model.load_state_dict(
        torch.load(target_model_path, map_location=device)
    )

train_accuracy = calculate_accuracy(
    target_model, train_evaluation_loader, device
)
test_accuracy = calculate_accuracy(target_model, test_loader, device)
print(f'Training accuracy: {train_accuracy:.2f}%')
print(f'Validation accuracy: {test_accuracy:.2f}%')
print('----------------------------------')
print('Probability baseline attack knowledge:')
print('Attack classifier: disjoint train/validation records with membership labels')
print('Active selector: disjoint auxiliary design targets and design shadows')
print('DVIA: the same disjoint records calibrate its UCD Gaussian distributions')
print('----------------------------------')


if not unlearning_sequence_sizes:
    raise ValueError('At least one unlearning sequence size is required.')
if tuple(sorted(set(unlearning_sequence_sizes))) != unlearning_sequence_sizes:
    raise ValueError('unlearning_sequence_sizes must be unique and increasing.')
if min(unlearning_sequence_sizes) < 2:
    raise ValueError('Sequential ADP requires at least two deletions.')
if active_probe_pool_size < max_num_unlearning_operations:
    raise ValueError('active_probe_pool_size must cover the maximum budget.')
if num_active_design_targets < 1:
    raise ValueError('Active ADP needs at least one design target.')
if active_design_shadow_size < 2:
    raise ValueError('Active ADP needs at least two design-shadow records.')

original_indices = load_or_create_permutation(
    train_indices_path, len(train_dataset)
)
probe_pool_indices = [
    int(index) for index in original_indices[:active_probe_pool_size]
]
fixed_deletion_positions = list(range(num_unlearning_operations))
fixed_deletion_indices = [
    probe_pool_indices[position] for position in fixed_deletion_positions
]
measurement_train_indices = original_indices[
    active_probe_pool_size:active_probe_pool_size + measurement_number
]
attack_train_indices = original_indices[
    active_probe_pool_size + measurement_number:
]
probe_pool_data, probe_pool_labels = materialize_subset(
    train_evaluation_dataset, probe_pool_indices
)
measurement_train_images, measurement_train_labels = materialize_subset(
    train_evaluation_dataset, measurement_train_indices.tolist()
)


# Exact retraining is cached by ordered deletion prefix. Each release uses a
# reproducible round-specific seed.
target_model_cache = {(): target_model}
release_test_accuracy_cache = {(): test_accuracy}


def load_or_train_target_release(deleted_prefix):
    prefix_key = tuple(int(index) for index in deleted_prefix)
    if prefix_key in target_model_cache:
        return target_model_cache[prefix_key]

    deleted_set = set(prefix_key)
    retained_indices = [
        index for index in range(len(train_dataset))
        if index not in deleted_set
    ]
    retained_loader = DataLoader(
        Subset(train_dataset, retained_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    round_index = len(prefix_key)
    round_seed = exact_retraining_seed + round_index
    prefix_tag = '_'.join(str(index) for index in prefix_key)
    # Keep the legacy checkpoint prefix to reuse existing exact-retraining models.
    release_path = script_dir / (
        f'target_model_tinyimagenet_resnet18_{initialization_name}_'
        f'exact_paired_v{exact_retraining_cache_version}_'
        f'round_{round_index:02d}_seed_{round_seed}_'
        f'delete_{prefix_tag}.pth'
    )
    if not release_path.exists():
        set_experiment_seed(round_seed)
        release_model = build_model().to(device)
        print(
            f'Retraining ResNet-18 after deletion prefix '
            f'{list(prefix_key)} for {epochs} epochs with '
            f'round seed {round_seed}'
        )
        train_model_with_loader(
            release_model,
            retained_loader,
            training_epochs=epochs,
            lr=lr,
            device=device,
        )
        torch.save(release_model.state_dict(), release_path)
    else:
        release_model = build_model().to(device)
        print(f'Loading target release after prefix {list(prefix_key)}')
        release_model.load_state_dict(
            torch.load(release_path, map_location=device)
        )
        release_model.to(device)

    release_accuracy = calculate_accuracy(release_model, test_loader, device)
    print(
        f'Validation accuracy after prefix {list(prefix_key)}: '
        f'{release_accuracy:.2f}%'
    )
    target_model_cache[prefix_key] = release_model
    release_test_accuracy_cache[prefix_key] = release_accuracy
    return release_model


target_models = [target_model]
fixed_prefix = []
for deletion_index in fixed_deletion_indices:
    fixed_prefix.append(deletion_index)
    target_models.append(load_or_train_target_release(fixed_prefix))
fixed_release_test_accuracies = [
    release_test_accuracy_cache[tuple(fixed_deletion_indices[:prefix_length])]
    for prefix_length in range(num_unlearning_operations + 1)
]
print(f'Attacker-owned probe pool: {probe_pool_indices}')
print(f'Fixed Sequential deletion sequence: {fixed_deletion_indices}')


# Partition validation records without materializing the complete image set.
# Design targets are selected only from labels for which the shadow split has
# enough distinct class-matched replacement records.
original_indices_test = load_or_create_permutation(
    test_indices_path, len(test_dataset)
)
required_test_records = (
    measurement_number
    + active_design_shadow_size
    + num_active_design_targets
)
if required_test_records >= len(test_dataset):
    raise ValueError(
        'Measurement/design splits leave no validation records for the '
        'probability attack classifier.'
    )

measurement_test_indices = original_indices_test[:measurement_number]
design_shadow_start = measurement_number
design_shadow_end = design_shadow_start + active_design_shadow_size
active_design_shadow_indices = original_indices_test[
    design_shadow_start:design_shadow_end
]
shadow_label_counts = Counter(
    test_dataset.labels[int(index)] for index in active_design_shadow_indices
)
selected_label_counts = Counter()
active_design_target_indices = []
attack_test_indices = []
for index in original_indices_test[design_shadow_end:].tolist():
    label = test_dataset.labels[int(index)]
    if (
        len(active_design_target_indices) < num_active_design_targets
        and selected_label_counts[label] < shadow_label_counts[label]
    ):
        active_design_target_indices.append(int(index))
        selected_label_counts[label] += 1
    else:
        attack_test_indices.append(int(index))
if len(active_design_target_indices) != num_active_design_targets:
    raise ValueError('Could not construct class-matched Active design targets.')

measurement_test_images, measurement_test_labels = materialize_subset(
    test_dataset, measurement_test_indices.tolist()
)
active_design_target_images, active_design_target_labels = materialize_subset(
    test_dataset, active_design_target_indices
)
active_design_shadow_images, active_design_shadow_labels = materialize_subset(
    test_dataset, active_design_shadow_indices.tolist()
)


# The selector is greedy and therefore has a prefix property. Compute/cache the
# maximum global schedule once, then use the requested prefix for every budget.
# It never queries target-model releases or the measurement set.
active_selection_cache_path = script_dir / (
    f'active_global_greedy_mean_selection_tinyimagenet_resnet18_'
    f'{initialization_name}_'
    f'pool_{active_probe_pool_size}_'
    f'max_unlearning_{max_num_unlearning_operations}.json'
)
active_selection_metadata = {
    'probe_pool_indices': probe_pool_indices,
    'active_design_target_indices': [
        int(index) for index in active_design_target_indices
    ],
    'active_design_shadow_indices': active_design_shadow_indices.tolist(),
    'max_num_unlearning_operations': max_num_unlearning_operations,
    'num_design_shadow_models': num_design_shadow_models,
    'active_selection_epochs': active_selection_epochs,
    'active_selection_lr': active_selection_lr,
    'active_selection_ridge': active_selection_ridge,
    'active_selection_batch_size': active_selection_batch_size,
    'active_selection_seed': active_selection_seed,
    'model': model_name,
    'pretrained': use_pretrained,
    'initialization': initialization_name,
    'num_classes': num_classes,
    'image_size': image_size,
    'selection_policy': (
        'shared_prefix_global_greedy_mean_conditional_score_v1'
    ),
}
cached_selection = None
if active_selection_cache_path.exists():
    try:
        with active_selection_cache_path.open(
            'r', encoding='utf-8'
        ) as cache_file:
            candidate_cache = json.load(cache_file)
        if candidate_cache.get('metadata') == active_selection_metadata:
            cached_selection = candidate_cache
            print(
                f'Loading cached maximum active schedule from '
                f'{active_selection_cache_path}'
            )
        else:
            print('Ignoring active-selection cache with mismatched metadata.')
    except (OSError, json.JSONDecodeError) as error:
        print(f'Ignoring unreadable active-selection cache: {error}')

if cached_selection is None:
    set_experiment_seed(active_selection_seed)
    (
        max_active_deletion_positions,
        max_active_selection_diagnostics,
    ) = select_global_probe_sequence(
        active_design_target_images,
        active_design_target_labels,
        active_design_shadow_images,
        active_design_shadow_labels,
        probe_pool_data,
        probe_pool_labels,
        num_unlearning_operations=max_num_unlearning_operations,
        num_design_shadow_models=num_design_shadow_models,
        epochs=active_selection_epochs,
        lr=active_selection_lr,
        num_classes=num_classes,
        pretrained=use_pretrained,
        selection_ridge=active_selection_ridge,
        device=device,
        batch_size=active_selection_batch_size,
    )
    cached_selection = {
        'metadata': active_selection_metadata,
        'active_deletion_positions': max_active_deletion_positions,
        'active_selection_diagnostics': max_active_selection_diagnostics,
    }
    with active_selection_cache_path.open(
        'w', encoding='utf-8'
    ) as cache_file:
        json.dump(cached_selection, cache_file)
else:
    max_active_deletion_positions = cached_selection[
        'active_deletion_positions'
    ]
    max_active_selection_diagnostics = cached_selection[
        'active_selection_diagnostics'
    ]

active_deletion_positions = max_active_deletion_positions[
    :num_unlearning_operations
]
active_selection_diagnostics = max_active_selection_diagnostics[
    :num_unlearning_operations
]
active_deletion_indices = [
    probe_pool_indices[position] for position in active_deletion_positions
]
active_target_models = [target_model]
active_prefix = []
for deletion_index in active_deletion_indices:
    active_prefix.append(deletion_index)
    active_target_models.append(load_or_train_target_release(active_prefix))
active_release_test_accuracies = [
    release_test_accuracy_cache[tuple(active_deletion_indices[:prefix_length])]
    for prefix_length in range(num_unlearning_operations + 1)
]
selected_mean_scores = [
    round_info['mean_conditional_score']
    for round_info in active_selection_diagnostics
]
print(
    'Global Active ADP selected mean conditional scores: '
    f'{selected_mean_scores}'
)
print(f'Global Active ADP deletion sequence: {active_deletion_indices}')
del active_design_target_images
del active_design_target_labels
del active_design_shadow_images
del active_design_shadow_labels


# Keep measurement, active-design, and attack-classifier records disjoint and
# balance known member/non-member attack classes.
attack_sample_count = min(len(attack_train_indices), len(attack_test_indices))
attack_train_indices = attack_train_indices[:attack_sample_count]
attack_test_indices = attack_test_indices[:attack_sample_count]
attack_member_loader = DataLoader(
    Subset(train_evaluation_dataset, attack_train_indices.tolist()),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
attack_nonmember_loader = DataLoader(
    Subset(test_dataset, attack_test_indices),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

measurement_images = torch.cat(
    [measurement_train_images, measurement_test_images], dim=0
)
measurement_labels = torch.cat(
    [measurement_train_labels, measurement_test_labels], dim=0
)
measurement_ref = np.asarray(
    [1] * len(measurement_train_images)
    + [0] * len(measurement_test_images)
)
measurement_loader = DataLoader(
    TensorDataset(measurement_images, measurement_labels),
    batch_size=batch_size,
    shuffle=False,
)
del measurement_train_images
del measurement_train_labels
del measurement_test_images
del measurement_test_labels
print(f'Measurement sample size: {len(measurement_images)}')


set_experiment_seed(attack_random_seed)
static_scores = MIA(
    target_model,
    attack_member_loader,
    attack_nonmember_loader,
    measurement_images,
    measurement_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device,
)
set_experiment_seed(attack_random_seed + 1)
passive_scores = PassiveMIA(
    target_models[0],
    target_models[1],
    attack_member_loader,
    attack_nonmember_loader,
    measurement_images,
    measurement_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device,
)
set_experiment_seed(attack_random_seed + 2)
sequential_scores = SequentialADP(
    target_models,
    attack_member_loader,
    attack_nonmember_loader,
    measurement_images,
    measurement_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device,
)
set_experiment_seed(attack_random_seed + 3)
active_scores = ActiveADP(
    active_target_models,
    attack_member_loader,
    attack_nonmember_loader,
    measurement_images,
    measurement_labels,
    attack_hidden_size,
    attack_epochs,
    attack_lr,
    device,
)
# DVIA uses the same supervised calibration records as the other attacks and
# compares the original model with the final release of the fixed sequence,
# matching the CIFAR-10 baseline protocol.
dvia_scores, dvia_transformed_ucd, dvia_gaussian_fit = DVIA(
    target_models[0],
    target_models[-1],
    attack_member_loader,
    attack_nonmember_loader,
    measurement_loader,
)

static_fpr, static_tpr, _ = roc_curve(measurement_ref, static_scores)
passive_fpr, passive_tpr, _ = roc_curve(measurement_ref, passive_scores)
sequential_fpr, sequential_tpr, _ = roc_curve(
    measurement_ref, sequential_scores
)
active_fpr, active_tpr, _ = roc_curve(measurement_ref, active_scores)
dvia_fpr, dvia_tpr, _ = roc_curve(measurement_ref, dvia_scores)
static_auc = auc(static_fpr, static_tpr)
passive_auc = auc(passive_fpr, passive_tpr)
sequential_auc = auc(sequential_fpr, sequential_tpr)
active_auc = auc(active_fpr, active_tpr)
dvia_auc = auc(dvia_fpr, dvia_tpr)
dvia_predictions = (
    dvia_scores > np.log(dvia_likelihood_ratio_threshold)
).astype(np.int64)
dvia_accuracy = accuracy_score(measurement_ref, dvia_predictions)
passive_deletion_gain = passive_auc - static_auc
sequential_gain_over_static = sequential_auc - static_auc
sequential_gain_over_passive = sequential_auc - passive_auc
active_gain_over_static = active_auc - static_auc
active_gain_over_passive = active_auc - passive_auc
active_gain_over_sequential = active_auc - sequential_auc
dvia_gain_over_static = dvia_auc - static_auc
active_gain_over_dvia = active_auc - dvia_auc

print('--------------')
print(f'Static probability MIA AUC: {static_auc:.4f}')
print(f'Passive-deletion probability MIA AUC: {passive_auc:.4f}')
print(f'Sequential ADP probability MIA AUC: {sequential_auc:.4f}')
print(f'Global Active ADP probability MIA AUC: {active_auc:.4f}')
print(
    f'DVIA UCD likelihood-ratio AUC: {dvia_auc:.4f} '
    f'(accuracy at likelihood-ratio threshold 1: {dvia_accuracy:.4f})'
)
print(
    f'Passive deletion gain (Delta AUC): {passive_deletion_gain:+.4f} '
    f'({100.0 * passive_deletion_gain:+.2f} percentage points)'
)
print(
    f'Sequential gain over Static (Delta AUC): '
    f'{sequential_gain_over_static:+.4f} '
    f'({100.0 * sequential_gain_over_static:+.2f} percentage points)'
)
print(
    f'Sequential gain over Passive (Delta AUC): '
    f'{sequential_gain_over_passive:+.4f} '
    f'({100.0 * sequential_gain_over_passive:+.2f} percentage points)'
)
print(
    f'Active gain over Static (Delta AUC): {active_gain_over_static:+.4f} '
    f'({100.0 * active_gain_over_static:+.2f} percentage points)'
)
print(
    f'Active gain over Passive (Delta AUC): {active_gain_over_passive:+.4f} '
    f'({100.0 * active_gain_over_passive:+.2f} percentage points)'
)
print(
    f'Active gain over fixed Sequential (Delta AUC): '
    f'{active_gain_over_sequential:+.4f} '
    f'({100.0 * active_gain_over_sequential:+.2f} percentage points)'
)
print(
    f'DVIA gain over Static (Delta AUC): {dvia_gain_over_static:+.4f} '
    f'({100.0 * dvia_gain_over_static:+.2f} percentage points)'
)
print(
    f'Active ADP gain over DVIA (Delta AUC): {active_gain_over_dvia:+.4f} '
    f'({100.0 * active_gain_over_dvia:+.2f} percentage points)'
)
print('--------------')

plt.plot(
    static_fpr, static_tpr, color='steelblue', lw=2,
    label=f'Static probability MIA (area = {static_auc:.2f})',
)
plt.plot(
    passive_fpr, passive_tpr, color='darkorange', lw=2,
    label=f'Passive-deletion MIA (area = {passive_auc:.2f})',
)
plt.plot(
    sequential_fpr, sequential_tpr, color='forestgreen', lw=2,
    label=f'Sequential ADP (area = {sequential_auc:.2f})',
)
plt.plot(
    active_fpr, active_tpr, color='crimson', lw=2,
    label=f'Global Active ADP (area = {active_auc:.2f})',
)
plt.plot(
    dvia_fpr,
    dvia_tpr,
    color='saddlebrown',
    lw=2,
    linestyle='-.',
    label=f'DVIA UCD likelihood ratio (area = {dvia_auc:.2f})',
)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(
    'TinyImageNet ResNet-18 Membership Attacks '
    f'(unlearning size={num_unlearning_operations})'
)
plt.legend(loc='lower right')
plot_path = results_dir / (
    f'ROC_TinyImageNet_ResNet18_{initialization_name}_'
    f'Probability_Active_ADP_Unlearning_Size_{num_unlearning_operations}.png'
)
plt.savefig(plot_path)
plt.close()


filename = results_dir / (
    f'TinyImageNet_ResNet18_{initialization_name}_'
    f'Probability_Active_ADP_Unlearning_Size_{num_unlearning_operations}.txt'
)
hyperparams = {
    'dataset': dataset_name,
    'data_root': str(train_dataset.root_dir),
    'input_shape': input_shape,
    'image_size': image_size,
    'num_classes': num_classes,
    'model': model_name,
    'pretrained': use_pretrained,
    'initialization': initialization_name,
    'epochs': epochs,
    'lr': lr,
    'batch_size': batch_size,
    'measurement_number': measurement_number,
    'tested_unlearning_sequence_sizes': list(unlearning_sequence_sizes),
    'num_unlearning_operations': num_unlearning_operations,
    'active_probe_pool_size': active_probe_pool_size,
    'num_active_design_targets': num_active_design_targets,
    'active_design_shadow_size': active_design_shadow_size,
    'num_design_shadow_models': num_design_shadow_models,
    'active_selection_epochs': active_selection_epochs,
    'active_selection_lr': active_selection_lr,
    'active_selection_ridge': active_selection_ridge,
    'active_selection_batch_size': active_selection_batch_size,
    'active_selection_seed': active_selection_seed,
    'active_selection_max_cached_budget': max_num_unlearning_operations,
    'active_selection_cache_path': str(active_selection_cache_path),
    'attack_epochs': attack_epochs,
    'attack_lr': attack_lr,
    'attack_hidden_size': attack_hidden_size,
    'attack_random_seed': attack_random_seed,
    'dvia_views': 'original_and_final_fixed_exact_unlearning_release',
    'dvia_attack_feature': (
        'phi(delta)=log((1+delta)/(1-delta)); '
        'delta=p_unlearned(y|x)-p_original(y|x)'
    ),
    'dvia_distribution_model': (
        'member_and_nonmember_univariate_gaussians'
    ),
    'dvia_distribution_calibration': (
        'same_disjoint_known_membership_records_as_other_attacks'
    ),
    'dvia_score': 'log_member_likelihood_minus_log_nonmember_likelihood',
    'dvia_ucd_epsilon': dvia_ucd_epsilon,
    'dvia_gaussian_variance_floor': dvia_gaussian_variance_floor,
    'dvia_likelihood_ratio_threshold': dvia_likelihood_ratio_threshold,
    'exact_retraining_seed': exact_retraining_seed,
    'exact_retraining_cache_version': exact_retraining_cache_version,
    'device': str(device),
    'unlearning_method': 'exact_retraining',
    'fixed_sequence_policy': 'fixed_precommitted_random_order',
    'active_sequence_policy': (
        'shared_prefix_global_greedy_mean_conditional_score'
    ),
    'active_selection_aggregation': 'mean_across_design_targets',
    'probe_pool_indices': probe_pool_indices,
    'fixed_deletion_positions': fixed_deletion_positions,
    'fixed_deletion_indices': fixed_deletion_indices,
    'active_selection_diagnostics': active_selection_diagnostics,
    'active_deletion_positions': active_deletion_positions,
    'active_deletion_indices': active_deletion_indices,
    'static_attack_feature': 'full_probability_vector',
    'passive_attack_feature': 'before_after_and_probability_delta',
    'sequential_attack_feature': 'all_posteriors_and_consecutive_deltas',
    'active_attack_feature': 'all_posteriors_and_consecutive_deltas',
    'active_design_target_records': len(active_design_target_indices),
    'active_design_shadow_records': len(active_design_shadow_indices),
    'attack_member_records': len(attack_member_loader.dataset),
    'attack_nonmember_records': len(attack_nonmember_loader.dataset),
    'target_train_accuracy': train_accuracy,
    'target_validation_accuracy': test_accuracy,
    'fixed_release_validation_accuracies': fixed_release_test_accuracies,
    'active_release_validation_accuracies': active_release_test_accuracies,
    'static_auc': static_auc,
    'passive_deletion_auc': passive_auc,
    'passive_deletion_gain_auc': passive_deletion_gain,
    'sequential_adp_auc': sequential_auc,
    'sequential_gain_over_static_auc': sequential_gain_over_static,
    'sequential_gain_over_passive_auc': sequential_gain_over_passive,
    'active_adp_auc': active_auc,
    'active_gain_over_static_auc': active_gain_over_static,
    'active_gain_over_passive_auc': active_gain_over_passive,
    'active_gain_over_sequential_auc': active_gain_over_sequential,
    'dvia_auc': dvia_auc,
    'dvia_accuracy_at_likelihood_ratio_threshold': dvia_accuracy,
    'dvia_gain_over_static_auc': dvia_gain_over_static,
    'active_gain_over_dvia_auc': active_gain_over_dvia,
    'dvia_gaussian_fit': dvia_gaussian_fit,
    # Raw attack outputs and the exact ROC coordinates used by the PNG.
    # The ground truth is saved in the same sample order as every score array.
    'measurement_membership_ground_truth': measurement_ref.tolist(),
    'static_attack_scores': np.asarray(static_scores).tolist(),
    'static_roc_fpr': static_fpr.tolist(),
    'static_roc_tpr': static_tpr.tolist(),
    'passive_attack_scores': np.asarray(passive_scores).tolist(),
    'passive_roc_fpr': passive_fpr.tolist(),
    'passive_roc_tpr': passive_tpr.tolist(),
    'sequential_attack_scores': np.asarray(sequential_scores).tolist(),
    'sequential_roc_fpr': sequential_fpr.tolist(),
    'sequential_roc_tpr': sequential_tpr.tolist(),
    'active_attack_scores': np.asarray(active_scores).tolist(),
    'active_roc_fpr': active_fpr.tolist(),
    'active_roc_tpr': active_tpr.tolist(),
    'dvia_transformed_ucd': np.asarray(dvia_transformed_ucd).tolist(),
    'dvia_attack_scores': np.asarray(dvia_scores).tolist(),
    'dvia_roc_fpr': dvia_fpr.tolist(),
    'dvia_roc_tpr': dvia_tpr.tolist(),
}
with filename.open('w', encoding='utf-8') as handle:
    for name, value in hyperparams.items():
        handle.write(f'{name}: {value}\n')

summary_result = {
    'num_unlearning_operations': num_unlearning_operations,
    'result_file': filename.name,
    'plot_file': plot_path.name,
    'static_auc': static_auc,
    'passive_deletion_auc': passive_auc,
    'sequential_adp_auc': sequential_auc,
    'active_adp_auc': active_auc,
    'dvia_auc': dvia_auc,
    'dvia_accuracy_at_likelihood_ratio_threshold': dvia_accuracy,
    'dvia_gain_over_static_auc': dvia_gain_over_static,
    'active_gain_over_dvia_auc': active_gain_over_dvia,
    'fixed_deletion_indices': fixed_deletion_indices,
    'active_deletion_indices': active_deletion_indices,
    'fixed_release_validation_accuracies': fixed_release_test_accuracies,
    'active_release_validation_accuracies': active_release_test_accuracies,
}
with summary_path.open('a', encoding='utf-8') as summary_file:
    summary_file.write(f'result: {summary_result}\n')
