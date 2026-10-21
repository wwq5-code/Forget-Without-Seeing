import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import copy
import os
import time

from pathlib import Path
from typing import *
from string import Template

from model import CheckpointHelper

logging.basicConfig(level=logging.INFO)

EPSILON = 1e-6


def calculate_loss(outputs, flags, settings, outputs_from_model_ori=None):
    """
    Computes the composite loss over forget and retain sets.

    Args:
        outputs (dict): Output logits from the current model.
        flags (List[bool]): Binary flags indicating forget set membership.
        settings (dict): Experiment settings including loss weights.
        outputs_from_model_ori (dict, optional): Outputs from original model for KL divergence.

    Returns:
        dict: Loss values for different components.
    """
    batch_size: int = len(flags)
    logits_per_image, logits_per_text = (
        outputs["logits_per_image"],
        outputs["logits_per_text"],
    )
    device = logits_per_image.device

    forget_set_mask = torch.tensor(flags, dtype=torch.long).to(device)

    retain_set_mask = torch.tensor(flags, dtype=torch.bool)
    retain_set_mask = ~retain_set_mask
    retain_set_mask = retain_set_mask.type(torch.long).to(device)

    targets = torch.arange(batch_size).to(torch.long).to(device)

    forget_set_image_loss = (
        forget_set_mask * F.cross_entropy(logits_per_image, targets, reduction="none")
    ).mean()
    forget_set_text_loss = (
        forget_set_mask * F.cross_entropy(logits_per_text, targets, reduction="none")
    ).mean()

    retain_set_image_loss = (
        retain_set_mask * F.cross_entropy(logits_per_image, targets, reduction="none")
    ).mean()
    retain_set_text_loss = (
        retain_set_mask * F.cross_entropy(logits_per_text, targets, reduction="none")
    ).mean()

    kl_div_image_loss = 0
    kl_div_text_loss = 0
    if outputs_from_model_ori:
        logits_per_image_from_model_ori = outputs_from_model_ori["logits_per_image"]
        logits_per_text_from_model_ori = outputs_from_model_ori["logits_per_text"]
        probs_per_image = F.softmax(logits_per_image, dim=-1)
        probs_per_text = F.softmax(logits_per_text, dim=-1)
        probs_per_image_from_model_ori = F.softmax(
            logits_per_image_from_model_ori, dim=-1
        )
        probs_per_text_from_model_ori = F.softmax(
            logits_per_text_from_model_ori, dim=-1
        )

        kl_div_image_loss = (
            retain_set_mask.unsqueeze(1)
            * F.kl_div(
                torch.log(probs_per_image + EPSILON),
                probs_per_image_from_model_ori,
                reduction="none",
            )
        ).mean()
        kl_div_text_loss = (
            retain_set_mask.unsqueeze(1)
            * F.kl_div(
                torch.log(probs_per_text + EPSILON),
                probs_per_text_from_model_ori,
                reduction="none",
            )
        ).mean()

    return {
        "forget_set_image_loss": forget_set_image_loss,
        "forget_set_text_loss": forget_set_text_loss,
        "retain_set_image_loss": retain_set_image_loss,
        "retain_set_text_loss": retain_set_text_loss,
        "kl_div_image_loss": kl_div_image_loss,
        "kl_div_text_loss": kl_div_text_loss,
    }


def calculate_acc(outputs, flags, settings):
    """
    Computes accuracy-related counts for forget and retain sets.

    Returns:
        dict: True/false counts for classification on forget/retain sets.
    """
    batch_size: int = len(flags)
    logits_per_image, logits_per_text = (
        outputs["logits_per_image"],
        outputs["logits_per_text"],
    )
    device = logits_per_image.device

    forget_set_mask = torch.tensor(flags, dtype=torch.long).to(device)

    retain_set_mask = torch.tensor(flags, dtype=torch.bool)
    retain_set_mask = ~retain_set_mask
    retain_set_mask = retain_set_mask.type(torch.long).to(device)

    targets = torch.arange(batch_size).to(torch.long).to(device)

    # Let's calculate some metrics
    true_cnt_forget_image = (
        forget_set_mask
        * (torch.argmax(logits_per_image, dim=1) != targets).type(torch.int32)
    ).sum()
    false_cnt_forget_image = (
        forget_set_mask
        * (torch.argmax(logits_per_image, dim=1) == targets).type(torch.int32)
    ).sum()
    true_cnt_forget_text = (
        forget_set_mask
        * (torch.argmax(logits_per_text, dim=1) != targets).type(torch.int32)
    ).sum()
    false_cnt_forget_text = (
        forget_set_mask
        * (torch.argmax(logits_per_text, dim=1) == targets).type(torch.int32)
    ).sum()

    curr_cnt_forget_samples = forget_set_mask.sum().item()

    true_cnt_retain_image = (
        retain_set_mask
        * (torch.argmax(logits_per_image, dim=1) == targets).type(torch.int32)
    ).sum()
    false_cnt_retain_image = (
        retain_set_mask
        * (torch.argmax(logits_per_image, dim=1) != targets).type(torch.int32)
    ).sum()
    true_cnt_retain_text = (
        retain_set_mask
        * (torch.argmax(logits_per_text, dim=1) == targets).type(torch.int32)
    ).sum()
    false_cnt_retain_text = (
        retain_set_mask
        * (torch.argmax(logits_per_text, dim=1) != targets).type(torch.int32)
    ).sum()

    curr_cnt_retain_samples = retain_set_mask.sum().item()

    return {
        "true_cnt_forget_image": true_cnt_forget_image,
        "false_cnt_forget_image": false_cnt_forget_image,
        "true_cnt_forget_text": true_cnt_forget_text,
        "false_cnt_forget_text": false_cnt_forget_text,
        "curr_cnt_forget_samples": curr_cnt_forget_samples,
        "true_cnt_retain_image": true_cnt_retain_image,
        "false_cnt_retain_image": false_cnt_retain_image,
        "true_cnt_retain_text": true_cnt_retain_text,
        "false_cnt_retain_text": false_cnt_retain_text,
        "curr_cnt_retain_samples": curr_cnt_retain_samples,
    }


class RecallMetrics:
    """
    Helper class to compute Recall@K metrics.
    """
    def __init__(self, name, topk):
        self.name = name
        self.topk = topk
        self.topk_cnts = {k: 0 for k in topk}
        self.num_samples = 0

    def add_cnt_by_key(self, key, count):
        self.topk_cnts[key] += count

    def keys(self):
        return self.topk

    def export(self) -> Dict[str, float]:
        return {f"Recall@{key}": value for key, value in self.topk_rate.items()}

    def average(self) -> float:
        return sum(self.topk_rate.values())

    def __str__(self):
        return f"Metrics of {self.name}: {self.topk_rate}"

    def __repr__(self):
        return str(self)

    @property
    def topk_rate(self):
        topk_rate = {
            k: self.topk_cnts[k] / self.num_samples if self.num_samples > 0 else 0
            for k in self.topk
        }
        return topk_rate


def calculate_metrics(similarity_matrix, all_flags):
    """
    Calculates Recall@K for forget and retain sets separately.

    Args:
        similarity_matrix (Tensor): Similarity scores between queries and targets.
        all_flags (List[bool]): Boolean flags for forget set membership.

    Returns:
        Tuple[RecallMetrics, RecallMetrics]: Recall metrics for forget and retain sets.
    """
    recall_metrics_forget = RecallMetrics("Forget Set Recall", [1, 5, 10])
    recall_metrics_retain = RecallMetrics("Retain Set Recall", [1, 5, 10])

    num_samples = similarity_matrix.size(0)
    targets = torch.arange(num_samples)
    flags = torch.tensor(all_flags).bool()


    for k in recall_metrics_forget.keys():
        _, topk_indices = similarity_matrix.topk(k=k, dim=-1)
        correct = topk_indices.eq(targets.unsqueeze(1))

        # Forget set metrics
        forget_correct = correct[flags]
        forget_hits = forget_correct.any(dim=1).sum().item()
        recall_metrics_forget.add_cnt_by_key(k, forget_hits)
        recall_metrics_forget.num_samples = flags.sum().item()

        # Retain set metrics
        retain_correct = correct[~flags]
        retain_hits = retain_correct.any(dim=1).sum().item()
        recall_metrics_retain.add_cnt_by_key(k, retain_hits)
        recall_metrics_retain.num_samples = (~flags).sum().item()

    return recall_metrics_forget, recall_metrics_retain


def train_one_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    tensorboard_writer,
    settings,
    checkpoint_helper,
    model_ori=None,
):
    """
    Trains the model for a single epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): Dataloader for training set.
        optimizer (Optimizer): Optimizer instance.
        tensorboard_writer (SummaryWriter): TensorBoard writer.
        settings (dict): Experiment configuration.
        checkpoint_helper (CheckpointHelper): Helper to save best models.
        model_ori (nn.Module, optional): Original model for KL divergence comparison.
    """
    mode: str = "train"
    device = torch.device(settings["misc"]["device"])
    epochs: int = settings[mode]["epochs"]
    checkpoint_template: Template = Template(
        os.path.join(settings["train"]["checkpoint"], "${loss}.pt")
    )

    model.train()
    pbar = tqdm(train_loader, total=len(train_loader), ncols=80)
    total_loss_epoch = 0

    optimizer.zero_grad()

    for batch in pbar:

        # flags: is in forget set.
        images, texts, flags = batch
        batch_size = len(flags)

        # Forward pass
        outputs = model(images, texts)

        if model_ori:
            model_ori.eval()
            with torch.no_grad():
                outputs_from_model_ori = model_ori(images, texts)
            loss_dict = calculate_loss(outputs, flags, settings, outputs_from_model_ori)
        else:
            loss_dict = calculate_loss(outputs, flags, settings)

        # Get losses and apply weights
        forget_set_image_loss = loss_dict["forget_set_image_loss"] * settings["loss"][
            "weights"
        ].get("forget_set_image_loss", 1)
        forget_set_text_loss = loss_dict["forget_set_text_loss"] * settings["loss"][
            "weights"
        ].get("forget_set_text_loss", 1)
        retain_set_image_loss = loss_dict["retain_set_image_loss"] * settings["loss"][
            "weights"
        ].get("retain_set_image_loss", 1)
        retain_set_text_loss = loss_dict["retain_set_text_loss"] * settings["loss"][
            "weights"
        ].get("retain_set_text_loss", 1)
        kl_div_image_loss = loss_dict["kl_div_image_loss"] * settings["loss"][
            "weights"
        ].get("kl_div_image_loss", 1)
        kl_div_text_loss = loss_dict["kl_div_text_loss"] * settings["loss"][
            "weights"
        ].get("kl_div_text_loss", 1)

        total_loss = (
            -forget_set_image_loss
            - forget_set_text_loss
            + retain_set_image_loss
            + retain_set_text_loss
            + kl_div_image_loss
            + kl_div_text_loss
        )
        total_loss_epoch += total_loss.item()

        pbar.set_description(
            f"Train: Epoch {epoch}/{epochs}, Loss: {total_loss.item():.4f}"
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss_epoch = total_loss_epoch / len(train_loader)
    tensorboard_writer.add_scalar("Training loss", avg_loss_epoch, epoch)
    tensorboard_writer.add_scalar("Forget Set Image Loss", forget_set_image_loss, epoch)
    tensorboard_writer.add_scalar("Forget Set Text Loss", forget_set_text_loss, epoch)
    tensorboard_writer.add_scalar("Retain Set Image Loss", retain_set_image_loss, epoch)
    tensorboard_writer.add_scalar("Retain Set Text Loss", retain_set_text_loss, epoch)
    tensorboard_writer.add_scalar("KL Divergence Image Loss", kl_div_image_loss, epoch)
    tensorboard_writer.add_scalar("KL Divergence Text Loss", kl_div_text_loss, epoch)


def train(model, train_loader, optimizer, settings, checkpoint_helper, val_loader=None):
    """
    Trains the model for multiple epochs and optionally evaluates.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        settings (dict): Training settings.
        checkpoint_helper (CheckpointHelper): Model saving helper.
        val_loader (DataLoader, optional): Optional validation loader.
    """
    mode: str = "train"
    epochs: int = settings[mode]["epochs"]

    writer = SummaryWriter(settings["train"]["tensorboard"])

    model_ori = copy.deepcopy(model)

    for epoch in range(epochs):
        train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            writer,
            settings,
            checkpoint_helper,
            model_ori,
        )
        if val_loader:
            test(model, val_loader, settings, checkpoint_helper, epoch)

    writer.close()


def extract_embeddings(model, dataloader, device):
    """
    Extracts image and text embeddings from a given model.

    Args:
        model (nn.Module): The trained model with processor and CLIP model.
        dataloader (DataLoader): Data loader.
        device (torch.device): Device to run inference on.

    Returns:
        Tuple[Tensor, Tensor, List]: Image embeddings, text embeddings, and forget flags.
    """
    image_embeddings = []
    text_embeddings = []
    flags = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            images, texts, batch_flags = batch
            inputs = model.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)
            outputs = model.clip_model(**inputs)
            image_embeddings.append(outputs.image_embeds.cpu())
            text_embeddings.append(outputs.text_embeds.cpu())
            flags.extend(batch_flags)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    return image_embeddings, text_embeddings, flags


def compute_similarity_matrix_in_batches(
    query_embeddings, gallery_embeddings, device, batch_size=1024
):
    """
    Computes similarity matrix in batches to reduce memory usage.

    Args:
        query_embeddings (Tensor): Embeddings of queries (e.g., images or texts).
        gallery_embeddings (Tensor): Embeddings to compare against.
        device (torch.device): Device to compute on.
        batch_size (int): Batch size for processing.

    Returns:
        Tensor: Full similarity matrix.
    """
    num_queries = query_embeddings.size(0)
    num_gallery = gallery_embeddings.size(0)
    similarity_matrix = torch.zeros((num_queries, num_gallery))

    for i in tqdm(
        range(0, num_queries, batch_size), desc="Computing Similarity Matrix"
    ):
        query_batch = query_embeddings[i : i + batch_size].to(device)
        similarities = []
        for j in range(0, num_gallery, batch_size):
            gallery_batch = gallery_embeddings[j : j + batch_size].to(device)
            sim = query_batch @ gallery_batch.t()
            similarities.append(sim.cpu())
            del gallery_batch
            torch.cuda.empty_cache()
        similarities = torch.cat(similarities, dim=1)
        similarity_matrix[i : i + batch_size] = similarities
        del query_batch
        torch.cuda.empty_cache()

    return similarity_matrix


def test(
    model: Any,
    dataloader: DataLoader,
    settings: Dict[str, Any],
    checkpoint_helper: CheckpointHelper,
    epoch: int,
):
    """
    Runs retrieval evaluation and logs Recall@K for forget and retain sets.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Evaluation data loader.
        settings (dict): Experiment config.
        checkpoint_helper (CheckpointHelper): Model-saving helper.
        epoch (int): Current epoch index.
    """
    device = torch.device(settings["misc"]["device"])
    mode = settings["mode"]
    batch_size = settings[mode]["batch_size"]
    shuffle: bool = settings["dataloader"]["shuffle"]
    num_workers: int = settings["dataloader"]["num_workers"]

    writer = SummaryWriter(settings["train"]["tensorboard"])

    checkpoint_template: Template = Template(
        os.path.join(settings["train"]["checkpoint"], "recall_${avg_recall}.pt")
    )

    model.eval()

    # Step 1: extract embeddings
    image_embeddings, text_embeddings, flags = extract_embeddings(
        model, dataloader, device
    )

    image_embeddings = image_embeddings.to(device)
    text_embeddings = text_embeddings.to(device)

    # Step 2: compute similarity matrix
    similarity_matrix_image_query = compute_similarity_matrix_in_batches(
        image_embeddings, text_embeddings, device, batch_size=1024
    )


    similarity_matrix_text_query = compute_similarity_matrix_in_batches(
        text_embeddings, image_embeddings, device, batch_size=1024
    )
    # Step 3: compute metrics for both directions
    recall_metrics_image_forget, recall_metrics_image_retain = calculate_metrics(
        similarity_matrix_image_query, flags
    )
    recall_metrics_text_forget, recall_metrics_text_retain = calculate_metrics(
        similarity_matrix_text_query, flags
    )
     # Step 4: log metrics
    print("Image Retrieval Metrics for Forget Set:")
    print(recall_metrics_image_forget)
    print("Image Retrieval Metrics for Retain Set:")
    print(recall_metrics_image_retain)

    print("Text Retrieval Metrics for Forget Set:")
    print(recall_metrics_text_forget)
    print("Text Retrieval Metrics for Retain Set:")
    print(recall_metrics_text_retain)

    for key, value in recall_metrics_image_forget.export().items():
        writer.add_scalar(
            f"Image Retrieval Metrics for Forget Set: {key}", value, epoch
        )
    for key, value in recall_metrics_image_retain.export().items():
        writer.add_scalar(
            f"Image Retrieval Metrics for Retain Set: {key}", value, epoch
        )
    for key, value in recall_metrics_text_forget.export().items():
        writer.add_scalar(f"Text Retrieval Metrics for Forget Set: {key}", value, epoch)
    for key, value in recall_metrics_text_retain.export().items():
        writer.add_scalar(f"Text Retrieval Metrics for Retain Set: {key}", value, epoch)

    avg_recall = (
        sum(
            (
                recall_metrics_image_forget.average(),
                recall_metrics_image_retain.average(),
                recall_metrics_text_forget.average(),
                recall_metrics_text_retain.average(),
            )
        )
        / 4.0
    )
    # Save checkpoint if avg_recall is the best so far
    checkpoint_helper.save_model_when_max_recall(
        avg_recall, model, checkpoint_template.substitute(avg_recall=avg_recall)
    )
