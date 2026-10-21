import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import copy
import os

from pathlib import Path
from string import Template

from vis_utils import generate_and_save_gradcam

logging.basicConfig(level=logging.INFO)

EPSILON = 1e-6


def calculate_loss(outputs, flags, settings, outputs_from_model_ori=None):
    """
    Computes multi-component loss for both forget and retain sets,
    including optional KL divergence from original model outputs.

    Args:
        outputs (dict): Output logits from the current model.
        flags (List[bool]): List of binary flags indicating if sample is in the forget set.
        settings (dict): Dictionary containing training configurations and loss weights.
        outputs_from_model_ori (dict, optional): Logits from the original model for KL divergence computation.

    Returns:
        dict: Dictionary of individual loss components:
            - forget_set_image_loss
            - forget_set_text_loss
            - retain_set_image_loss
            - retain_set_text_loss
            - kl_div_image_loss
            - kl_div_text_loss
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
        logits_per_image_from_model_ori, logits_per_text_from_model_ori = (
            outputs_from_model_ori["logits_per_image"],
            outputs_from_model_ori["logits_per_text"],
        )
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
    Calculates retrieval accuracy metrics by comparing predictions to targets.

    Args:
        outputs (dict): Logits output by the model.
        flags (List[bool]): List indicating forget set membership.
        settings (dict): Training or evaluation configuration.

    Returns:
        dict: Accuracy counts for both forget and retain sets:
            - true/false counts for image and text
            - number of forget/retain samples
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
    Trains the model for a single epoch and logs intermediate losses.

    Args:
        epoch (int): Current epoch index.
        model (nn.Module): Model being trained.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        tensorboard_writer (SummaryWriter): TensorBoard writer for logging.
        settings (dict): Configuration with training/loss settings.
        checkpoint_helper (CheckpointHelper): Utility for saving model checkpoints.
        model_ori (nn.Module, optional): Original model for KL divergence computation.
    """
    mode: str = "train"
    device = torch.device(settings["misc"]["device"])
    epochs: int = settings[mode]["epochs"]

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
        kl_div_image_loss = loss_dict.get("kl_div_image_loss", 0) * settings["loss"][
            "weights"
        ].get("kl_div_image_loss", 1)
        kl_div_text_loss = loss_dict.get("kl_div_text_loss", 0) * settings["loss"][
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
    mode: str = "train"
    epochs: int = settings[mode]["epochs"]

    writer = SummaryWriter(settings["train"]["tensorboard"])

    model_ori = copy.deepcopy(model)

    for epoch in range(epochs):
        train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            tensorboard_writer=writer,
            settings=settings,
            checkpoint_helper=checkpoint_helper,
            model_ori=model_ori,
        )
        if val_loader:
            # generate_and_save_gradcam(copy.deepcopy(model), copy.deepcopy(val_loader.dataset), settings)
            test(model, val_loader, settings, checkpoint_helper, epoch)

    writer.close()


def test(model, dataloader, settings, checkpoint_helper, epoch):
    mode: str = "eval"
    device = torch.device(settings["misc"]["device"])

    writer = SummaryWriter(settings["train"]["tensorboard"])

    checkpoint_template: Template = Template(
        os.path.join(settings["train"]["checkpoint"], "acc_${avg_acc}.pt")
    )

    model.eval()
    pbar = tqdm(dataloader, total=len(dataloader))
    total_loss_epoch = 0
    total_samples = 0
    total_cnt_forget_samples = 0
    total_cnt_retain_samples = 0
    total_cnt_forget_true = 0
    total_cnt_retain_true = 0
    total_img_ret_forget_true = 0
    total_text_ret_forget_true = 0
    total_img_ret_retain_true = 0
    total_text_ret_retain_true = 0

    with torch.no_grad():
        for batch in pbar:
            images, texts, flags = batch
            batch_size = len(flags)

            # Forward pass
            outputs = model(images, texts)
            logits_per_image, logits_per_text = (
                outputs["logits_per_image"],
                outputs["logits_per_text"],
            )

            # loss_dict = calculate_loss(logits, flags)
            loss_dict = calculate_loss(outputs, flags, settings)
            forget_set_image_loss = loss_dict["forget_set_image_loss"] * settings[
                "loss"
            ]["weights"].get("forget_set_image_loss", 1)
            forget_set_text_loss = loss_dict["forget_set_text_loss"] * settings["loss"][
                "weights"
            ].get("forget_set_text_loss", 1)
            retain_set_image_loss = loss_dict["retain_set_image_loss"] * settings[
                "loss"
            ]["weights"].get("retain_set_image_loss", 1)
            retain_set_text_loss = loss_dict["retain_set_text_loss"] * settings["loss"][
                "weights"
            ].get("retain_set_text_loss", 1)
            kl_div_image_loss = loss_dict.get("kl_div_image_loss", 0) * settings[
                "loss"
            ]["weights"].get("kl_div_image_loss", 1)
            kl_div_text_loss = loss_dict.get("kl_div_text_loss", 0) * settings["loss"][
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

            acc_dict = calculate_acc(outputs, flags, settings)

            curr_cnt_forget_samples = acc_dict["curr_cnt_forget_samples"]
            true_cnt_forget_image = acc_dict["true_cnt_forget_image"]
            true_cnt_forget_text = acc_dict["true_cnt_forget_text"]
            curr_cnt_retain_samples = acc_dict["curr_cnt_retain_samples"]
            true_cnt_retain_image = acc_dict["true_cnt_retain_image"]
            true_cnt_retain_text = acc_dict["true_cnt_retain_text"]

            total_cnt_forget_samples += curr_cnt_forget_samples

            total_cnt_retain_samples += curr_cnt_retain_samples

            total_samples += batch_size * 4

            # Let's compute retrevel accuracy

            total_img_ret_forget_true += true_cnt_forget_image
            total_text_ret_forget_true += true_cnt_forget_text
            total_img_ret_retain_true += true_cnt_retain_image
            total_text_ret_retain_true += true_cnt_retain_text

            pbar.set_description(
                f"Test / Val: Epoch 1 / 1, Loss: {total_loss.item():.4f}, Forget Set: {true_cnt_forget_image + true_cnt_forget_text} / {curr_cnt_forget_samples * 2}, Retain Set: {true_cnt_retain_image + true_cnt_retain_text} / {curr_cnt_retain_samples * 2}"
            )

        # Let's compute retrevel accuracy
        total_image_ret_forget_set = (
            total_img_ret_forget_true / total_cnt_forget_samples
        )
        total_image_ret_retain_set = (
            total_img_ret_retain_true / total_cnt_retain_samples
        )
        total_text_ret_forget_set = (
            total_text_ret_forget_true / total_cnt_forget_samples
        )
        total_text_ret_retain_set = (
            total_text_ret_retain_true / total_cnt_retain_samples
        )

        logging.info(f"Total loss: {total_loss}")

        # Let's add some print value for retrevel
        logging.info(
            f"Total Image Retrieval/0-shot accuracy in forget set: {total_image_ret_forget_set * 100} %"
        )
        logging.info(
            f"Total Image Retrieval/0-shot accuracy in retain set: {total_image_ret_retain_set * 100} %"
        )
        logging.info(
            f"Total Text Retrieval accuracy in forget set: {total_text_ret_forget_set * 100} %"
        )
        logging.info(
            f"Total Text Retrieval accuracy in retain set: {total_text_ret_retain_set * 100} %"
        )

        writer.add_scalar("Test Loss", total_loss, epoch)
        # Let's add some print value for retrevel
        writer.add_scalar(
            "Test Image Retrieval/0-shot Accuracy per Forget Set",
            total_image_ret_forget_set,
            epoch,
        )
        writer.add_scalar(
            "Test Image Retrieval/0-shot Accuracy per Retain Set",
            total_image_ret_retain_set,
            epoch,
        )
        writer.add_scalar(
            "Test Text Retrieval Accuracy per Forget Set",
            total_text_ret_forget_set,
            epoch,
        )
        writer.add_scalar(
            "Test Text Retrieval Accuracy per Retain Set",
            total_text_ret_retain_set,
            epoch,
        )

        avg_acc = (
            total_image_ret_forget_set
            + total_image_ret_retain_set
            + total_text_ret_forget_set
            + total_text_ret_retain_set
        ) / 4.0
        checkpoint_helper.save_model_when_max_recall(
            total_loss,
            model,
            checkpoint_template.substitute(avg_acc=float(avg_acc.item())),
        )
        del avg_acc
