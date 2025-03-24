import time

import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model import FCN


def _validate(model: FCN, dataloader: DataLoader, device: torch.device) -> float:
    # TODO: divide the loss by total instances instead of dataloader len
    # NOTE: leaving it in train to get loss output.
    #   wrapping in a no_grad to ensure no learning

    model.train()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="validating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            val_loss += loss.item()  # type: ignore

    return val_loss / len(dataloader)


def _train_epoch(
    model: FCN,
    optimizer,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0

    for images, targets, _ in tqdm(dataloader, desc="training"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()

        epoch_loss += loss.item()  # type: ignore

    return epoch_loss / len(dataloader)


def train_model(
    model: FCN,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    ckpt_path: str,
) -> None:
    optimizer = None  # TODO: add optimizer

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = _train_epoch(model, optimizer, train_dataloader, device)
        val_loss = _validate(model, val_dataloader, device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        time_taken = time.time() - start_time
        print(
            f"epoch {epoch + 1}/{num_epochs} : train Loss: {train_loss:.4f} - val Loss: {val_loss:.4f} - time: {time_taken:.2f}s"
        )

        torch.save(model.state_dict(), ckpt_path)
