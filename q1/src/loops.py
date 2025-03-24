import time

import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model import FCN


def _validate(
    model: FCN, dataloader: DataLoader, criterion: torch.nn.Module, device: torch.device
) -> float:
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()  # type: ignore

    num_items = len(dataloader.dataset)  # type: ignore
    return val_loss / num_items


def _train_epoch(
    model: FCN,
    optimizer,
    dataloader: DataLoader,
    criterion: torch.nn.Module,  # TODO: fix type
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0

    for images, labels in tqdm(dataloader, desc="training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()

        epoch_loss += loss.item()  # type: ignore

    num_items = len(dataloader.dataset)  # type: ignore
    return epoch_loss / num_items


def train_model(
    model: FCN,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    ckpt_path: str,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = _train_epoch(model, optimizer, train_dataloader, criterion, device)
        val_loss = _validate(model, val_dataloader, criterion, device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        time_taken = time.time() - start_time
        print(
            f"epoch {epoch + 1}/{num_epochs} : train Loss: {train_loss:.4f} - val Loss: {val_loss:.4f} - time: {time_taken:.2f}s"
        )

        torch.save(model.state_dict(), ckpt_path)
