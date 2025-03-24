import os
import argparse

import torch
import wandb

from src.model import FCN
from src.loops import train_model, test_model
from src.dataset import get_class_names, get_dataloader, Mode


def get_project_name(variant: str, freeze_backbone: bool):
    return f"road-segmentation-{variant}-{'freeze' if freeze_backbone else 'unfreeze'}"


def main(
    data_dir: str,
    batch_size: int,
    variant: str,
    freeze_backbone: bool,
    num_epochs: int,
    lr: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = get_dataloader(data_dir, Mode("train"), batch_size=batch_size)
    valid_dataloader = get_dataloader(data_dir, Mode("valid"), batch_size=batch_size)

    classes = get_class_names()
    model = FCN(variant=variant, num_classes=len(classes)).to(device)
    if freeze_backbone:
        model.freeze_backbone()
    proj_name = get_project_name(variant, freeze_backbone)

    ckpt_path = os.path.join("ckpts", f"{proj_name}.pth")

    run = wandb.init(project="cv-a4", name=proj_name)
    train_model(
        model,
        train_dataloader,
        valid_dataloader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        ckpt_path=ckpt_path,
    )

    test_dataloader = get_dataloader(data_dir, Mode("test"), batch_size=batch_size)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_model(model, test_dataloader, device)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("road segmentation")
    parser.add_argument("--data_dir", type=str, default="../data/dataset_224")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--variant", type=str, default="fcn8s")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    main(
        args.data_dir,
        args.batch_size,
        args.variant,
        args.freeze_backbone,
        args.num_epochs,
        args.lr,
    )
