import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader

from src.unet.unet import UNet


def visualize_predictions(
    model: UNet, dataloader: DataLoader, device: torch.device, res_dir: str
):
    model.eval()
    model.to(device)

    # get a batch from the dataloader
    images, ground_truth_masks = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    predictions = torch.argmax(outputs, dim=1)

    images = images.cpu()
    ground_truth_masks = ground_truth_masks.cpu()
    predictions = predictions.cpu()

    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0)

        true_mask = ground_truth_masks[i]
        pred_mask = predictions[i]

        _, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(img)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(true_mask, cmap="viridis")
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        axs[2].imshow(pred_mask, cmap="viridis")
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(res_dir, f"pred_{i}.png")
        plt.savefig(save_path)
