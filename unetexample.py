import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn

### Config ###

DATA_DIR = r"/Users/pyn/Downloads/finalproject475-main/Raster_Burn_Data"
DATA_DIR2 = r"/Users/pyn/Downloads/finalproject475-main/Raster_Burn_Mit"
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-4
PATCH_SIZE = 128
PATCHES_PER_IMAGE = 8
#try 8, 16, 32


### Data loading ###
def load_raster_as_arrays(file_path):
    #load geotiff and split
    #y = last fire shape band
    #input = all other bands, preceding fire shapes + environmental

    with rasterio.open(file_path) as src:
        data = src.read().astype(np.float32)  # bands, H, W

    num_bands = data.shape[0]
    last_fire = None

    # Find the first environmental band; Elevation
    for i in range(num_bands):
        band_max = data[i].max()
        if band_max > 1:
            last_fire = i - 1  # the band before this is the last fire day
            break

    # split data
    target = data[last_fire:last_fire + 1]  # 1, H, W
    inputs = np.concatenate([data[:last_fire], data[last_fire + 1:]], axis=0)  # all other bands

    # print(f"Last fire mask band: {last_fire_idx}")
    # print(f"Input bands shape: {inputs.shape}, Target shape: {target.shape}")

    # z score normalization
    mean = inputs.mean(axis=(1, 2), keepdims=True)
    std = inputs.std(axis=(1, 2), keepdims=True) + 1e-6
    inputs = (inputs - mean) / std

    return inputs, target


class WildfireRasterDataset(Dataset):
    # compute total min band count
    # use as input channel
    # compute accumulative patch HxW from smallest

    def __init__(self, data_dir, patch_size=None, patches_per_image=1):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".tif")]
        self.file_list.sort()

        if len(self.file_list) == 0:
            raise RuntimeError(f"No .tif files found in {data_dir}")

        self.patch_size_requested = patch_size
        self.patches_per_image = patches_per_image if patch_size is not None else 1

        # find min band and HxW
        band_counts = []
        heights = []
        widths = []

        for f in self.file_list:
            path = os.path.join(self.data_dir, f)
            with rasterio.open(path) as src:
                band_counts.append(src.count)
                heights.append(src.height)
                widths.append(src.width)

        self.min_total_bands = min(band_counts)
        self.n_input_channels = self.min_total_bands - 1

        self.min_height = min(heights)
        self.min_width = min(widths)

        # patch size for all tif
        if self.patch_size_requested is not None:
            self.patch_h = min(self.patch_size_requested, self.min_height)
            self.patch_w = min(self.patch_size_requested, self.min_width)
        else:
            self.patch_h = None
            self.patch_w = None

        if self.patch_h is not None:
            print(f"Globla patch size: {self.patch_h} x {self.patch_w}")

    def __len__(self):
        # expand length by patches_per_image per file if using patches
        if self.patch_h is not None:
            return len(self.file_list) * self.patches_per_image
        else:
            return len(self.file_list)

    def __getitem__(self, idx):
        if self.patch_h is not None:
            img_idx = idx // self.patches_per_image  # map idx to image_idx and patch_idx
        else:
            img_idx = idx

        file_name = self.file_list[img_idx]
        file_path = os.path.join(self.data_dir, file_name)

        x, y = load_raster_as_arrays(file_path)

        if x.shape[0] > self.n_input_channels:  # cut channels to the minimum across the dataset
            x = x[:self.n_input_channels, :, :]

        if self.patch_h is not None:
            C, H, W = x.shape
            ph = self.patch_h
            pw = self.patch_w

            # if some file are smaller thane patch size
            ph = min(ph, H)
            pw = min(pw, W)

            # random crop
            top = 0 if H == ph else np.random.randint(0, H - ph + 1)
            left = 0 if W == pw else np.random.randint(0, W - pw + 1)

            x = x[:, top:top + ph, left:left + pw]
            y = y[:, top:top + ph, left:left + pw]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y


### U-Net model ###
class DoubleConv(nn.Module):
    # conv to BN to ReLu x 2
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def center_crop(enc_feat, target_feat): #ensuing fit of encoder to decoder images for skip connect
    # crop to HxW
    _, _, H_enc, W_enc = enc_feat.shape
    _, _, H_trg, W_trg = target_feat.shape

    delta_h = H_enc - H_trg
    delta_w = W_enc - W_trg

    top = max(delta_h // 2, 0)
    left = max(delta_w // 2, 0)

    return enc_feat[:, :, top:top + H_trg, left:left + W_trg]


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        # final 1x1 conv
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        # bottleneck
        bn = self.bottleneck(p4)

        # decoder with center cropping for skip connections
        u4 = self.up4(bn)
        c4_crop = center_crop(c4, u4)
        u4 = torch.cat([u4, c4_crop], dim=1)
        c5 = self.dec4(u4)

        u3 = self.up3(c5)
        c3_crop = center_crop(c3, u3)
        u3 = torch.cat([u3, c3_crop], dim=1)
        c6 = self.dec3(u3)

        u2 = self.up2(c6)
        c2_crop = center_crop(c2, u2)
        u2 = torch.cat([u2, c2_crop], dim=1)
        c7 = self.dec2(u2)

        u1 = self.up1(c7)
        c1_crop = center_crop(c1, u1)
        u1 = torch.cat([u1, c1_crop], dim=1)
        c8 = self.dec1(u1)

        out = self.final_conv(c8)  # logits
        return out


### Losses & metrics ###
def dice_loss(logits, targets, eps=1e-6):
    # soft dice loss qnd binary target
    probs = torch.sigmoid(logits)
    # reshape instead of view to handle distant tensors
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5, eps=1e-6):
    # return F1 metrics
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.reshape(-1)  # Use reshape instead of view
    targets = targets.reshape(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return float(precision), float(recall), float(f1)


def estimate_pos_weight(dataset, max_samples=None):
    # compute positive weight from dataset
    total_pos = 0
    total_neg = 0

    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for i in range(n):
        _, y = dataset[i]  # y = (1, H, W)
        y_np = y.numpy()
        pos = (y_np == 1).sum()
        neg = (y_np == 0).sum()
        total_pos += pos
        total_neg += neg

    total_pos = max(total_pos, 1)
    pos_weight = total_neg / total_pos
    return float(pos_weight)


def focal_loss(logits, targets, alpha=1.0, gamma=2.0, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    # BCE
    bce = -(targets * torch.log(probs + eps) + (1 - targets) * torch.log(1 - probs + eps))

    # Focal weight
    pt = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - pt) ** gamma
    loss = alpha * focal_weight * bce

    return loss.mean()


def tversky_loss(logits, targets, alpha=0.7, beta=0.3, eps=1e-6):
    # alpha = weight for false negatives
    # beta = weight for false positives
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    TP = (probs * targets).sum(dim=1)
    FN = ((1 - probs) * targets).sum(dim=1)
    FP = (probs * (1 - targets)).sum(dim=1)

    tversky = (TP + eps) / (TP + alpha * FN + beta * FP + eps)
    return 1.0 - tversky.mean()


### Training / evaluation ###
def train_one_epoch(model, loader, optimizer, bce_loss, device, dice_weight=0.5):
    model.train()
    epoch_loss = 0.0

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        logits = model(x)

        # target matches logits size
        if logits.shape[-2:] != y.shape[-2:]:
            y = center_crop(y, logits)

        bce = bce_loss(logits, y)
        dl = dice_loss(logits, y)
        loss = (1.0 - dice_weight) * bce + dice_weight * dl

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)

    return epoch_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, bce_loss, device, dice_weight=0.5):
    model.eval()
    epoch_loss = 0.0
    f1_sum = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        logits = model(x)

        # target matches logits size
        if logits.shape[-2:] != y.shape[-2:]:
            y = center_crop(y, logits)

        bce = bce_loss(logits, y)
        dl = dice_loss(logits, y)
        loss = (1.0 - dice_weight) * bce + dice_weight * dl
        epoch_loss += loss.item() * x.size(0)

        _, _, f1 = compute_metrics(logits, y)
        f1_sum += f1
        count += 1

    return epoch_loss / len(loader.dataset), f1_sum / max(count, 1)


def train_and_evaluate_for_losses(
        loss_dict, dataset, in_channels, device, patch_size=128, epochs=5, test_split=0.2, eval_test_each_epoch=True):
    results = {}

    # Train, Val, Test Split
    total_len = len(dataset)
    test_len = int(test_split * total_len)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len - test_len

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for loss_name, loss_fn in loss_dict.items():
        print(f"\n=== Training with loss: {loss_name} ===\n")

        model = UNet(in_channels=in_channels, out_channels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        train_losses, val_losses, val_f1_history = [], [], []
        test_losses_epoch, test_f1_history = [], []

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

            val_loss, val_f1 = evaluate(model, val_loader, loss_fn, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1_history.append(val_f1)

            if eval_test_each_epoch:
                test_loss_epoch, test_f1_epoch = evaluate(model, test_loader, loss_fn, device)
                test_losses_epoch.append(test_loss_epoch)
                test_f1_history.append(test_f1_epoch)

            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f}" + (f" | Test F1: {test_f1_epoch:.4f}" if eval_test_each_epoch else ""))
            scheduler.step()

        # final test evaluation if not done each epoch
        if not eval_test_each_epoch:
            test_loss, test_f1 = evaluate(model, test_loader, loss_fn, device)
            test_losses_epoch = [test_loss]
            test_f1_history = [test_f1]

        else:
            # use last epoch test metrics as final
            test_loss, test_f1 = test_losses_epoch[-1], test_f1_history[-1]

        print(f"\n=== FINAL TEST METRICS ({loss_name}) ===\nTest Loss: {test_loss:.4f} | Test F1: {test_f1:.4f}")

        results[loss_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_f1": val_f1_history,
            "test_losses_epoch": test_losses_epoch,
            "test_f1_history": test_f1_history,
            "test_loss": test_loss,
            "test_f1": test_f1
        }

    return results


### Main ###
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WildfireRasterDataset(
        DATA_DIR2,
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE
    )
    in_channels = dataset.n_input_channels

    dice_weight = 0.5

    def combined_loss(logits, targets): #combine best performing loss funcs
        return dice_weight * dice_loss(logits, targets) + (1 - dice_weight) * focal_loss(logits, targets)

    loss_dict = {"Dice+Focal": combined_loss}

    # Train / evaluate
    results = train_and_evaluate_for_losses(
        loss_dict=loss_dict,
        dataset=dataset,
        in_channels=in_channels,
        device=device,
        patch_size=PATCH_SIZE,
        epochs=EPOCHS,
        test_split=0.2
    )

    # plots
    for loss_name, metrics in results.items():
        epochs_range = list(range(1, len(metrics["train_losses"]) + 1))

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # loss curves; left axis
        ax1.plot(epochs_range, metrics["train_losses"], marker='o', label="Train Loss")
        ax1.plot(epochs_range, metrics["val_losses"], marker='o', label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        # F1 scores, right axis
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, metrics["val_f1"], marker='x', linestyle='-', label="Val F1")

        if "test_f1_history" in metrics and len(metrics["test_f1_history"]) == len(epochs_range):
            ax2.plot(epochs_range, metrics["test_f1_history"], marker='s', linestyle='--', label="Test F1 (per-epoch)")
        else:
            ax2.axhline(y=metrics["test_f1"], color='red', linestyle='--',
                        label=f"Final Test F1 = {metrics['test_f1']:.3f}")

        ax2.set_ylabel("F1 Score")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="upper right")

        plt.title(f"{loss_name} Training Summary")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
